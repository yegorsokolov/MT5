"""Orchestrate the full MT5 lifecycle from a single command.

The pipeline executes the training, backtesting, strategy creation and
realtime stages sequentially so operators can simply run ``python -m mt5`` and
obtain a fully prepared environment.  The implementation intentionally reuses
the legacy command line entry points to avoid duplicating CLI behaviour while
also exposing a lightweight caching layer for backtest artifacts.  Cached
metrics are reused when their estimated data quality exceeds a configurable
threshold; otherwise a fresh backtest is executed and the artifacts are
rewritten.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import runpy
import shlex
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

try:  # Optional dependency during some tests
    import numpy as np
except Exception:  # pragma: no cover - numpy may be unavailable
    np = None  # type: ignore

try:
    from mt5.log_utils import LOG_DIR, setup_logging
except Exception:  # pragma: no cover - optional dependency may be missing
    LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    def setup_logging() -> None:
        logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


PIPELINE_ARTIFACT_SUBDIR = "pipeline"
BACKTEST_ARTIFACT_NAME = "backtest_metrics.json"
STRATEGY_ARTIFACT_NAME = "strategy.json"


def _serialise(obj: Any) -> Any:
    """Convert ``obj`` into a JSON serialisable structure."""

    if isinstance(obj, Mapping):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_serialise(v) for v in obj]
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if np is not None and isinstance(obj, np.generic):  # type: ignore[arg-type]
        return obj.item()
    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        try:
            return obj.item()
        except Exception:  # pragma: no cover - defensive
            pass
    return obj


def _run_module(module: str, args: Sequence[str]) -> None:
    """Execute ``module`` as ``python -m module`` passing ``args``."""

    previous_argv = sys.argv[:]
    sys.argv = [module, *args]
    try:
        runpy.run_module(module, run_name="__main__")
    finally:
        sys.argv = previous_argv


def _split_args(arg_string: str | None) -> list[str]:
    if not arg_string:
        return []
    return shlex.split(arg_string)


def _load_artifact(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf8") as fh:
            return json.load(fh)
    except Exception:
        logger.warning("Failed to load cached artifact from %s", path, exc_info=True)
        return None


def _save_artifact(path: Path, payload: Mapping[str, Any]) -> None:
    data = _serialise(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
    tmp_path.replace(path)


def _estimate_data_quality(metrics: Mapping[str, Any]) -> float:
    """Heuristic data quality score derived from backtest metrics."""

    trade_count = float(metrics.get("trade_count") or metrics.get("trades") or 0.0)
    skipped = float(metrics.get("skipped_trades") or 0.0)
    if trade_count <= 0:
        return 0.0
    score = 1.0 - min(1.0, max(0.0, skipped) / trade_count)
    return max(0.0, min(1.0, score))


def _run_training(train_args: Sequence[str]) -> None:
    logger.info("Starting training stage")
    _run_module("mt5.train", train_args)
    logger.info("Training stage completed")


def _run_backtest(
    artifact_dir: Path,
    *,
    reuse: bool,
    quality_threshold: float,
    force: bool,
) -> Mapping[str, Any]:
    from utils import load_config

    from mt5.backtest import MODEL_PATH, init_logging as init_backtest_logging
    from mt5.backtest import run_backtest, run_rolling_backtest

    artifact_path = artifact_dir / BACKTEST_ARTIFACT_NAME
    if reuse and not force:
        cached = _load_artifact(artifact_path)
        if cached is not None:
            quality = float(cached.get("quality", 0.0))
            if quality >= quality_threshold:
                logger.info(
                    "Reusing cached backtest metrics from %s (quality %.3f)",
                    artifact_path,
                    quality,
                )
                return cached
            logger.info(
                "Cached backtest quality %.3f below threshold %.3f; recomputing",
                quality,
                quality_threshold,
            )

    logger.info("Starting backtesting stage")
    cfg = load_config()
    cfg_dict = cfg.model_dump() if hasattr(cfg, "model_dump") else dict(cfg)
    init_backtest_logging()

    model = None
    if MODEL_PATH.exists():
        try:
            import joblib

            model = joblib.load(MODEL_PATH)
        except Exception:
            logger.warning("Failed to load cached model from %s", MODEL_PATH, exc_info=True)

    metrics = dict(run_backtest(cfg_dict, model=model))
    rolling = run_rolling_backtest(cfg_dict, model=model)
    if rolling:
        metrics["rolling"] = rolling
    quality = _estimate_data_quality(metrics)
    metrics["quality"] = quality
    metrics["updated_at"] = time.time()
    _save_artifact(artifact_path, metrics)
    logger.info(
        "Backtesting stage completed (quality %.3f, Sharpe %.4f)",
        quality,
        float(metrics.get("sharpe", 0.0)),
    )
    return metrics


def _run_strategy_creation(artifact_dir: Path, *, episodes: int) -> Mapping[str, Any] | None:
    logger.info("Starting strategy creation stage")
    try:
        from models.strategy_graph_controller import train_strategy_graph_controller
    except Exception:  # pragma: no cover - optional dependency missing
        logger.warning("Strategy creation skipped because the graph controller is unavailable", exc_info=True)
        return None

    data = [
        {"price": 1.0, "ma": 0.9},
        {"price": 1.1, "ma": 1.0},
        {"price": 1.2, "ma": 1.1},
        {"price": 1.4, "ma": 1.5},
    ]

    model = train_strategy_graph_controller(data, episodes=episodes)
    _features, summary = model.prepare_graph_inputs(data)
    macro = summary["macro"] if summary.get("has_macro") else None
    best: dict[str, Any] | None = None
    for action in range(getattr(model, "actions", 0)):
        graph = model.build_graph(action, risk=0.5, macro=macro)
        pnl = float(graph.run(data))
        candidate = {
            "action": action,
            "pnl": pnl,
            "graph": graph.to_dict(),
        }
        if best is None or pnl > float(best.get("pnl", float("-inf"))):
            best = candidate

    if best is None:
        logger.warning("Strategy creation did not produce any candidate graphs")
        return None

    artifact = {
        "episodes": episodes,
        "summary": summary,
        "best_strategy": best,
        "updated_at": time.time(),
    }
    artifact_path = artifact_dir / STRATEGY_ARTIFACT_NAME
    _save_artifact(artifact_path, artifact)
    logger.info(
        "Strategy creation completed (action %s, pnl %.4f)",
        best["action"],
        best["pnl"],
    )
    return artifact


async def _run_realtime_async(duration: float | None) -> None:
    from mt5.realtime_train import train_realtime

    if duration is not None and duration > 0:
        task = asyncio.create_task(train_realtime())
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=duration)
        except asyncio.TimeoutError:
            logger.info("Realtime stage reached %.1f seconds; shutting down", duration)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
    else:
        await train_realtime()


def _run_realtime(duration: float | None) -> None:
    logger.info("Starting realtime stage")
    try:
        asyncio.run(_run_realtime_async(duration))
    except Exception:
        logger.exception("Realtime stage failed")
        raise
    logger.info("Realtime stage completed")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run training, backtesting, strategy creation and realtime trading sequentially",
    )
    parser.add_argument(
        "--train-args",
        type=str,
        default="",
        help="Additional arguments forwarded to mt5.train",
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default="",
        help="Directory used to store and reuse pipeline artifacts",
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=0.85,
        help="Minimum quality score required to reuse cached backtest metrics",
    )
    parser.add_argument(
        "--strategy-episodes",
        type=int,
        default=100,
        help="Number of episodes used when training strategy graphs",
    )
    parser.add_argument(
        "--realtime-duration",
        type=float,
        default=5.0,
        help="Maximum number of seconds to run realtime training (0 runs indefinitely)",
    )
    parser.add_argument(
        "--force-backtest",
        action="store_true",
        help="Ignore cached backtest artifacts and recompute metrics",
    )
    parser.add_argument(
        "--no-reuse-backtest",
        action="store_true",
        help="Disable reuse of cached backtest metrics",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip the training stage",
    )
    parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Skip the backtesting stage",
    )
    parser.add_argument(
        "--skip-strategy",
        action="store_true",
        help="Skip the strategy creation stage",
    )
    parser.add_argument(
        "--skip-realtime",
        action="store_true",
        help="Skip the realtime stage",
    )
    return parser


def _resolve_artifact_dir(custom: str | None) -> Path:
    if custom:
        return Path(custom).expanduser().resolve()
    return (LOG_DIR / PIPELINE_ARTIFACT_SUBDIR).resolve()


def main(argv: Sequence[str] | None = None) -> int:
    setup_logging()
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    artifact_dir = _resolve_artifact_dir(args.artifact_dir or None)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_training:
        _run_training(_split_args(args.train_args))
    else:
        logger.info("Training stage skipped via CLI flag")

    backtest_metrics: Mapping[str, Any] | None = None
    if not args.skip_backtest:
        backtest_metrics = _run_backtest(
            artifact_dir,
            reuse=not args.no_reuse_backtest,
            quality_threshold=args.quality_threshold,
            force=args.force_backtest,
        )
    else:
        logger.info("Backtesting stage skipped via CLI flag")

    strategy_artifact: Mapping[str, Any] | None = None
    if not args.skip_strategy:
        strategy_artifact = _run_strategy_creation(
            artifact_dir,
            episodes=args.strategy_episodes,
        )
    else:
        logger.info("Strategy creation stage skipped via CLI flag")

    if not args.skip_realtime:
        _run_realtime(args.realtime_duration if args.realtime_duration > 0 else None)
    else:
        logger.info("Realtime stage skipped via CLI flag")

    if backtest_metrics is not None:
        logger.info(
            "Backtest summary: quality %.3f, sharpe %.4f",
            backtest_metrics.get("quality", 0.0),
            backtest_metrics.get("sharpe", 0.0),
        )
    if strategy_artifact is not None:
        best = strategy_artifact.get("best_strategy", {})
        logger.info(
            "Strategy summary: action %s pnl %.4f",
            best.get("action"),
            best.get("pnl", 0.0),
        )

    return 0


if __name__ == "__main__":  # pragma: no cover - invoked via python -m mt5 pipeline
    sys.exit(main())

