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
import os
import logging
import runpy
import shlex
import subprocess
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

from mt5.run_state import PIPELINE_STAGES, PipelineState
from strategy.archive import StrategyArchive

logger = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
PIPELINE_ARTIFACT_SUBDIR = "pipeline"
BACKTEST_ARTIFACT_NAME = "backtest_metrics.json"
STRATEGY_ARTIFACT_NAME = "strategy.json"
PIPELINE_STATE_FILE = "state.json"


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


def _run_helper_script(script_name: str, *args: str) -> None:
    script = SCRIPTS_DIR / script_name
    if not script.exists():
        logger.debug("Helper script %s not found; skipping", script_name)
        return
    cmd = [sys.executable, str(script), *args]
    logger.info("Executing helper script %s", script.name)
    try:
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    except subprocess.CalledProcessError as exc:
        if os.getenv("PRE_FLIGHT_STRICT", "1").strip().lower() in {"1", "true", "yes", "on"}:
            raise
        logger.warning(
            "Preflight helper %s failed but PRE_FLIGHT_STRICT=0; continuing. err=%s",
            script_name,
            exc,
        )


def _collect_preflight_artifacts() -> list[Path]:
    artifacts: list[Path] = []
    data_dir = REPO_ROOT / "data"
    if data_dir.exists():
        artifacts.extend(sorted(p for p in data_dir.glob("*.parquet")))
    feature_log = LOG_DIR / "feature_eval.csv"
    if feature_log.exists():
        artifacts.append(feature_log)
    return artifacts


def _run_preflight_scripts() -> list[Path]:
    for script_name in ("migrate_to_parquet.py", "evaluate_features.py"):
        try:
            _run_helper_script(script_name)
        except subprocess.CalledProcessError:
            logger.exception("Helper script %s failed during preflight", script_name)
            raise
    return _collect_preflight_artifacts()


def _run_auto_optimisation() -> tuple[dict[str, Any], list[Path]]:
    logger.info("Starting automated optimisation stage")
    config_path = REPO_ROOT / "config.yaml"
    changes_path = LOG_DIR / "config_changes.csv"
    before_config = config_path.stat().st_mtime if config_path.exists() else None
    before_changes = changes_path.stat().st_mtime if changes_path.exists() else None

    _run_module("mt5.auto_optimize", [])

    metrics: dict[str, Any] = {}
    artifacts: list[Path] = []
    if config_path.exists():
        artifacts.append(config_path)
        after_config = config_path.stat().st_mtime
        metrics["config_updated"] = before_config is None or after_config > before_config
    if changes_path.exists():
        artifacts.append(changes_path)
        after_changes = changes_path.stat().st_mtime
        metrics["config_changes_updated"] = (
            before_changes is None or after_changes > before_changes
        )
    return metrics, artifacts


def _start_artifact_sync() -> subprocess.Popen[bytes] | None:
    script = SCRIPTS_DIR / "hourly_artifact_push.py"
    if not script.exists():
        logger.debug("Artifact sync helper not found; skipping background uploader")
        return None
    logger.info("Launching background artifact sync helper")
    return subprocess.Popen([sys.executable, str(script)], cwd=str(REPO_ROOT))


def _stop_artifact_sync(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    logger.info("Stopping background artifact sync helper")
    try:
        process.terminate()
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        logger.warning("Artifact sync helper did not exit in time; forcing shutdown")
        process.kill()
        with contextlib.suppress(Exception):
            process.wait(timeout=5.0)
    except Exception:
        logger.exception("Failed to terminate artifact sync helper cleanly")


def _collect_realtime_artifacts() -> list[Path]:
    sync_dir = REPO_ROOT / "synced_artifacts"
    return [sync_dir] if sync_dir.exists() else []


def _collect_training_artifacts() -> list[Path]:
    root = Path(__file__).resolve().parent
    candidates = {
        root / "model.joblib",
        root / "scaler.pkl",
        root / "models" / "model.joblib",
        root / "models" / "regime_models",
        root / "reports" / "training" / "progress.json",
    }
    return [path for path in candidates if path.exists()]


def _record_strategy_archive(
    archive: StrategyArchive,
    artifact: Mapping[str, Any],
    episodes: int,
    artifact_path: Path,
) -> Path | None:
    best = artifact.get("best_strategy")
    if not isinstance(best, Mapping):
        return None
    monthly_profit = best.get("pnl")
    metadata = {
        "episodes": episodes,
        "summary": artifact.get("summary"),
        "source": "pipeline",
        "artifact_path": artifact_path.as_posix(),
        "updated_at": artifact.get("updated_at"),
    }
    if monthly_profit is not None:
        metadata["monthly_profit"] = monthly_profit
    return archive.record(best, metadata=metadata)


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
        "--skip-preflight",
        action="store_true",
        help="Skip dataset migration and feature evaluation preflight steps",
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
        "--skip-optimise",
        action="store_true",
        help="Skip automated hyperparameter optimisation",
    )
    parser.add_argument(
        "--skip-realtime",
        action="store_true",
        help="Skip the realtime stage",
    )
    parser.add_argument(
        "--fresh-start",
        action="store_true",
        help="Ignore saved pipeline state and rerun every stage",
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

    state = PipelineState(artifact_dir / PIPELINE_STATE_FILE)
    if args.fresh_start:
        state.reset()
    resume_stage: str | None = None
    if not args.fresh_start and state.should_resume():
        resume_stage = state.resume_stage()
        if resume_stage:
            logger.info("Resuming pipeline from stage %s", resume_stage)

    stage_flags = {
        "preflight": bool(args.skip_preflight),
        "training": bool(args.skip_training),
        "backtest": bool(args.skip_backtest),
        "strategy": bool(args.skip_strategy),
        "optimise": bool(args.skip_optimise),
        "realtime": bool(args.skip_realtime),
    }
    if resume_stage:
        for stage in PIPELINE_STAGES:
            if stage == resume_stage:
                break
            stage_flags[stage] = True

    state.begin_run(args=vars(args), resume_from=resume_stage)
    archive = StrategyArchive()

    preflight_artifacts: list[Path] | None = None
    backtest_metrics: Mapping[str, Any] | None = None
    strategy_artifact: Mapping[str, Any] | None = None
    optimise_metrics: Mapping[str, Any] | None = None

    try:
        if not stage_flags["preflight"]:
            state.mark_stage_started("preflight")
            try:
                preflight_artifacts = _run_preflight_scripts()
            except Exception as exc:
                state.mark_stage_failed("preflight", exc)
                raise
            state.mark_stage_complete(
                "preflight",
                artifacts=preflight_artifacts or [],
                resume=bool(preflight_artifacts),
            )
        else:
            logger.info("Preflight stage skipped via CLI flag or resume")

        if not stage_flags["training"]:
            state.mark_stage_started("training")
            try:
                _run_training(_split_args(args.train_args))
            except Exception as exc:
                state.mark_stage_failed("training", exc)
                raise
            artifacts = _collect_training_artifacts()
            resume_ready = bool(artifacts)
            state.mark_stage_complete("training", artifacts=artifacts, resume=resume_ready)
        else:
            logger.info("Training stage skipped via CLI flag or resume")

        if not stage_flags["backtest"]:
            state.mark_stage_started("backtest")
            try:
                backtest_metrics = _run_backtest(
                    artifact_dir,
                    reuse=not args.no_reuse_backtest,
                    quality_threshold=args.quality_threshold,
                    force=args.force_backtest,
                )
            except Exception as exc:
                state.mark_stage_failed("backtest", exc)
                raise
            artifact_path = artifact_dir / BACKTEST_ARTIFACT_NAME
            state.mark_stage_complete(
                "backtest",
                artifacts=[artifact_path],
                metrics=backtest_metrics or {},
                resume=True,
            )
        else:
            logger.info("Backtesting stage skipped via CLI flag or resume")

        archive_path: Path | None = None
        if not stage_flags["strategy"]:
            state.mark_stage_started("strategy")
            try:
                strategy_artifact = _run_strategy_creation(
                    artifact_dir,
                    episodes=args.strategy_episodes,
                )
            except Exception as exc:
                state.mark_stage_failed("strategy", exc)
                raise
            artifacts: list[Path] = []
            if strategy_artifact is not None:
                artifact_path = artifact_dir / STRATEGY_ARTIFACT_NAME
                artifacts.append(artifact_path)
                try:
                    archive_path = _record_strategy_archive(
                        archive,
                        strategy_artifact,
                        args.strategy_episodes,
                        artifact_path,
                    )
                except Exception:
                    logger.exception("Failed to archive generated strategy")
                else:
                    if archive_path is not None:
                        artifacts.append(archive_path)
            state.mark_stage_complete(
                "strategy",
                artifacts=artifacts,
                metrics=strategy_artifact or {},
                resume=bool(artifacts),
            )
        else:
            logger.info("Strategy creation stage skipped via CLI flag or resume")

        optimise_artifacts: list[Path] = []
        if not stage_flags["optimise"]:
            state.mark_stage_started("optimise")
            try:
                optimise_metrics, optimise_artifacts = _run_auto_optimisation()
            except Exception as exc:
                state.mark_stage_failed("optimise", exc)
                raise
            state.mark_stage_complete(
                "optimise",
                artifacts=optimise_artifacts,
                metrics=optimise_metrics or {},
                resume=bool(optimise_artifacts),
            )
        else:
            logger.info("Optimisation stage skipped via CLI flag or resume")

        if not stage_flags["realtime"]:
            state.mark_stage_started("realtime")
            artifact_process: subprocess.Popen[bytes] | None = None
            try:
                artifact_process = _start_artifact_sync()
                _run_realtime(args.realtime_duration if args.realtime_duration > 0 else None)
            except Exception as exc:
                state.mark_stage_failed("realtime", exc)
                raise
            finally:
                if artifact_process is not None:
                    _stop_artifact_sync(artifact_process)
            state.mark_stage_complete(
                "realtime",
                artifacts=_collect_realtime_artifacts(),
                resume=False,
            )
        else:
            logger.info("Realtime stage skipped via CLI flag or resume")
    except Exception as exc:
        state.mark_run_failed(exc)
        raise
    else:
        state.mark_run_completed()

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
    if optimise_metrics:
        logger.info(
            "Optimisation summary: config_updated=%s, config_changes_updated=%s",
            optimise_metrics.get("config_updated"),
            optimise_metrics.get("config_changes_updated"),
        )

    return 0


if __name__ == "__main__":  # pragma: no cover - invoked via python -m mt5 pipeline
    sys.exit(main())

