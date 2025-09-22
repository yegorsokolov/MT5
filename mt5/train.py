"""CLI wrapper around :mod:`training.pipeline`."""

from __future__ import annotations

import argparse
import json
from mt5.ray_utils import init as ray_init, shutdown as ray_shutdown
from training.pipeline import init_logging as init_pipeline_logging, launch, main
from utils import ensure_environment, load_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Adaptive MT5 training CLI")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter search")
    parser.add_argument(
        "--evo-search",
        action="store_true",
        help="Run evolutionary multi-objective parameter search",
    )
    parser.add_argument("--export", action="store_true", help="Export model to ONNX")
    parser.add_argument(
        "--resume-online",
        action="store_true",
        help="Resume incremental training from the latest checkpoint",
    )
    parser.add_argument(
        "--transfer-from",
        type=str,
        help="Initialize model using weights from a donor symbol",
    )
    parser.add_argument(
        "--meta-train",
        action="store_true",
        help="Run meta-training to produce meta-initialised weights",
    )
    parser.add_argument(
        "--fine-tune",
        action="store_true",
        help="Fine-tune from meta weights on the latest regime",
    )
    parser.add_argument(
        "--meta-init",
        action="store_true",
        help="Initialise from meta weights and adapt to the current dataset",
    )
    parser.add_argument(
        "--use-pseudo-labels",
        action="store_true",
        help="Include pseudo-labeled samples during training",
    )
    parser.add_argument(
        "--use-price-distribution",
        action="store_true",
        help="Train auxiliary PriceDistributionModel",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=None,
        help="Number of mixture components for PriceDistributionModel",
    )
    parser.add_argument(
        "--strategy-graph",
        action="store_true",
        help="Generate and backtest strategy graphs",
    )
    parser.add_argument(
        "--strategy-controller",
        action="store_true",
        help="Train a neural controller that emits DSL trading actions",
    )
    parser.add_argument(
        "--risk-target",
        type=str,
        default=None,
        help="JSON string specifying risk constraints",
    )
    return parser


def _run_strategy_controller(risk_target: dict | None) -> None:
    from models.strategy_controller import evaluate_controller, train_strategy_controller

    controller = train_strategy_controller()
    market_data = [
        {"price": 1.0, "ma": 2.0},
        {"price": 3.0, "ma": 2.0},
        {"price": 1.0, "ma": 2.0},
    ]
    pnl = evaluate_controller(controller, market_data)
    print(f"Strategy controller PnL: {pnl:.2f}")


def _run_strategy_graph(risk_target: dict | None) -> None:
    from models.strategy_graph_controller import StrategyGraphController
    import numpy as np

    features = np.array([[1.0, 2.0]])
    risk_profile = (risk_target or {}).get("risk", 0.5)
    controller = StrategyGraphController(input_dim=features.shape[1])
    graph = controller.generate(features, risk_profile)
    data = [
        {"price": 1.0, "ma": 0.0},
        {"price": 2.0, "ma": 3.0},
    ]
    pnl = graph.run(data)
    print(f"Strategy graph PnL: {pnl:.2f}")


def _run_tuning(risk_target: dict | None) -> None:
    from tuning.bayesian_search import run_search

    cfg = load_config().model_dump()

    def train_fn(c: dict, _trial) -> float:
        return main(c, risk_target=risk_target)

    run_search(train_fn, cfg)


def _run_evolutionary_search(risk_target: dict | None) -> None:
    from copy import deepcopy
from mt5.backtest import run_backtest
    from tuning.evolutionary_search import run_evolutionary_search

    cfg = load_config().model_dump()

    def eval_fn(params: dict) -> tuple[float, float, float]:
        trial_cfg = deepcopy(cfg)
        trial_cfg.update(params)
        main(trial_cfg, risk_target=risk_target)
        metrics = run_backtest(trial_cfg)
        return (
            -float(metrics.get("return", 0.0)),
            float(metrics.get("max_drawdown", 0.0)),
            -float(metrics.get("trade_count", metrics.get("trades", 0.0))),
        )

    space = {
        "learning_rate": (1e-4, 2e-1, "log"),
        "num_leaves": (16, 255, "int"),
        "max_depth": (3, 12, "int"),
    }
    run_evolutionary_search(eval_fn, space)


def main_cli() -> None:
    init_pipeline_logging()
    ensure_environment()
    args = _build_parser().parse_args()
    risk_target = json.loads(args.risk_target) if args.risk_target else None

    ray_init()
    try:
        if args.strategy_controller:
            _run_strategy_controller(risk_target)
        elif args.strategy_graph:
            _run_strategy_graph(risk_target)
        elif args.tune:
            _run_tuning(risk_target)
        elif args.evo_search:
            _run_evolutionary_search(risk_target)
        else:
            cfg = load_config()
            cfg_dict = cfg.model_dump()
            if args.meta_train:
                cfg_dict["meta_train"] = True
            if args.meta_init:
                cfg_dict["meta_init"] = True
            if args.fine_tune:
                cfg_dict["fine_tune"] = True
            if args.use_pseudo_labels:
                cfg_dict["use_pseudo_labels"] = True
            if args.use_price_distribution:
                cfg_dict["use_price_distribution"] = True
            if args.n_components is not None:
                cfg_dict["n_components"] = args.n_components
            launch(
                cfg_dict,
                export=args.export,
                resume_online=args.resume_online,
                transfer_from=args.transfer_from,
                use_pseudo_labels=args.use_pseudo_labels,
                risk_target=risk_target,
            )
    finally:
        ray_shutdown()


if __name__ == "__main__":
    main_cli()
