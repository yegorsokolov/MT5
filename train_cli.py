from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import typer
import pandas as pd
import numpy as np

from tuning.baseline_opt import backtest, run_search as baseline_run_search
from tuning.auto_search import run_search as auto_model_search
from train_graphnet import train_graphnet
from train_price_distribution import prepare_features, train_price_distribution
from train_nn import main as train_nn_main
from walk_forward import walk_forward_train
from train_ensemble import (
    main as train_ensemble_main,
    train_moe_ensemble,
    ResourceCapabilities,
)
from training.pipeline import launch as pipeline_launch
from train_utils import setup_training, end_training
from utils import ensure_environment

app = typer.Typer(help="Unified training interface")


@app.command()
def pipeline(
    config: Optional[Path] = typer.Option(None, help="Path to config YAML"),
    export: bool = typer.Option(False, help="Export model to ONNX"),
    resume_online: bool = typer.Option(False, help="Resume incremental training"),
    transfer_from: Optional[str] = typer.Option(None, help="Donor symbol for transfer"),
    use_pseudo_labels: bool = typer.Option(False, help="Include pseudo-labeled samples"),
    risk_target: Optional[str] = typer.Option(None, help="JSON string specifying risk constraints"),
) -> None:
    cfg = setup_training(config, experiment="pipeline")
    risk_cfg = json.loads(risk_target) if risk_target else None
    pipeline_launch(
        cfg,
        export=export,
        resume_online=resume_online,
        transfer_from=transfer_from,
        use_pseudo_labels=use_pseudo_labels,
        risk_target=risk_cfg,
    )
    end_training()


@app.command()
def baseline(
    data: Path = typer.Option(..., help="Historical data file (csv or parquet)"),
    tune: bool = typer.Option(False, help="Run Optuna parameter search"),
    trials: int = typer.Option(30, help="Number of Optuna trials"),
    config: Optional[Path] = typer.Option(None, help="Path to config YAML"),
) -> None:
    cfg = setup_training(config, experiment="baseline")
    p = Path(data)
    if p.suffix == ".csv":
        df = pd.read_csv(p)
    else:
        df = pd.read_parquet(p)
    if tune:
        best = baseline_run_search(df, n_trials=trials)
        typer.echo(best)
    else:
        params = {
            "short_window": 5,
            "long_window": 20,
            "atr_window": 14,
            "cvd_threshold": 0.0,
            "stop_mult": 3.0,
        }
        score = backtest(params, df)
        typer.echo(f"Sharpe: {score:.3f}")
    end_training()


@app.command()
def neural(
    config: Optional[Path] = typer.Option(None, help="Path to config YAML"),
    resume_online: bool = typer.Option(False, help="Resume online training"),
    transfer_from: Optional[str] = typer.Option(
        None, help="Donor symbol for transfer learning"
    ),
) -> None:
    cfg = setup_training(config, experiment="training_nn")
    train_nn_main(cfg=cfg, resume_online=resume_online, transfer_from=transfer_from)
    end_training()


@app.command()
def graph(
    data: Path = typer.Option(..., help="CSV or parquet file of return series"),
    config: Optional[Path] = typer.Option(None, help="Path to config YAML"),
) -> None:
    cfg = setup_training(config, experiment="graph")
    p = Path(data)
    if p.suffix == ".csv":
        df = pd.read_csv(p)
    else:
        df = pd.read_parquet(p)
    train_graphnet(df, cfg)
    end_training()


@app.command("price-distribution")
def price_distribution(
    train: Path = typer.Option(..., help="Training CSV"),
    val: Path = typer.Option(..., help="Validation CSV"),
    n_components: int = typer.Option(3, help="Number of mixture components"),
    epochs: int = typer.Option(100, help="Training epochs"),
    config: Optional[Path] = typer.Option(None, help="Path to config YAML"),
) -> None:
    cfg = setup_training(config, experiment="price_distribution")
    train_df = pd.read_csv(train)
    val_df = pd.read_csv(val)
    X_train, y_train = prepare_features(train_df)
    X_val, y_val = prepare_features(val_df)
    train_price_distribution(
        X_train, y_train, X_val, y_val, n_components=n_components, epochs=epochs
    )
    end_training()


@app.command("walk-forward")
def walk_forward(
    data: Path = typer.Option(..., help="CSV or parquet file containing returns"),
    window_length: int = typer.Option(100, help="Length of the training window"),
    step_size: int = typer.Option(10, help="Step size and test window length"),
    model_type: str = typer.Option("mean", help="Type of model to train"),
    config: Optional[Path] = typer.Option(None, help="Path to config YAML"),
) -> None:
    cfg = setup_training(config, experiment="walk_forward")
    results = walk_forward_train(data, window_length, step_size, model_type)
    typer.echo(results.to_json(orient="records"))
    end_training()


@app.command("auto-search")
def auto_search(
    data: Path = typer.Option(
        ..., help="CSV or parquet with feature columns and target column"
    ),
    target: str = typer.Option("y", help="Name of target column in data"),
    trials: int = typer.Option(10, help="Number of Optuna trials"),
    config: Optional[Path] = typer.Option(None, help="Path to config YAML"),
) -> None:
    cfg = setup_training(config, experiment="auto_search")
    p = Path(data)
    if p.suffix == ".csv":
        df = pd.read_csv(p)
    else:
        df = pd.read_parquet(p)
    y = df[target].values
    X = df.drop(columns=[target]).values
    best, summary = auto_model_search(X, y, n_trials=trials)
    typer.echo(best)
    typer.echo(summary.to_json(orient="records"))
    end_training()


def _load_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise typer.BadParameter("Data file must be CSV or parquet", param_hint="data")


def _sort_history_columns(columns: List[str], prefix: str) -> List[str]:
    def key(col: str) -> tuple[int, str]:
        suffix = col[len(prefix) :]
        try:
            return (int(suffix), "")
        except ValueError:
            return (float("inf"), suffix)

    return sorted(columns, key=key)


@app.command()
def ensemble(
    data: Path = typer.Option(..., help="CSV or parquet with ensemble features"),
    target: str = typer.Option("target", help="Target column for ensemble training"),
    feature: Optional[List[str]] = typer.Option(
        None,
        "--feature",
        "-f",
        help="Feature column(s) for base learners; defaults to all non-target columns.",
    ),
    config: Optional[Path] = typer.Option(None, help="Path to config YAML"),
    moe_regime: Optional[str] = typer.Option(
        None,
        help="Column containing regime labels for mixture-of-experts gating.",
    ),
    moe_history_prefix: str = typer.Option(
        "history_",
        help="Prefix used for mixture-of-experts history columns.",
    ),
    moe_target: Optional[str] = typer.Option(
        None, help="Target column for mixture-of-experts (defaults to --target)."
    ),
    gating_sharpness: float = typer.Option(
        5.0, help="Softmax sharpness for the gating network"
    ),
    expert_weight: Optional[List[float]] = typer.Option(
        None,
        "--expert-weight",
        help="Expert weight applied before gating. Provide once per expert.",
    ),
    diversity_weight: Optional[List[float]] = typer.Option(
        None,
        "--diversity-weight",
        help="Optional diversity weights passed to the gating network.",
    ),
) -> None:
    cfg = setup_training(config, experiment="ensemble")
    data_path = Path(data)
    df = _load_frame(data_path)
    if target not in df.columns:
        raise typer.BadParameter(
            f"Target column '{target}' not present in data", param_hint="target"
        )
    feature_cols = feature or [col for col in df.columns if col != target]
    feature_cols = [col for col in feature_cols if col != target]
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise typer.BadParameter(
            f"Unknown feature columns: {', '.join(missing)}", param_hint="--feature"
        )
    X = df[feature_cols]
    y = df[target].to_numpy()

    try:
        metrics = train_ensemble_main(cfg=cfg, data=(X, y))
        typer.echo(json.dumps(metrics, sort_keys=True))

        if moe_regime is not None:
            if moe_regime not in df.columns:
                raise typer.BadParameter(
                    f"Regime column '{moe_regime}' not present in data",
                    param_hint="--moe-regime",
                )
            moe_target_col = moe_target or target
            if moe_target_col not in df.columns:
                raise typer.BadParameter(
                    f"Mixture target column '{moe_target_col}' not present in data",
                    param_hint="--moe-target",
                )
            history_cols = [
                col for col in df.columns if col.startswith(moe_history_prefix)
            ]
            if not history_cols:
                raise typer.BadParameter(
                    "No history columns matching prefix",
                    param_hint="--moe-history-prefix",
                )
            history_cols = _sort_history_columns(history_cols, moe_history_prefix)
            moe_frame = df[history_cols + [moe_regime, moe_target_col]].dropna()
            if moe_frame.empty:
                raise typer.BadParameter(
                    "No rows remain after dropping NaNs for mixture training",
                    param_hint="--moe-history-prefix",
                )
            histories = moe_frame[history_cols].to_numpy(dtype=float)
            regimes = moe_frame[moe_regime].tolist()
            moe_targets = moe_frame[moe_target_col].to_numpy(dtype=float)

            moe_cfg: dict[str, List[float] | float] = {"sharpness": gating_sharpness}
            if expert_weight is not None:
                moe_cfg["expert_weights"] = list(expert_weight)
            if diversity_weight is not None:
                moe_cfg["diversity_weights"] = list(diversity_weight)

            mix_preds, expert_preds = train_moe_ensemble(
                [row.tolist() for row in histories],
                regimes,
                moe_targets,
                ResourceCapabilities(4, 16, False, gpu_count=0),
                cfg=moe_cfg,
            )
            expert_mse = ((expert_preds - moe_targets[:, None]) ** 2).mean(axis=0)
            mix_mse = float(np.mean((mix_preds - moe_targets) ** 2))
            moe_metrics = {
                "mse_best_expert": float(expert_mse.min()),
                "mse_mixture": mix_mse,
                "mixture_improvement": float(expert_mse.min() - mix_mse),
            }
            typer.echo(json.dumps(moe_metrics, sort_keys=True))
    finally:
        end_training()


def main() -> None:
    ensure_environment()
    app()


if __name__ == "__main__":
    main()
