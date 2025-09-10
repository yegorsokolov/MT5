from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
import pandas as pd

from tuning.baseline_opt import backtest, run_search
from train_graphnet import train_graphnet
from train_price_distribution import prepare_features, train_price_distribution
from train_nn import main as train_nn_main
from train_utils import setup_training, end_training

app = typer.Typer(help="Unified training interface")


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
        best = run_search(df, n_trials=trials)
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
    transfer_from: Optional[str] = typer.Option(None, help="Donor symbol for transfer learning"),
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


if __name__ == "__main__":
    app()
