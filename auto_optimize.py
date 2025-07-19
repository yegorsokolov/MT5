import copy

from train import main as train_model
from utils import load_config, update_config
from backtest import run_backtest


def main():
    # Train latest model
    train_model()

    cfg = load_config()
    orig_threshold = cfg.get("threshold", 0.55)
    orig_stop = cfg.get("trailing_stop_pips", 20)

    thresholds = [0.50, 0.55, 0.60, 0.65]
    stops = [15, 20, 25]

    best = None
    best_sharpe = -1e9
    for th in thresholds:
        for st in stops:
            test_cfg = copy.deepcopy(cfg)
            test_cfg["threshold"] = th
            test_cfg["trailing_stop_pips"] = st
            metrics = run_backtest(test_cfg)
            if metrics["sharpe"] > best_sharpe:
                best_sharpe = metrics["sharpe"]
                best = (th, st)

    best_threshold, best_stop = best
    reason = f"auto optimisation sharpe {best_sharpe:.4f}"
    if best_threshold != orig_threshold:
        update_config("threshold", best_threshold, reason)
    if best_stop != orig_stop:
        update_config("trailing_stop_pips", best_stop, reason)


if __name__ == "__main__":
    main()
