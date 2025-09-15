import argparse
import os

import state_manager
import risk_manager as rm_mod


def configure_user_risk(argv: list[str] | None = None) -> tuple[float, float, int]:
    """Obtain user risk limits via CLI or defaults.

    CLI arguments are interpreted as absolute currency amounts. When no
    prior user input is available the limits default to 4.9% and 9.8% of
    the initial capital for daily and total drawdown respectively.
    """

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--daily-drawdown", type=float)
    parser.add_argument("--total-drawdown", type=float)
    parser.add_argument("--news-blackout-minutes", type=int)
    parser.add_argument("--allow-hedging", action="store_true")
    args, _ = parser.parse_known_args(argv)

    saved = state_manager.load_user_risk()
    dd = args.daily_drawdown or saved["daily_drawdown"]
    td = args.total_drawdown or saved["total_drawdown"]
    nb = args.news_blackout_minutes or saved["news_blackout_minutes"]
    hedge = saved.get("allow_hedging", False)
    if args.allow_hedging:
        hedge = True
    state_manager.save_user_risk(dd, td, nb, hedge)
    rm_mod.risk_manager.update_drawdown_limits(dd, td)
    rm_mod.risk_manager.set_allow_hedging(hedge)
    os.environ["NEWS_BLACKOUT_MINUTES"] = str(nb)
    return dd, td, nb
