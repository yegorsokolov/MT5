import argparse
import os

import state_manager
import risk_manager as rm_mod


def configure_user_risk(argv: list[str] | None = None) -> tuple[float, float, int]:
    """Obtain user risk limits via CLI or interactive prompts."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--daily-drawdown", type=float)
    parser.add_argument("--total-drawdown", type=float)
    parser.add_argument("--news-blackout-minutes", type=int)
    args, _ = parser.parse_known_args(argv)

    saved = state_manager.load_user_risk() or {}
    dd = args.daily_drawdown or saved.get("daily_drawdown")
    if dd is None:
        dd = float(input("Enter max daily drawdown: "))
    td = args.total_drawdown or saved.get("total_drawdown")
    if td is None:
        td = float(input("Enter max total drawdown: "))
    nb = args.news_blackout_minutes or saved.get("news_blackout_minutes")
    if nb is None:
        nb = int(input("Enter news blackout minutes (+/-): "))
    state_manager.save_user_risk(dd, td, nb)
    rm_mod.risk_manager.max_drawdown = dd
    rm_mod.risk_manager.max_total_drawdown = td
    os.environ["NEWS_BLACKOUT_MINUTES"] = str(nb)
    return dd, td, nb
