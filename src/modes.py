from enum import Enum


class Mode(Enum):
    """Operational modes for the trading system."""

    TRAINING = "training"
    PAPER_TRADING = "paper_trading"
    LIVE_TRADING = "live_trading"
