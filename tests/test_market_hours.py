import datetime
from utils.market_hours import is_market_open


def test_is_market_open_weekend():
    saturday = datetime.datetime(2023, 1, 7)
    assert not is_market_open(now=saturday)
