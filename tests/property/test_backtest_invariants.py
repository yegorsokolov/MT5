import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hypothesis import given, strategies as st


def run_backtest(prices: list[float], signals: list[float], *, initial_cash: float = 1000.0, max_position: float = 1.0) -> tuple[list[float], list[float]]:
    """Simple backtest ensuring cash and positions respect bounds."""
    cash = initial_cash
    pos = 0.0
    cash_hist = [cash]
    pos_hist = [pos]

    for i in range(len(signals)):
        price = prices[i]
        next_price = prices[i + 1]

        target = max(min(signals[i], max_position), -max_position)
        delta = target - pos
        cost = delta * price
        if cost > cash:
            delta = cash / price
            cost = delta * price
            target = pos + delta
        cash -= cost
        pos = target
        cash += pos * (next_price - price)
        cash_hist.append(cash)
        pos_hist.append(pos)

    return cash_hist, pos_hist


@given(
    st.integers(min_value=2, max_value=50).flatmap(
        lambda n: st.tuples(
            st.lists(st.floats(min_value=1, max_value=100), min_size=n, max_size=n),
            st.lists(
                st.floats(min_value=-2, max_value=2), min_size=n - 1, max_size=n - 1
            ),
            st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
        )
    )
)
def test_backtest_invariants(data) -> None:
    prices, signals, max_pos = data
    cash, pos = run_backtest(prices, signals, max_position=max_pos)
    assert len(cash) == len(prices)
    assert len(pos) == len(prices)
    assert all(abs(p) <= max_pos + 1e-6 for p in pos)
    max_price = max(prices)
    assert all(
        c + p * price >= -max_pos * max_price - 1e-6
        for price, c, p in zip(prices, cash, pos)
    )
