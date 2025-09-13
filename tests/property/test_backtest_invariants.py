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
            st.lists(st.floats(min_value=-2, max_value=2), min_size=n - 1, max_size=n - 1),
        )
    )
)
def test_backtest_invariants(data) -> None:
    prices, signals = data
    cash, pos = run_backtest(prices, signals)
    assert all(c >= 0 for c in cash)
    assert all(abs(p) <= 1.0 + 1e-6 for p in pos)
