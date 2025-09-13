import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from hypothesis import given, strategies as st

from analysis.diff_backtest import simulate_pnl, soft_position


@given(
    st.lists(
        st.tuples(
            st.floats(-10, 10),
            st.floats(-10, 10),
            st.floats(-10, 10),
        ),
        min_size=1,
        max_size=50,
    )
)
def test_inventory_bounds(logit_tuples):
    """Positions derived from logits stay within [-1, 1]."""
    logits = torch.tensor(logit_tuples, dtype=torch.float32)
    pos = soft_position(logits)
    assert torch.all(pos <= 1.0)
    assert torch.all(pos >= -1.0)


@given(
    st.integers(min_value=2, max_value=50).flatmap(
        lambda n: st.tuples(
            st.lists(
                st.floats(-1, 1),
                min_size=n - 1,
                max_size=n - 1,
            ),
            st.lists(
                st.floats(0, 1),
                min_size=n - 1,
                max_size=n - 1,
            ),
            st.lists(
                st.floats(0, 1),
                min_size=n,
                max_size=n,
            ),
        )
    )
)
def test_pnl_monotonicity_zero_slippage(data):
    """More favourable prices yield no worse PnL when slippage is zero."""
    returns, extras, positions = data

    price1 = [100.0]
    price2 = [100.0]
    for r, e in zip(returns, extras):
        price1.append(price1[-1] + r)
        price2.append(price2[-1] + r + e)

    pos = torch.tensor(positions, dtype=torch.float32)
    pnl1 = simulate_pnl(torch.tensor(price1), pos, slippage=0.0)
    pnl2 = simulate_pnl(torch.tensor(price2), pos, slippage=0.0)

    assert pnl2.sum().item() >= pnl1.sum().item()
