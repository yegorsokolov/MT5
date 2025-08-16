import os
import sys

import pandas as pd

# ensure project root on path for direct execution
sys.path.append(os.getcwd())

from analysis.session_features import classify_session, add_session_features


def test_london_session_across_dst_start_and_end():
    """London session should handle DST transitions correctly."""
    # Before DST start 2021-03-28: session 08:00-16:00 UTC
    ts_out_winter = pd.Timestamp("2021-03-26T07:00Z")
    ts_in_winter = pd.Timestamp("2021-03-26T08:00Z")
    # After DST start the session shifts one hour earlier in UTC
    ts_out_spring = pd.Timestamp("2021-03-29T06:59Z")
    ts_in_spring = pd.Timestamp("2021-03-29T07:01Z")

    assert classify_session(ts_out_winter) != "london"
    assert classify_session(ts_in_winter) == "london"
    assert classify_session(ts_out_spring) != "london"
    assert classify_session(ts_in_spring) == "london"

    # Before DST end 2021-10-31: session still at 07:00 UTC start
    ts_in_summer = pd.Timestamp("2021-10-29T07:00Z")
    # After DST end the session moves back to 08:00 UTC
    ts_out_autumn = pd.Timestamp("2021-11-01T07:00Z")
    ts_in_autumn = pd.Timestamp("2021-11-01T08:00Z")

    assert classify_session(ts_in_summer) == "london"
    assert classify_session(ts_out_autumn) != "london"
    assert classify_session(ts_in_autumn) == "london"


def test_add_session_features_onehot():
    times = pd.to_datetime([
        "2021-03-26T09:00Z",  # London session
        "2021-03-26T02:00Z",  # Tokyo session
        "2021-03-26T18:00Z",  # New York session
    ])
    df = pd.DataFrame({"Timestamp": times})
    out = add_session_features(df)
    assert list(out["session_london"]) == [1, 0, 0]
    assert list(out["session_tokyo"]) == [0, 1, 0]
    assert list(out["session_new_york"]) == [0, 0, 1]
