"""News and calendar related features."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from data.events import get_events


def _merge_asof(left: pd.DataFrame, right: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return pd.merge_asof(left, right, **kwargs)


def add_economic_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    try:
        events = get_events(past_events=True)
    except Exception:  # pragma: no cover - network issues
        events = []
    if not events:
        df["minutes_to_event"] = np.nan
        df["minutes_from_event"] = np.nan
        df["nearest_news_minutes"] = np.nan
        df["upcoming_red_news"] = 0
        return df
    event_time = pd.to_datetime(events[0]["date"]).tz_localize(None)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"]).dt.tz_localize(None)
    df["minutes_to_event"] = (event_time - df["Timestamp"]).dt.total_seconds() / 60
    df["minutes_from_event"] = (df["Timestamp"] - event_time).dt.total_seconds() / 60
    df["nearest_news_minutes"] = np.minimum(
        df["minutes_to_event"].abs(), df["minutes_from_event"].abs()
    )
    df["upcoming_red_news"] = (
        (df["minutes_to_event"] >= 0) & (df["minutes_to_event"] <= 60)
    ).astype(int)
    return df


def add_news_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    path = Path(__file__).resolve().parent / "data" / "news_sentiment.csv"
    if not path.exists():
        df["news_sentiment"] = 0.0
        df["news_summary"] = ""
        return df
    news = pd.read_csv(path)
    if "Timestamp" not in news.columns or "sentiment" not in news.columns:
        df["news_sentiment"] = 0.0
        df["news_summary"] = ""
        return df
    news["Timestamp"] = pd.to_datetime(news["Timestamp"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    news = news.sort_values("Timestamp")
    df = df.sort_values("Timestamp")
    df = _merge_asof(df, news, on="Timestamp", direction="backward")
    df["news_sentiment"] = df["sentiment"].fillna(0.0)
    if "summary" in news.columns:
        df["news_summary"] = df["summary"].fillna("")
    else:
        df["news_summary"] = ""
    df = df.drop(columns=["sentiment", "summary"], errors="ignore")
    return df


def compute(df: pd.DataFrame) -> pd.DataFrame:
    from data import features as base

    df = base.add_economic_calendar_features(df)
    df = base.add_news_sentiment_features(df)
    return df


__all__ = [
    "add_economic_calendar_features",
    "add_news_sentiment_features",
    "compute",
]
