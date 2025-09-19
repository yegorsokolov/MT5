"""Market oracle data scalper.

This module introduces :class:`OracleScalper`, a lightweight component that
collects probabilistic market intelligence from prediction markets such as
`Polymarket <https://polymarket.com>`_ and `Metaculus
<https://metaculus.com>`_.  The scalper normalises outcomes into a common
representation, estimates confidences from available liquidity/participation
metrics and produces feature frames suitable for downstream models.

The implementation is intentionally dependency free – it relies on
``urllib`` rather than third-party HTTP clients – so it can operate in the
same constrained environments as the existing baseline strategy code.
Consumers may inject custom fetchers when unit testing or when additional
authentication is required (for example when accessing private Metaculus
questions).
"""

from __future__ import annotations

import json
import logging
import math
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional

import pandas as pd
import statistics

logger = logging.getLogger(__name__)


def _safe_float(value: Any) -> float | None:
    """Return ``value`` converted to ``float`` when possible."""

    if value in (None, "", b""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_timestamp(value: Any) -> pd.Timestamp | None:
    """Parse ``value`` into a timezone aware :class:`~pandas.Timestamp`."""

    if not value:
        return None
    try:
        ts = pd.to_datetime(value, utc=True)
    except Exception:  # pragma: no cover - best effort parsing
        return None
    if isinstance(ts, pd.Timestamp):
        return ts.tz_convert(timezone.utc) if ts.tzinfo else ts.tz_localize(timezone.utc)
    return None


def _probability_to_odds(prob: float) -> float | None:
    """Return decimal odds for ``prob`` with guards for edge cases."""

    if prob <= 0 or prob >= 1:
        return None
    return prob / (1.0 - prob)


def _normalise_keywords(symbols: Iterable[str], aliases: Mapping[str, Iterable[str]] | None) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    alias_map = aliases or {}
    for sym in symbols:
        default = [sym]
        extra = list(alias_map.get(sym, ()))
        keywords = {k.strip().lower() for k in default + extra if k}
        if not keywords:
            continue
        result[sym] = sorted(keywords)
    return result


def _matches_keywords(text: str, keywords: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def _compute_confidence(weight: float, event_count: int) -> float:
    """Map raw ``weight``/``event_count`` into a 0-1 confidence score."""

    if weight <= 0 and event_count <= 0:
        return 0.0
    base = math.log1p(max(weight, 0.0)) + 0.25 * event_count
    return float(max(0.0, min(1.0, base / 10.0)))


@dataclass
class OracleEvent:
    """Normalised prediction-market outcome."""

    oracle: str
    symbol: str
    market_id: str
    question: str
    outcome: str
    probability: float
    liquidity: float = 0.0
    volume: float = 0.0
    weight: float = 1.0
    implied_odds: float | None = None
    confidence: float | None = None
    closing_time: pd.Timestamp | None = None
    collected_at: pd.Timestamp = field(
        default_factory=lambda: pd.Timestamp(datetime.now(tz=timezone.utc))
    )
    metadata: MutableMapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        data = {
            "oracle": self.oracle,
            "symbol": self.symbol,
            "market_id": self.market_id,
            "question": self.question,
            "outcome": self.outcome,
            "probability": self.probability,
            "liquidity": self.liquidity,
            "volume": self.volume,
            "weight": self.weight,
            "implied_odds": self.implied_odds,
            "confidence": self.confidence,
            "closing_time": self.closing_time,
            "collected_at": self.collected_at,
        }
        data.update({f"meta_{k}": v for k, v in (self.metadata or {}).items()})
        return data


FetchCallable = Callable[[int], Iterable[Mapping[str, Any]]]


class PolymarketClient:
    """Client for the public Polymarket markets API."""

    base_url = "https://gamma-api.polymarket.com/markets"

    def __init__(
        self,
        *,
        limit: int = 200,
        fetcher: FetchCallable | None = None,
        user_agent: str = "Mozilla/5.0",
        timeout: int = 15,
    ) -> None:
        self.limit = limit
        self.user_agent = user_agent
        self.timeout = timeout
        self._fetcher = fetcher or self._default_fetcher

    def _default_fetcher(self, limit: int) -> Iterable[Mapping[str, Any]]:
        params = urllib.parse.urlencode({"limit": limit})
        url = f"{self.base_url}?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = resp.read()
        except urllib.error.URLError as exc:  # pragma: no cover - network failure
            logger.warning("Polymarket request failed: %s", exc)
            return []
        try:
            data = json.loads(payload.decode("utf-8"))
        except json.JSONDecodeError:  # pragma: no cover - unexpected format
            logger.warning("Failed to decode Polymarket payload")
            return []
        if isinstance(data, dict):
            return data.get("markets", [])
        return data

    def fetch(self) -> List[Mapping[str, Any]]:
        try:
            data = list(self._fetcher(self.limit))
            logger.debug("Fetched %d Polymarket markets", len(data))
            return data
        except Exception:  # pragma: no cover - defensive guard
            logger.exception("Unexpected error while fetching Polymarket data")
            return []


class MetaculusClient:
    """Client for the public Metaculus questions API."""

    base_url = "https://www.metaculus.com/api2/questions/"

    def __init__(
        self,
        *,
        limit: int = 100,
        fetcher: FetchCallable | None = None,
        user_agent: str = "Mozilla/5.0",
        timeout: int = 15,
        query_params: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.limit = limit
        self.user_agent = user_agent
        self.timeout = timeout
        self.query_params = {"limit": limit}
        if query_params:
            self.query_params.update(query_params)
        self._fetcher = fetcher or self._default_fetcher

    def _default_fetcher(self, limit: int) -> Iterable[Mapping[str, Any]]:
        params = dict(self.query_params)
        params["limit"] = limit
        url = f"{self.base_url}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = resp.read()
        except urllib.error.URLError as exc:  # pragma: no cover
            logger.warning("Metaculus request failed: %s", exc)
            return []
        try:
            data = json.loads(payload.decode("utf-8"))
        except json.JSONDecodeError:  # pragma: no cover
            logger.warning("Failed to decode Metaculus payload")
            return []
        results = data.get("results") if isinstance(data, dict) else None
        if not isinstance(results, list):
            return []
        return results

    def fetch(self) -> List[Mapping[str, Any]]:
        try:
            data = list(self._fetcher(self.limit))
            logger.debug("Fetched %d Metaculus questions", len(data))
            return data
        except Exception:  # pragma: no cover
            logger.exception("Unexpected error while fetching Metaculus data")
            return []


class OracleScalper:
    """Collect, normalise and score prediction market data."""

    def __init__(
        self,
        *,
        polymarket: PolymarketClient | None = None,
        metaculus: MetaculusClient | None = None,
        enabled_oracles: Optional[Iterable[str]] = None,
    ) -> None:
        self.polymarket = polymarket or PolymarketClient()
        self.metaculus = metaculus or MetaculusClient()
        self.enabled_oracles = {
            oracle.lower() for oracle in (enabled_oracles or {"polymarket", "metaculus"})
        }

    def collect(
        self,
        symbols: Iterable[str],
        *,
        aliases: Optional[Mapping[str, Iterable[str]]] = None,
    ) -> pd.DataFrame:
        keywords = _normalise_keywords(symbols, aliases)
        if not keywords:
            sample_columns = list(OracleEvent("", "", "", "", "", 0.0).as_dict().keys())
            return pd.DataFrame(columns=sample_columns)

        events: List[OracleEvent] = []

        if "polymarket" in self.enabled_oracles:
            events.extend(self._collect_polymarket(keywords))
        if "metaculus" in self.enabled_oracles:
            events.extend(self._collect_metaculus(keywords))

        if not events:
            sample_columns = list(OracleEvent("", "", "", "", "", 0.0).as_dict().keys())
            return pd.DataFrame(columns=sample_columns)

        return pd.DataFrame([event.as_dict() for event in events])

    def assess_probabilities(self, events: pd.DataFrame) -> pd.DataFrame:
        if events.empty:
            cols = [
                "symbol",
                "oracle",
                "prob_mean",
                "prob_weighted",
                "prob_median",
                "prob_std",
                "event_count",
                "confidence",
                "latest_close",
                "latest_update",
            ]
            return pd.DataFrame(columns=cols)

        grouped = []
        for (symbol, oracle), subset in events.groupby(["symbol", "oracle"]):
            if subset.empty:
                continue
            weights = [
                float(w) if w is not None and math.isfinite(float(w)) else 0.0
                for w in subset["weight"].tolist()
            ]
            weights = [w if w > 0 else 1.0 for w in weights]
            probs = [float(p) for p in subset["probability"].tolist()]
            total_weight = sum(weights)
            if total_weight > 0:
                prob_weighted = sum(p * w for p, w in zip(probs, weights)) / total_weight
            else:
                prob_weighted = sum(probs) / len(probs)
            row = {
                "symbol": symbol,
                "oracle": oracle,
                "prob_mean": sum(probs) / len(probs),
                "prob_weighted": prob_weighted,
                "prob_median": statistics.median(probs),
                "prob_std": statistics.pstdev(probs) if len(probs) > 1 else 0.0,
                "event_count": int(len(subset)),
                "confidence": _compute_confidence(float(sum(weights)), len(subset)),
                "latest_close": subset["closing_time"].dropna().max(),
                "latest_update": subset["collected_at"].dropna().max(),
            }
            grouped.append(row)

        return pd.DataFrame(grouped)

    def to_feature_frame(self, summary: pd.DataFrame) -> pd.DataFrame:
        columns = {
            "prob_mean": "prob_mean",
            "prob_weighted": "prob_weighted",
            "prob_median": "prob_median",
            "prob_std": "prob_std",
            "event_count": "event_count",
            "confidence": "confidence",
        }

        if summary.empty:
            base_cols = [
                "Symbol",
                "Timestamp",
                "oracle_prob_mean",
                "oracle_prob_weighted",
                "oracle_prob_median",
                "oracle_prob_std",
                "oracle_event_count",
                "oracle_confidence",
            ]
            return pd.DataFrame(columns=base_cols)

        features: List[Dict[str, Any]] = []
        for symbol, subset in summary.groupby("symbol"):
            row: Dict[str, Any] = {"Symbol": symbol}
            for _, entry in subset.iterrows():
                prefix = entry["oracle"]
                for key, col_name in columns.items():
                    row[f"{prefix}_{col_name}"] = entry[key]
            row["oracle_prob_mean"] = subset["prob_weighted"].mean()
            row["oracle_prob_weighted"] = subset["prob_weighted"].mean()
            row["oracle_prob_median"] = subset["prob_median"].mean()
            row["oracle_prob_std"] = subset["prob_std"].mean()
            row["oracle_event_count"] = float(subset["event_count"].sum())
            row["oracle_confidence"] = float(subset["confidence"].mean())
            features.append(row)

        return pd.DataFrame(features)

    def augment_dataframe(
        self,
        df: pd.DataFrame,
        *,
        aliases: Optional[Mapping[str, Iterable[str]]] = None,
    ) -> pd.DataFrame:
        if df.empty:
            return df

        df = df.copy()
        timestamp_col = "Timestamp"
        if timestamp_col not in df.columns:
            raise KeyError("DataFrame must include a 'Timestamp' column for oracle features")

        if "Symbol" in df.columns and df["Symbol"].notna().any():
            symbols = sorted({str(sym) for sym in df["Symbol"].dropna().unique()})
        else:
            symbol_attr = df.attrs.get("symbol")
            if symbol_attr:
                df["Symbol"] = symbol_attr
                symbols = [str(symbol_attr)]
            else:
                logger.debug("Oracle scalper skipped: no symbol metadata available")
                for col in self.feature_columns:
                    df[col] = float("nan")
                return df

        events = self.collect(symbols, aliases=aliases)
        summary = self.assess_probabilities(events) if not events.empty else pd.DataFrame()
        features = self.to_feature_frame(summary)

        if features.empty:
            for col in self.feature_columns:
                df[col] = float("nan")
            return df

        merged = df.merge(features, on="Symbol", how="left")
        merged = merged.sort_values(["Symbol", timestamp_col])
        merged[self.feature_columns] = merged.groupby("Symbol")[self.feature_columns].transform(
            lambda group: group.ffill().bfill()
        )
        return merged

    @property
    def feature_columns(self) -> List[str]:
        base = [
            "oracle_prob_mean",
            "oracle_prob_weighted",
            "oracle_prob_median",
            "oracle_prob_std",
            "oracle_event_count",
            "oracle_confidence",
        ]
        oracle_specific = []
        for oracle in sorted(self.enabled_oracles):
            for suffix in ("prob_mean", "prob_weighted", "prob_median", "prob_std", "event_count", "confidence"):
                oracle_specific.append(f"{oracle}_{suffix}")
        return sorted(set(base + oracle_specific))

    def _collect_polymarket(self, keywords: Mapping[str, List[str]]) -> List[OracleEvent]:
        markets = self.polymarket.fetch()
        events: List[OracleEvent] = []
        for market in markets:
            question = str(market.get("question", ""))
            slug = str(market.get("slug", ""))
            text = f"{question} {slug}".strip()
            outcomes = market.get("outcomes") or []
            prices = market.get("outcomePrices") or []
            if not outcomes or not prices:
                continue
            liquidity = _safe_float(market.get("liquidityNum"))
            if liquidity is None:
                liquidity = _safe_float(market.get("liquidity")) or 0.0
            volume = _safe_float(market.get("volumeNum"))
            if volume is None:
                volume = _safe_float(market.get("volume")) or 0.0
            closing_time = _parse_timestamp(market.get("endDate"))
            market_id = str(market.get("id", ""))
            best_bid = _safe_float(market.get("bestBid"))
            best_ask = _safe_float(market.get("bestAsk"))

            outcome_pairs = list(zip(outcomes, prices))
            preferred = [
                pair
                for pair in outcome_pairs
                if str(pair[0]).strip().lower() in {"yes", "true"}
            ]
            if preferred:
                outcome_pairs = preferred

            for symbol, terms in keywords.items():
                if not _matches_keywords(text, terms):
                    continue
                for outcome, price in outcome_pairs:
                    prob = _safe_float(price)
                    if prob is None:
                        continue
                    odds = _probability_to_odds(prob)
                    weight = liquidity if liquidity and liquidity > 0 else volume or 1.0
                    event = OracleEvent(
                        oracle="polymarket",
                        symbol=symbol,
                        market_id=market_id,
                        question=question,
                        outcome=str(outcome),
                        probability=prob,
                        liquidity=liquidity or 0.0,
                        volume=volume or 0.0,
                        weight=float(weight),
                        implied_odds=odds,
                        closing_time=closing_time,
                        metadata={
                            "slug": slug,
                            "best_bid": best_bid,
                            "best_ask": best_ask,
                        },
                    )
                    event.confidence = _compute_confidence(event.weight, 1)
                    events.append(event)
        return events

    def _collect_metaculus(self, keywords: Mapping[str, List[str]]) -> List[OracleEvent]:
        questions = self.metaculus.fetch()
        events: List[OracleEvent] = []
        for entry in questions:
            title = str(entry.get("title") or entry.get("question", {}).get("title", ""))
            description = str(entry.get("question", {}).get("description", ""))
            combined = f"{title} {description}".strip()
            question_info = entry.get("question", {})
            aggregations = question_info.get("aggregations", {})
            history = (
                aggregations.get("recency_weighted", {}).get("history")
                or aggregations.get("unweighted", {}).get("history")
                or []
            )
            if not history:
                continue
            latest = history[-1]
            centers = latest.get("centers") or []
            if not centers:
                continue
            probability = _safe_float(centers[-1])
            if probability is None:
                continue
            lower = _safe_float((latest.get("interval_lower_bounds") or [None])[-1])
            upper = _safe_float((latest.get("interval_upper_bounds") or [None])[-1])
            forecasters = latest.get("forecaster_count") or 0
            closing_time = _parse_timestamp(
                entry.get("scheduled_close_time")
                or question_info.get("scheduled_close_time")
            )
            market_id = str(entry.get("id", ""))

            for symbol, terms in keywords.items():
                if not _matches_keywords(combined, terms):
                    continue
                weight = float(forecasters) if forecasters else 1.0
                event = OracleEvent(
                    oracle="metaculus",
                    symbol=symbol,
                    market_id=market_id,
                    question=title,
                    outcome="community",
                    probability=float(probability),
                    liquidity=float(forecasters or 0),
                    volume=float(entry.get("forecasts_count") or 0),
                    weight=weight,
                    implied_odds=_probability_to_odds(float(probability)),
                    closing_time=closing_time,
                    metadata={
                        "interval_lower": lower,
                        "interval_upper": upper,
                    },
                )
                event.confidence = _compute_confidence(event.weight, 1)
                events.append(event)
        return events


__all__ = ["OracleScalper", "PolymarketClient", "MetaculusClient", "OracleEvent"]

