"""Helpers for diagnosing and remediating MetaTrader5 errors.

The trading stack relies on the ``MetaTrader5`` Python package for order
execution and market data.  The broker APIs occasionally return terse error
codes such as ``4301`` or free-form strings like ``volume too high``.  Operators
previously had to triage these issues manually.  This module provides a
light-weight knowledge base so the model can offer actionable remediation
steps whenever an :class:`~brokers.mt5_direct.MT5Error` bubbles up.

The resolver returns structured plans describing the likely root cause, a
checklist of suggested steps and optional references to the official MT5
documentation.  It can also raise events through
``analytics.issue_client.IssueClient`` so unresolved problems are recorded in a
shared issue tracker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Pattern, Tuple

from brokers.mt5_direct import MT5Error

try:  # pragma: no cover - optional dependency during lightweight tests
    from analytics.issue_client import IssueClient
except Exception:  # pragma: no cover - analytics optional in some environments
    IssueClient = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _format_template(template: str, params: Mapping[str, Any]) -> str:
    """Safely format ``template`` with ``params``."""

    try:
        return template.format(**params)
    except Exception:
        return template


def _maybe_list(value: Iterable[str] | str | None) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


@dataclass(frozen=True)
class KnowledgeEntry:
    """Description of a known MT5 issue."""

    summary: str
    steps: Tuple[str, ...]
    category: str = "generic"
    severity: str = "error"
    references: Tuple[str, ...] = field(default_factory=tuple)
    requires_manual_action: bool = True
    auto_issue: bool = False


_VOLUME_CONSTRAINT = KnowledgeEntry(
    summary="Requested order volume violates broker constraints.",
    steps=(
        "Inspect the MT5 specification for {symbol} to confirm lot size limits.",
        "Adjust position sizing so the requested volume {volume} respects min, max and step.",
        "Retry the order after updating risk settings or the strategy configuration.",
    ),
    category="risk",
    references=(
        "https://www.mql5.com/en/docs/constants/errorswarnings/enum_trade_return_codes",
    ),
)

_SYMBOL_UNKNOWN = KnowledgeEntry(
    summary="Symbol is not available or enabled in the connected MT5 terminal.",
    steps=(
        "Open the MT5 terminal and ensure {symbol} is visible in the Market Watch window.",
        "Call brokers.mt5_direct.symbol_select('{symbol}', True) before submitting orders.",
        "Verify the strategy configuration uses the broker suffix (e.g. '.m' or '.pro').",
    ),
    category="configuration",
    references=(
        "https://www.metatrader5.com/en/terminal/help/trading/symbols_list",
    ),
)

_TRADE_DISABLED = KnowledgeEntry(
    summary="Trading is disabled for {symbol} at the moment.",
    steps=(
        "Check the broker's trading session calendar for {symbol} and confirm the market is open.",
        "Inspect the symbol specification for trade and freeze levels; widen stops if necessary.",
        "Contact the broker if trading remains disabled or restricted for the account.",
    ),
    category="broker",
    references=(
        "https://www.mql5.com/en/docs/constants/errorswarnings/enum_trade_return_codes",
    ),
)

_NO_MARKET_DATA = KnowledgeEntry(
    summary="No historical data available for {symbol} in the requested window.",
    steps=(
        "Use MT5's \"Download History\" dialog to refresh the price history for {symbol}.",
        "Ensure the terminal stays connected until the download completes.",
        "Retry the backfill or switch to a symbol that has active quotes.",
    ),
    category="data",
    references=(
        "https://www.metatrader5.com/en/terminal/help/history_center",
    ),
)

_CONNECTION_LOST = KnowledgeEntry(
    summary="The MT5 terminal reported a connection problem.",
    steps=(
        "Confirm the MetaTrader terminal is running and logged into the target account.",
        "Check the network connection between this host and the MT5 terminal (RDP/VPS).",
        "Re-run brokers.mt5_direct.initialize() after the terminal reconnects.",
    ),
    category="infrastructure",
    references=(
        "https://www.metatrader5.com/en/terminal/help/service/startservers",
    ),
)


DEFAULT_KNOWLEDGE_BASE: Dict[int, KnowledgeEntry] = {
    4106: _VOLUME_CONSTRAINT,
    4108: _VOLUME_CONSTRAINT,
    4301: _SYMBOL_UNKNOWN,
    4401: _NO_MARKET_DATA,
    4756: _TRADE_DISABLED,
    4757: _TRADE_DISABLED,
    10004: _CONNECTION_LOST,
    10006: _CONNECTION_LOST,
}


DEFAULT_MESSAGE_PATTERNS: Tuple[Tuple[Pattern[str], KnowledgeEntry], ...] = (
    (re.compile(r"volume\s+(?:too\s+(?:high|low)|not\s+allowed|invalid)", re.I), _VOLUME_CONSTRAINT),
    (re.compile(r"unknown\s+symbol", re.I), _SYMBOL_UNKNOWN),
    (re.compile(r"trade\s+disabled|market\s+closed", re.I), _TRADE_DISABLED),
    (re.compile(r"no\s+data", re.I), _NO_MARKET_DATA),
    (re.compile(r"no\s+connection|timeout", re.I), _CONNECTION_LOST),
)


GENERIC_ENTRY = KnowledgeEntry(
    summary="Unrecognised MetaTrader5 error.",
    steps=(
        "Inspect the MT5 journal for additional diagnostics and broker-side restrictions.",
        "Verify account permissions, leverage limits and the trading session schedule.",
        "If the issue persists capture the logs and escalate via the shared issue tracker.",
    ),
    auto_issue=True,
)


def _extract_context(error: MT5Error) -> Dict[str, Any]:
    details: MutableMapping[str, Any] = {}
    if hasattr(error, "details") and isinstance(error.details, MutableMapping):
        details = error.details
    request = details.get("request") if isinstance(details.get("request"), Mapping) else {}
    comment = details.get("comment")
    last_message = details.get("last_error_message")
    volume = None
    symbol = None
    if isinstance(request, Mapping):
        symbol = request.get("symbol")
        volume = request.get("volume")
    text_parts = [str(error)]
    if isinstance(comment, str):
        text_parts.append(comment)
    if isinstance(last_message, str):
        text_parts.append(last_message)
    return {
        "details": details,
        "request": request if isinstance(request, Mapping) else {},
        "comment": comment,
        "last_error_message": last_message,
        "symbol": symbol,
        "volume": volume,
        "text": " ".join(part for part in text_parts if part),
    }


class MT5IssueSolver:
    """Diagnose ``MT5Error`` instances and suggest remediation steps."""

    def __init__(
        self,
        *,
        knowledge_base: Mapping[int, KnowledgeEntry] | None = None,
        message_patterns: Iterable[Tuple[Pattern[str], KnowledgeEntry]] | None = None,
        issue_client: IssueClient | None = None,
    ) -> None:
        self.knowledge_base: Mapping[int, KnowledgeEntry] = knowledge_base or DEFAULT_KNOWLEDGE_BASE
        self.message_patterns: Tuple[Tuple[Pattern[str], KnowledgeEntry], ...] = tuple(
            message_patterns or DEFAULT_MESSAGE_PATTERNS
        )
        self.generic_entry = GENERIC_ENTRY
        self.issue_client = issue_client

    # ------------------------------------------------------------------
    def _lookup(self, error: MT5Error) -> Tuple[KnowledgeEntry, Dict[str, Any]]:
        context = _extract_context(error)
        code = getattr(error, "code", None)
        if code is not None and code in self.knowledge_base:
            return self.knowledge_base[code], context
        text = context.get("text", "")
        for pattern, spec in self.message_patterns:
            if pattern.search(text):
                return spec, context
        return self.generic_entry, context

    # ------------------------------------------------------------------
    def _build_plan(
        self, spec: KnowledgeEntry, context: Dict[str, Any], error: MT5Error
    ) -> Tuple[Dict[str, Any], bool]:
        symbol = context.get("symbol") or "the requested symbol"
        volume = context.get("volume")
        params = {
            "symbol": symbol,
            "volume": volume if volume is not None else "the requested volume",
            "message": str(error),
        }
        plan = {
            "summary": _format_template(spec.summary, params),
            "steps": [_format_template(step, params) for step in spec.steps],
            "references": _maybe_list(spec.references),
            "category": spec.category,
            "severity": spec.severity,
            "requires_manual_action": spec.requires_manual_action,
            "context": {k: context.get(k) for k in ("symbol", "volume") if context.get(k) is not None},
        }
        return plan, spec.auto_issue

    # ------------------------------------------------------------------
    def explain(self, error: MT5Error) -> Dict[str, Any]:
        spec, context = self._lookup(error)
        plan, _ = self._build_plan(spec, context, error)
        return plan

    # ------------------------------------------------------------------
    def solve(
        self,
        error: MT5Error,
        *,
        create_issue: Optional[bool] = None,
    ) -> Dict[str, Any]:
        spec, context = self._lookup(error)
        plan, auto_issue = self._build_plan(spec, context, error)
        plan.update({
            "code": getattr(error, "code", None),
            "message": str(error),
        })
        plan.setdefault("issue_id", None)
        if create_issue is None:
            create_issue = auto_issue
        if create_issue and self.issue_client:
            details = {
                "code": getattr(error, "code", None),
                "message": str(error),
                "category": plan.get("category"),
                "context": plan.get("context"),
            }
            try:  # pragma: no cover - best-effort logging
                plan["issue_id"] = self.issue_client.post_event(
                    "mt5_error", details, severity=plan.get("severity", "error")
                )
            except Exception:
                logger.debug("IssueClient post_event failed", exc_info=True)
        return plan


__all__ = ["MT5IssueSolver", "KnowledgeEntry", "DEFAULT_KNOWLEDGE_BASE"]

