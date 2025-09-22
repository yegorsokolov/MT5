import logging
import smtplib
from collections.abc import Mapping, Sequence
from email.message import EmailMessage
from typing import Any

import requests

from . import load_config

logger = logging.getLogger(__name__)

_SMTP_ATTR_ALIASES = {"from": "sender", "to": "recipients", "starttls": "use_tls"}


def _as_mapping(value: Any) -> dict[str, Any]:
    """Return ``value`` as a mapping when possible."""

    if isinstance(value, Mapping):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            data = model_dump(by_alias=True)
        except TypeError:
            data = model_dump()
        if isinstance(data, Mapping):
            return dict(data)
    dict_method = getattr(value, "dict", None)
    if callable(dict_method):
        try:
            data = dict_method(by_alias=True)
        except TypeError:
            data = dict_method()
        if isinstance(data, Mapping):
            return dict(data)
    return {}


def send_alert(msg: str) -> None:
    """Send ``msg`` via Telegram bot and/or SMTP email.

    Reads configuration from ``config.yaml`` under the ``alerting`` key. Both
    Telegram and email can be configured simultaneously; failures are logged but
    otherwise ignored.
    """
    cfg = load_config()
    alert_cfg = cfg.get("alerting")
    alert_data = _as_mapping(alert_cfg)
    telegram_token: str | None = alert_data.get("telegram_bot_token")
    if telegram_token is None and alert_cfg is not None:
        telegram_token = getattr(alert_cfg, "telegram_bot_token", None)

    telegram_chat_id: str | None = alert_data.get("telegram_chat_id")
    if telegram_chat_id is None and alert_cfg is not None:
        telegram_chat_id = getattr(alert_cfg, "telegram_chat_id", None)

    smtp_cfg_obj = getattr(alert_cfg, "smtp", None) if alert_cfg is not None else None
    if smtp_cfg_obj is None:
        smtp_cfg_obj = alert_data.get("smtp")
    smtp_cfg = _as_mapping(smtp_cfg_obj)

    def _smtp_field(*names: str, default: Any | None = None) -> Any:
        for name in names:
            if name in smtp_cfg:
                return smtp_cfg[name]
            attr_name = _SMTP_ATTR_ALIASES.get(name, name)
            if smtp_cfg_obj is not None and hasattr(smtp_cfg_obj, attr_name):
                return getattr(smtp_cfg_obj, attr_name)
        return default

    sent = False

    if telegram_token and telegram_chat_id:
        try:
            requests.post(
                f"https://api.telegram.org/bot{telegram_token}/sendMessage",
                json={"chat_id": telegram_chat_id, "text": msg},
                timeout=5,
            )
            sent = True
        except Exception:
            logger.exception("Failed to send Telegram alert")

    host = _smtp_field("host")
    recipients_raw = _smtp_field("to", "recipients")
    recipients: list[str]
    if isinstance(recipients_raw, str):
        recipients = [addr.strip() for addr in recipients_raw.split(",") if addr.strip()]
    elif isinstance(recipients_raw, Sequence):
        recipients = [str(addr) for addr in recipients_raw if str(addr)]
    else:
        recipients = []

    if host and recipients:
        sender = _smtp_field("from", "sender", default="alerts@example.com")
        port = _smtp_field("port", default=587)
        try:
            port = int(port)
        except (TypeError, ValueError):
            port = 587
        use_tls = _smtp_field("starttls", "use_tls")
        use_tls = True if use_tls is None else bool(use_tls)
        username = _smtp_field("username")
        password = _smtp_field("password")
        try:
            em = EmailMessage()
            em.set_content(msg)
            em["Subject"] = "MT5 Alert"
            em["From"] = sender if sender is not None else "alerts@example.com"
            em["To"] = ", ".join(recipients)
            with smtplib.SMTP(host, port, timeout=10) as s:
                if use_tls:
                    s.starttls()
                if username:
                    s.login(username, password)
                s.send_message(em)
            sent = True
        except Exception:
            logger.exception("Failed to send email alert")

    if not sent:
        logger.warning("No alert sent for message: %s", msg)
