import logging
import smtplib
from email.message import EmailMessage
from typing import Any, Dict

import requests

from . import load_config

logger = logging.getLogger(__name__)


def send_alert(msg: str) -> None:
    """Send ``msg`` via Slack webhook and/or SMTP email.

    Reads configuration from ``config.yaml`` under the ``alerting`` key. Both
    Slack and email can be configured simultaneously; failures are logged but
    otherwise ignored.
    """
    cfg = load_config()
    alert_cfg: Dict[str, Any] = cfg.get("alerting", {}) or {}
    slack_url: str | None = alert_cfg.get("slack_webhook")
    smtp_cfg: Dict[str, Any] = alert_cfg.get("smtp", {}) or {}

    sent = False

    if slack_url:
        try:
            requests.post(slack_url, json={"text": msg}, timeout=5)
            sent = True
        except Exception:
            logger.exception("Failed to send Slack alert")

    host = smtp_cfg.get("host")
    recipient = smtp_cfg.get("to")
    if host and recipient:
        try:
            em = EmailMessage()
            em.set_content(msg)
            em["Subject"] = "MT5 Alert"
            em["From"] = smtp_cfg.get("from", "alerts@example.com")
            em["To"] = recipient
            with smtplib.SMTP(host, int(smtp_cfg.get("port", 587)), timeout=10) as s:
                if smtp_cfg.get("starttls", True):
                    s.starttls()
                user = smtp_cfg.get("username")
                if user:
                    s.login(user, smtp_cfg.get("password"))
                s.send_message(em)
            sent = True
        except Exception:
            logger.exception("Failed to send email alert")

    if not sent:
        logger.warning("No alert sent for message: %s", msg)
