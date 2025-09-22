import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

sys.modules.pop("utils.alerting", None)
sys.modules.pop("config_models", None)
sys.modules.pop("pydantic", None)
sys.modules.pop("utils", None)

import utils  # noqa: E402  # pylint: disable=wrong-import-position
from mt5.config_models import AlertingConfig, AppConfig, StrategyConfig  # noqa: E402
from utils import alerting as alert_mod  # noqa: E402


def test_send_alert_uses_configured_telegram(monkeypatch: pytest.MonkeyPatch) -> None:
    token = "123456:ABCDEF"
    chat_id = "987654321"
    cfg = AppConfig(
        strategy=StrategyConfig(symbols=["EURUSD"], risk_per_trade=0.1),
        alerting=AlertingConfig(
            telegram_bot_token=token,
            telegram_chat_id=chat_id,
        ),
    )

    monkeypatch.setattr(alert_mod, "load_config", lambda: cfg)

    calls: list[tuple[str, dict[str, object]]] = []

    def fake_post(url: str, **kwargs: object) -> None:
        calls.append((url, kwargs))
        return None

    monkeypatch.setattr(alert_mod.requests, "post", fake_post)

    class _NoSMTP:
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401
            raise AssertionError("SMTP should not be used when not configured")

    monkeypatch.setattr(alert_mod.smtplib, "SMTP", _NoSMTP)

    alert_mod.send_alert("Alert message")

    assert calls
    url, kwargs = calls[0]
    assert url == f"https://api.telegram.org/bot{token}/sendMessage"
    assert kwargs["json"] == {"chat_id": chat_id, "text": "Alert message"}
    assert calls[0][1]["timeout"] == 5
