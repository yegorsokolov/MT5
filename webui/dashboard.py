import json
import os
from pathlib import Path
from typing import Any, Dict

import requests
import streamlit as st
import yaml

from analytics.metrics_store import MetricsStore, query_metrics

from config_schema import ConfigSchema

API_URL = os.getenv("REMOTE_API_URL", "https://localhost:8000")
CERT_PATH = os.getenv("API_CERT", "certs/api.crt")

@st.cache_data
def load_current_config() -> Dict[str, Any]:
    cfg_file = Path("config.yaml")
    if cfg_file.exists():
        with cfg_file.open() as f:
            return yaml.safe_load(f) or {}
    return {}

@st.cache_data
def schema_table(current: Dict[str, Any]):
    rows = []
    for name, field in ConfigSchema.model_fields.items():
        default = field.default if field.default is not None else "required"
        rows.append({
            "parameter": name,
            "description": field.description or "",
            "default": default,
            "current": current.get(name, default),
        })
    return rows


def _auth_headers(api_key: str) -> Dict[str, str]:
    return {"x-api-key": api_key}


def _ssl_opts():
    if CERT_PATH and Path(CERT_PATH).exists():
        return {"verify": CERT_PATH}
    return {"verify": True}


def fetch_json(path: str, api_key: str) -> Dict[str, Any]:
    try:
        resp = requests.get(f"{API_URL}{path}", headers=_auth_headers(api_key), **_ssl_opts())
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {}


def post_json(path: str, api_key: str):
    try:
        resp = requests.post(f"{API_URL}{path}", headers=_auth_headers(api_key), **_ssl_opts())
        resp.raise_for_status()
        return True
    except Exception:
        return False


def main() -> None:
    st.set_page_config(page_title="Trading Dashboard", layout="wide")
    st.title("Trading Bot Dashboard")

    api_key = st.sidebar.text_input("API Key", type="password")
    if not api_key:
        st.info("Enter API key to connect")
        return

    tabs = st.tabs(["Overview", "Performance", "Config Explorer", "Logs", "Traces"])

    # Overview tab
    with tabs[0]:
        metrics = fetch_json("/risk/status", api_key)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("PnL", metrics.get("daily_loss", 0))
        col2.metric("Exposure", metrics.get("exposure", 0))
        col3.metric("VaR", metrics.get("var", 0))
        col4.metric("Trading Halted", metrics.get("trading_halted", False))

        bots = fetch_json("/bots", api_key)
        st.subheader("Running Bots")
        for bid, info in bots.items():
            c1, c2, c3 = st.columns([3,1,1])
            running = info.get("running")
            c1.write(f"{bid} (restarts: {info.get('restart_count',0)})")
            if running:
                if c2.button("Pause", key=f"pause-{bid}"):
                    post_json(f"/bots/{bid}/stop", api_key)
            else:
                if c2.button("Resume", key=f"resume-{bid}"):
                    post_json(f"/bots/{bid}/start", api_key)
            if c3.button("Logs", key=f"logs-{bid}"):
                log_data = fetch_json(f"/bots/{bid}/logs", api_key)
                st.download_button(
                    label=f"Download {bid} logs",
                    data=log_data.get("logs", ""),
                    file_name=f"{bid}.log"
                )

        cp_file = Path("reports/change_points/latest.json")
        if cp_file.exists():
            try:
                cp_data = json.loads(cp_file.read_text())
                if any(cp_data.values()):
                    st.error(f"Change points detected: {cp_data}")
            except Exception:
                pass

    # Performance tab
    with tabs[1]:
        store = MetricsStore()
        df = store.load()
        if not df.empty:
            st.subheader("Daily Metrics")
            st.line_chart(df["return"], height=150)
            st.line_chart(df["sharpe"], height=150)
            st.line_chart(df["drawdown"], height=150)

        pnl = query_metrics("pnl", tags={"summary": "daily"})
        if not pnl.empty:
            st.subheader("PnL")
            st.line_chart(pnl.set_index("timestamp")["value"], height=150)

        latency = query_metrics("queue_depth")
        if not latency.empty:
            st.subheader("Queue Depth")
            st.line_chart(latency.set_index("timestamp")["value"], height=150)

        cpu = query_metrics("cpu_usage_pct")
        rss = query_metrics("rss_usage_mb")
        if not cpu.empty and not rss.empty:
            st.subheader("Resource Usage")
            st.line_chart(cpu.set_index("timestamp")["value"], height=150)
            st.line_chart(rss.set_index("timestamp")["value"], height=150)

        drift = query_metrics("drift_events")
        if not drift.empty:
            st.subheader("Drift Events")
            st.line_chart(drift.set_index("timestamp")["value"], height=150)

    # Config Explorer tab
    with tabs[2]:
        current = load_current_config()
        rows = schema_table(current)
        st.subheader("Config Explorer")
        st.markdown(
            "[Full config documentation](https://github.com/USERNAME/MT5/blob/main/docs/config.md)"
        )
        st.table(rows)

    # Logs tab
    with tabs[3]:
        if st.button("Refresh Logs"):
            st.session_state["logs"] = fetch_json("/logs", api_key)
        logs = st.session_state.get("logs", fetch_json("/logs", api_key))
        st.text_area("Recent Logs", logs.get("logs", ""), height=300)
        st.download_button("Download Logs", logs.get("logs", ""), file_name="system.log")

    # Traces tab
    with tabs[4]:
        jaeger = os.getenv("JAEGER_QUERY_URL", "http://localhost:16686")
        service = st.text_input("Service", "mt5")
        if st.button("Load Traces"):
            try:
                resp = requests.get(f"{jaeger}/api/traces", params={"service": service, "limit": 20})
                resp.raise_for_status()
                st.session_state["trace_data"] = resp.json().get("data", [])
            except Exception:
                st.session_state["trace_data"] = []
        traces = st.session_state.get("trace_data", [])
        for tr in traces:
            tid = tr.get("traceID")
            st.write(f"Trace {tid}")
            if st.button("Show Logs", key=f"tlog-{tid}"):
                log_path = Path("logs/app.log")
                if log_path.exists():
                    lines = [
                        line for line in log_path.read_text().splitlines() if tid and tid in line
                    ]
                    st.text("\n".join(lines))
                else:
                    st.write("No logs available")


if __name__ == "__main__":
    main()
