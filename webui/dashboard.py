import json
import os
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import requests
import streamlit as st
import yaml

from analytics.metrics_store import MetricsStore, query_metrics
from analytics.regime_performance_store import RegimePerformanceStore
from log_utils import read_decisions

from config_schema import ConfigSchema
import numpy as np

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

        fid_path = Path("reports/feature_drift/latest.json")
        if fid_path.exists():
            try:
                fid_data = json.loads(fid_path.read_text())
                flagged = fid_data.get("flagged", {})
                if flagged:
                    msg = ", ".join(f"{k} ({v:.2f})" for k, v in flagged.items())
                    st.error(f"Feature importance drift: {msg}")
            except Exception:
                pass

        st.subheader("Backup Status")
        bstatus = Path("reports/backup_status.json")
        if bstatus.exists():
            try:
                data = json.loads(bstatus.read_text())
                st.write(f"Last Run: {data.get('last_run', 'n/a')}")
                st.write(f"Success: {data.get('last_success', False)}")
            except Exception:
                st.write("Failed to load status")
        else:
            st.write("No backup information available")

    # Performance tab
    with tabs[1]:
        store = MetricsStore()
        df = store.load()
        if not df.empty:
            st.subheader("Daily Metrics")
            st.line_chart(df["return"], height=150)
            st.line_chart(df["sharpe"], height=150)
            st.line_chart(df["drawdown"], height=150)

        # Model vs. Market section
        reg_store = RegimePerformanceStore()
        reg_df = reg_store.load()
        corr_file = Path("reports/performance_correlations.parquet")
        corr_df = (
            pd.read_parquet(corr_file) if corr_file.exists() else pd.DataFrame()
        )
        if not reg_df.empty:
            st.subheader("Model vs. Market")
            regimes = sorted(reg_df["regime"].unique())
            mv_regime = st.selectbox("Regime", regimes, key="mv-regime")
            sub = reg_df[reg_df["regime"] == mv_regime]
            pnl_pivot = sub.pivot_table(
                index="date", columns="model", values="pnl_daily"
            ).cumsum()
            st.line_chart(pnl_pivot, height=150)

            sharpe_vals = sub.groupby("model")["pnl_daily"].apply(
                lambda x: np.sqrt(252) * x.mean() / x.std(ddof=0)
                if x.std(ddof=0)
                else 0.0
            )
            st.bar_chart(sharpe_vals, height=150)

            if not corr_df.empty:
                corr_sub = corr_df[corr_df["regime"] == mv_regime]
                mv_model = st.selectbox(
                    "Model", sorted(sub["model"].unique()), key="mv-model"
                )
                latest_corr = (
                    corr_sub[corr_sub["algorithm"] == mv_model]
                    .sort_values("timestamp")
                    .groupby("feature")
                    .tail(1)
                )
                if not latest_corr.empty:
                    st.bar_chart(
                        latest_corr.set_index("feature")["pearson"], height=150
                    )
                    hist = corr_df[corr_df["algorithm"] == mv_model]
                    alerts = []
                    for feat, row in latest_corr.set_index("feature").iterrows():
                        series = hist[hist["feature"] == feat]["pearson"]
                        if len(series) >= 2:
                            mean = series.mean()
                            std = series.std(ddof=0)
                            if std > 0 and abs(row["pearson"] - mean) > 2 * std:
                                alerts.append(
                                    f"{feat} ({row['pearson']:.2f} vs {mean:.2f})"
                                )
                    if alerts:
                        st.error(
                            "Correlation divergence: " + ", ".join(alerts)
                        )

            # Drill-down metrics for each model
            dd_model = st.selectbox(
                "Drill-down Model", sorted(sub["model"].unique()), key="mv-drill"
            )
            mdf = reg_df[reg_df["model"] == dd_model].sort_values("date")
            win_rate = float((mdf["pnl_daily"] > 0).mean())
            cumulative = mdf["pnl_daily"].cumsum()
            drawdown = float((cumulative - cumulative.cummax()).min())
            st.write(f"Win rate: {win_rate:.2%}")
            st.write(f"Max drawdown: {drawdown:.2f}")

            mdf["prev_regime"] = mdf["regime"].shift(1)
            transitions = mdf[
                mdf["prev_regime"].notna() & (mdf["prev_regime"] != mdf["regime"])
            ]
            if not transitions.empty:
                trans_perf = transitions.groupby([
                    "prev_regime",
                    "regime",
                ])["pnl_daily"].mean()
                st.subheader("Regime Transition PnL")
                st.table(
                    trans_perf.reset_index().rename(
                        columns={"pnl_daily": "mean_pnl"}
                    )
                )

        pnl = query_metrics("pnl", tags={"summary": "daily"})
        if not pnl.empty:
            st.subheader("PnL")
            st.line_chart(pnl.set_index("timestamp")["value"], height=150)

        hold_path = Path("reports/hold_duration/pnl_by_duration.csv")
        if hold_path.exists():
            try:
                hdf = pd.read_csv(hold_path).sort_values("duration_min")
                st.subheader("PnL by Holding Period (min)")
                st.bar_chart(hdf.set_index("duration_min")["pnl"], height=150)
            except Exception:
                pass

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

        fid = query_metrics("feature_importance_drift")
        if not fid.empty:
            st.subheader("Feature Importance Drift")
            pivot = fid.pivot_table(index="timestamp", columns="feature", values="value")
            st.line_chart(pivot, height=150)

        skipped = query_metrics("trades_skipped_news")
        if not skipped.empty:
            st.subheader("Trades Skipped Due To News")
            st.line_chart(skipped.set_index("timestamp")["value"], height=150)

        replay = query_metrics("replay_pnl_diff")
        if not replay.empty:
            st.subheader("Replay PnL Delta")
            st.line_chart(replay.set_index("timestamp")["value"], height=150)

        replay_dir = Path("reports/replay")
        pnl_summary = replay_dir / "pnl_summary.parquet"
        model_summary = replay_dir / "summary.parquet"
        flag_file = replay_dir / "latest.json"
        if pnl_summary.exists():
            try:
                ps = pd.read_parquet(pnl_summary)
                st.subheader("Replay PnL Summary")
                st.table(ps)
            except Exception:
                pass
        if model_summary.exists():
            try:
                ms = pd.read_parquet(model_summary)
                st.subheader("Model Replay MAE")
                st.table(ms)
            except Exception:
                pass
        if flag_file.exists():
            try:
                data = json.loads(flag_file.read_text())
                flagged = data.get("flagged", {})
                if flagged:
                    msg = ", ".join(f"{k} ({v:.3f})" for k, v in flagged.items())
                    st.error(f"Replay discrepancies: {msg}")
            except Exception:
                pass

        corr_path = Path("reports/performance_correlations.parquet")
        if corr_path.exists():
            try:
                cdf = pd.read_parquet(corr_path)
                if not cdf.empty:
                    st.subheader("Feature-PnL Correlations")
                    regimes = sorted(cdf["regime"].unique())
                    regime = st.selectbox("Regime", regimes)
                    latest = (
                        cdf[cdf["regime"] == regime]
                        .sort_values("timestamp")
                        .groupby(["algorithm", "feature"])
                        .tail(1)
                    )
                    pivot = latest.pivot_table(
                        index="feature", columns="algorithm", values="pearson"
                    )
                    try:
                        import seaborn as sns  # type: ignore
                        import matplotlib.pyplot as plt  # type: ignore

                        fig, ax = plt.subplots()
                        sns.heatmap(pivot, ax=ax, annot=True, cmap="RdBu", center=0)
                        st.pyplot(fig)
                    except Exception:
                        st.dataframe(pivot)

                    st.subheader("Correlation History")
                    hist = cdf[cdf["regime"] == regime]
                    for feat in hist["feature"].unique():
                        fdf = hist[hist["feature"] == feat]
                        pivot_hist = fdf.pivot(
                            index="timestamp", columns="algorithm", values="pearson"
                        )
                        st.line_chart(pivot_hist, height=150)
            except Exception:
                pass

        for name in ["gaps", "zscore", "median"]:
            dq = query_metrics(f"data_quality_{name}")
            if not dq.empty:
                st.subheader(f"Data Quality - {name.title()}")
                st.line_chart(dq.set_index("timestamp")["value"], height=150)

    # Config Explorer tab
    with tabs[2]:
        current = load_current_config()
        rows = schema_table(current)
        st.subheader("Config Explorer")
        st.markdown(
            "[Full config documentation](https://github.com/USERNAME/MT5/blob/main/docs/config.md)"
        )
        st.table(rows)
        cards_dir = Path("reports/model_cards")
        if cards_dir.exists():
            st.subheader("Model Cards")
            for card in sorted(cards_dir.glob("model_card_*.md")):
                st.markdown(f"[{card.name}]({card.resolve().as_uri()})")

    # Logs tab
    with tabs[3]:
        if st.button("Refresh Logs"):
            st.session_state["logs"] = fetch_json("/logs", api_key)
        logs = st.session_state.get("logs", fetch_json("/logs", api_key))
        st.text_area("Recent Logs", logs.get("logs", ""), height=300)
        st.download_button("Download Logs", logs.get("logs", ""), file_name="system.log")
        dec = read_decisions()
        if not dec.empty:
            cols = [
                c
                for c in ["timestamp", "event", "Symbol", "algorithm", "position_size", "reason"]
                if c in dec.columns
            ]
            st.subheader("Recent Decisions")
            st.dataframe(dec.tail(50)[cols])

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
