import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import requests
import streamlit as st
import yaml

from analytics.metrics_aggregator import query_metrics
from analytics.metrics_store import query_metrics as query_local_metrics
from analytics.regime_performance_store import RegimePerformanceStore
from analytics.issue_client import load_default as issue_client
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
        rows.append(
            {
                "parameter": name,
                "description": field.description or "",
                "default": default,
                "current": current.get(name, default),
            }
        )
    return rows


def _auth_headers(api_key: str) -> Dict[str, str]:
    return {"x-api-key": api_key}


def _ssl_opts():
    if CERT_PATH and Path(CERT_PATH).exists():
        return {"verify": CERT_PATH}
    return {"verify": True}


def fetch_json(path: str, api_key: str) -> Dict[str, Any]:
    try:
        resp = requests.get(
            f"{API_URL}{path}", headers=_auth_headers(api_key), **_ssl_opts()
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {}


def post_json(path: str, api_key: str):
    try:
        resp = requests.post(
            f"{API_URL}{path}", headers=_auth_headers(api_key), **_ssl_opts()
        )
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

    if st.sidebar.button("Export state"):
        try:
            archive_path = (
                subprocess.check_output(["bash", "scripts/export_state.sh"], text=True)
                .strip()
            )
            with open(archive_path, "rb") as f:
                st.sidebar.download_button(
                    label="Download project state",
                    data=f,
                    file_name=os.path.basename(archive_path),
                    mime="application/gzip",
                )
            os.remove(archive_path)
        except Exception as exc:  # pragma: no cover - GUI feedback only
            st.sidebar.error(f"Export failed: {exc}")

    uploaded = st.sidebar.file_uploader("Upload state archive", type="tar.gz")
    if st.sidebar.button("Import state") and uploaded:
        if uploaded.size > 1_000_000_000:
            st.sidebar.error("Archive too large")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tar.gz") as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = tmp.name
            try:
                subprocess.check_call(["bash", "scripts/import_state.sh", tmp_path])
                load_current_config.clear()
                st.sidebar.success("Import complete. Reloading...")
                st.experimental_rerun()
            except Exception as exc:  # pragma: no cover - GUI feedback only
                st.sidebar.error(f"Import failed: {exc}")
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    tabs = st.tabs(["Overview", "Performance", "Config Explorer", "Logs", "Traces"])

    # Overview tab
    with tabs[0]:
        metrics = fetch_json("/risk/status", api_key)
        fund_df = query_metrics("expected_funding_cost")
        margin_req_df = query_metrics("margin_required")
        margin_avail_df = query_metrics("margin_available")
        fund_val = float(fund_df["value"].iloc[-1]) if not fund_df.empty else 0.0
        margin_req_val = (
            float(margin_req_df["value"].iloc[-1]) if not margin_req_df.empty else 0.0
        )
        margin_avail_val = (
            float(margin_avail_df["value"].iloc[-1])
            if not margin_avail_df.empty
            else 0.0
        )
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
        col1.metric("PnL", metrics.get("daily_loss", 0))
        col2.metric("Exposure", metrics.get("exposure", 0))
        col3.metric("VaR", metrics.get("var", 0))
        col4.metric("Trading Halted", metrics.get("trading_halted", False))
        col5.metric("Funding Cost", fund_val)
        col6.metric("Margin Req.", margin_req_val)
        col7.metric("Free Margin", margin_avail_val)

        # Show per-broker performance metrics
        lat_df = query_local_metrics("broker_fill_latency_ms")
        slip_df = query_local_metrics("broker_slippage_bps")
        if not lat_df.empty or not slip_df.empty:
            st.subheader("Broker Latency/Slippage")
            lat = (
                lat_df.groupby("broker")["value"].mean()
                if not lat_df.empty
                else pd.Series()
            )
            slip = (
                slip_df.groupby("broker")["value"].mean()
                if not slip_df.empty
                else pd.Series()
            )
            table = pd.concat(
                [lat.rename("latency_ms"), slip.rename("slippage_bps")], axis=1
            )
            st.table(table.fillna(0))

        client = issue_client()
        open_issues = client.list_open()
        if open_issues:
            st.subheader("Open Issues")
            st.table(pd.DataFrame(open_issues)[["id", "event", "status"]])

        rate_df = query_metrics("tick_anomaly_rate")
        if not rate_df.empty:
            rate = float(rate_df["value"].iloc[-1])
            if rate > 0:
                st.error(f"Tick anomaly rate {rate:.2%}")

        exp_file = Path("reports/currency_exposure/latest.json")
        if exp_file.exists():
            try:
                exp_data = json.loads(exp_file.read_text())
                st.subheader("Currency-adjusted Exposure")
                st.bar_chart(pd.Series(exp_data))
            except Exception:
                pass

        expm_file = Path("reports/exposure_matrix/latest.json")
        if expm_file.exists():
            try:
                mat = pd.read_json(expm_file)
                if not mat.empty:
                    st.subheader("Exposure Heatmap")
                    try:
                        import seaborn as sns  # type: ignore
                        import matplotlib.pyplot as plt  # type: ignore

                        fig, ax = plt.subplots()
                        sns.heatmap(mat, ax=ax, cmap="RdBu", center=0)
                        st.pyplot(fig)
                    except Exception:
                        st.dataframe(mat)
            except Exception:
                pass

        bots = fetch_json("/bots", api_key)
        st.subheader("Running Bots")
        for bid, info in bots.items():
            c1, c2, c3 = st.columns([3, 1, 1])
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
                    file_name=f"{bid}.log",
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

        feat_file = Path("reports/feature_status/latest.json")
        if feat_file.exists():
            try:
                feat_data = json.loads(feat_file.read_text())
                st.subheader("Feature Availability")
                table = pd.DataFrame(feat_data.get("features", []))
                if not table.empty:
                    st.table(table[["name", "status"]])
                suggestion = feat_data.get("suggestion")
                if suggestion:
                    st.info(suggestion)
            except Exception:
                pass

        energy_file = Path("reports/energy/latest.json")
        if energy_file.exists():
            try:
                energy = json.loads(energy_file.read_text())
                st.subheader("Energy Usage")
                mods = energy.get("usage", {}).get("modules", {})
                for mod, stats in mods.items():
                    st.write(
                        f"{mod}: CPU {stats.get('cpu_avg',0):.1f}% | Power {stats.get('power_avg',0):.1f}"
                    )
                decisions = energy.get("decisions", [])
                if decisions:
                    st.warning(f"Throttling active: {', '.join(decisions)}")
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
        ret_df = query_metrics("return")
        sharpe_df = query_metrics("sharpe")
        draw_df = query_metrics("drawdown")
        if not ret_df.empty or not sharpe_df.empty or not draw_df.empty:
            st.subheader("Daily Metrics")
            if not ret_df.empty:
                st.line_chart(ret_df.set_index("timestamp")["value"], height=150)
            if not sharpe_df.empty:
                st.line_chart(sharpe_df.set_index("timestamp")["value"], height=150)
            if not draw_df.empty:
                st.line_chart(draw_df.set_index("timestamp")["value"], height=150)

        # Model vs. Market section
        reg_store = RegimePerformanceStore()
        reg_df = reg_store.load()
        corr_file = Path("reports/performance_correlations.parquet")
        corr_df = pd.read_parquet(corr_file) if corr_file.exists() else pd.DataFrame()
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
                lambda x: (
                    np.sqrt(252) * x.mean() / x.std(ddof=0) if x.std(ddof=0) else 0.0
                )
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
                        st.error("Correlation divergence: " + ", ".join(alerts))

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
                trans_perf = transitions.groupby(
                    [
                        "prev_regime",
                        "regime",
                    ]
                )["pnl_daily"].mean()
                st.subheader("Regime Transition PnL")
                st.table(
                    trans_perf.reset_index().rename(columns={"pnl_daily": "mean_pnl"})
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
            pivot = fid.pivot_table(
                index="timestamp", columns="feature", values="value"
            )
            st.line_chart(pivot, height=150)

        skipped = query_metrics("trades_skipped_news")
        if not skipped.empty:
            st.subheader("Trades Skipped Due To News")
            st.line_chart(skipped.set_index("timestamp")["value"], height=150)

        replay = query_metrics("replay_pnl_diff")
        if not replay.empty:
            st.subheader("Replay PnL Delta")
            st.line_chart(replay.set_index("timestamp")["value"], height=150)

        replay_dir = Path("reports/replays")
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

        risk_dir = Path("reports/replay_risk")
        risk_file = risk_dir / "risk_comparison.parquet"
        if risk_file.exists() or (risk_dir / "risk_comparison.csv").exists():
            try:
                if risk_file.exists():
                    rcomp = pd.read_parquet(risk_file)
                else:
                    rcomp = pd.read_csv(risk_dir / "risk_comparison.csv")
                if not rcomp.empty:
                    st.subheader("Replay Risk Comparison")
                    cols = st.columns(len(rcomp))
                    for col, row in zip(cols, rcomp.itertuples(index=False)):
                        col.metric(
                            row.metric.capitalize(),
                            f"{row.replay:.2f}",
                            f"{row.delta:+.2f}",
                        )
                    st.table(rcomp)
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

        # Basket/Strategy metrics -------------------------------------------------
        score_path = Path("reports/strategy_scores.parquet")
        if score_path.exists():
            try:
                sboard = pd.read_parquet(score_path).reset_index()
            except Exception:
                try:
                    sboard = pd.read_csv(score_path).reset_index()  # pragma: no cover
                except Exception:
                    sboard = pd.DataFrame()
        else:
            sboard = pd.DataFrame()

        if not sboard.empty:
            if "instrument" not in sboard.columns:
                sboard["instrument"] = "N/A"
            instruments = ["All"] + sorted(
                map(str, sboard["instrument"].dropna().unique())
            )
            inst = st.selectbox("Instrument", instruments, key="bs-inst")
            if inst != "All":
                sboard = sboard[sboard["instrument"].astype(str) == inst]

            trade_counts = pd.DataFrame()
            trades = pd.DataFrame()
            trade_path = Path("reports/trades.csv")
            if trade_path.exists():
                try:
                    trades = pd.read_csv(
                        trade_path,
                        parse_dates=["exit_time"],
                        infer_datetime_format=True,
                    )
                    if "model" in trades.columns and "algorithm" not in trades.columns:
                        trades.rename(columns={"model": "algorithm"}, inplace=True)
                    group_cols = [
                        c
                        for c in ["instrument", "market_basket", "algorithm"]
                        if c in trades.columns
                    ]
                    if group_cols:
                        trade_counts = (
                            trades.groupby(group_cols).size().reset_index(name="obs")
                        )
                        if "pnl" in trades.columns and "exit_time" in trades.columns:
                            recent = trades[
                                trades["exit_time"]
                                >= pd.Timestamp.utcnow() - pd.Timedelta(days=30)
                            ]
                            trend = (
                                recent.groupby(group_cols)["pnl"]
                                .sum()
                                .reset_index(name="pnl_30d")
                            )
                            trade_counts = trade_counts.merge(
                                trend, on=group_cols, how="left"
                            )
                except Exception:
                    trade_counts = pd.DataFrame()
                    trades = pd.DataFrame()

            if not trade_counts.empty:
                sboard = sboard.merge(
                    trade_counts,
                    on=[
                        c
                        for c in ["instrument", "market_basket", "algorithm"]
                        if c in sboard.columns and c in trade_counts.columns
                    ],
                    how="left",
                )

            sboard["obs"] = sboard.get("obs", 0).fillna(0).astype(int)
            sboard["pnl_30d"] = sboard.get("pnl_30d", 0.0).fillna(0.0)

            st.subheader("Basket/Strategy Metrics")

            # Alerts for insufficient data
            counts_by_basket = sboard.groupby("market_basket")["obs"].sum()
            low = counts_by_basket[counts_by_basket < 5]
            if not low.empty:
                st.warning(
                    "Insufficient data for baskets: " + ", ".join(map(str, low.index))
                )

            # Alert on top strategy change
            tops = (
                sboard.sort_values("sharpe", ascending=False)
                .dropna(subset=["sharpe"])
                .groupby("market_basket")
                .first()
            )
            prev_tops = st.session_state.get("_prev_tops", {})
            changes = []
            for basket, row in tops.iterrows():
                algo = row.get("algorithm")
                prev = prev_tops.get(basket)
                if prev and prev != algo:
                    changes.append(f"{basket}: {prev} -> {algo}")
                prev_tops[basket] = algo
            st.session_state["_prev_tops"] = prev_tops
            if changes:
                st.error("Top strategy changed: " + ", ".join(changes))

            # Heatmap of Sharpe ratios
            pivot = sboard.pivot_table(
                index="market_basket", columns="algorithm", values="sharpe"
            )
            try:
                import seaborn as sns  # type: ignore
                import matplotlib.pyplot as plt  # type: ignore

                fig, ax = plt.subplots()
                sns.heatmap(pivot, ax=ax, annot=True, cmap="RdBu", center=0)
                st.pyplot(fig)
            except Exception:
                st.dataframe(pivot)

            st.dataframe(sboard)
            csv = sboard.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Report", csv, file_name="basket_strategy_metrics.csv"
            )

            if (
                not trades.empty
                and "exit_time" in trades.columns
                and "pnl" in trades.columns
            ):
                baskets = sorted(sboard["market_basket"].dropna().unique())
                if baskets:
                    bsel = st.selectbox("Basket Trend", baskets, key="bs-trend")
                    btrades = trades[trades["market_basket"] == bsel]
                    if inst != "All" and "instrument" in btrades.columns:
                        btrades = btrades[btrades["instrument"].astype(str) == inst]
                    if not btrades.empty:
                        daily = (
                            btrades.groupby(
                                [pd.Grouper(key="exit_time", freq="D"), "algorithm"]
                            )["pnl"]
                            .sum()
                            .reset_index()
                        )
                        trend_pivot = daily.pivot_table(
                            index="exit_time", columns="algorithm", values="pnl"
                        )
                        st.line_chart(trend_pivot, height=150)

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
        st.download_button(
            "Download Logs", logs.get("logs", ""), file_name="system.log"
        )
        dec = read_decisions()
        if not dec.empty:
            cols = [
                c
                for c in [
                    "timestamp",
                    "event",
                    "Symbol",
                    "algorithm",
                    "position_size",
                    "reason",
                ]
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
                resp = requests.get(
                    f"{jaeger}/api/traces", params={"service": service, "limit": 20}
                )
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
                        line
                        for line in log_path.read_text().splitlines()
                        if tid and tid in line
                    ]
                    st.text("\n".join(lines))
                else:
                    st.write("No logs available")


if __name__ == "__main__":
    main()
