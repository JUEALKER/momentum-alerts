# streamlit_app.py — Build: SR-smooth-refresh-v2 (admin-pin)
import os, time
from datetime import datetime
import numpy as np
import pandas as pd
import requests
import pytz
import streamlit as st
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

BUILD = "SR-smooth-refresh-v2"

# -------------------- PAGE SETUP --------------------
st.set_page_config(page_title="Momentum Signals", layout="wide")
st.title("📈 Momentum Signals & Market Bias")

# -------------------- ADMIN MODE (URL PIN) --------------------
def _get_query_params():
    # Streamlit >= 1.30
    try:
        return dict(st.query_params)
    except Exception:
        # Older fallback
        return st.experimental_get_query_params()

def is_admin() -> bool:
    """
    Admin mode OFF by default.
    ON only if secrets has ADMIN_PIN and URL query param ?admin=<ADMIN_PIN> is present
    (sets a session flag). Hidden for normal visitors.
    """
    pin = str(st.secrets.get("ADMIN_PIN", "")).strip()
    if not pin:
        return False
    qp = _get_query_params()
    if "admin" in qp and str(qp["admin"]) == pin:
        st.session_state["_is_admin"] = True
    return bool(st.session_state.get("_is_admin", False))

IS_ADMIN = is_admin()

# -------------------- SECRET HELPERS --------------------
def get_secret(name: str, default: str = "") -> str:
    v = st.secrets.get(name) if hasattr(st, "secrets") else None
    if v is None:
        v = os.getenv(name, default)
    return str(v).strip()

TG_TOKEN = get_secret("TELEGRAM_BOT_TOKEN")
TG_CHAT_ID = get_secret("TELEGRAM_CHAT_ID")

# -------------------- SESSION DEFAULTS --------------------
for key, val in {
    "prev_bias": {},
    "last_alert_ts": {},
    "last_snapshot": None,
    "last_bias_sig": "",
    "heat_fig": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.header("Settings")
    default_assets = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"]
    assets = st.multiselect("Assets", default_assets, default=default_assets)
    tfs = st.multiselect("Timeframes", ["5m", "1h", "4h"], default=["5m", "1h", "4h"])

    entry_long = st.slider("Long threshold", 50, 80, 60, 1)
    entry_short = st.slider("Short threshold", 20, 60, 40, 1)
    spark_len = st.slider("Sparkline length", 50, 200, 100, 10)

    st.subheader("Display")
    show_summary = st.toggle("Show Summary Table", value=True)
    show_asset_insights = st.toggle("Show Per-Asset Insights", value=True)
    show_heatgrid = st.toggle("Show Heat Grid", value=True)
    show_sparklines = st.toggle("Show Live Cards (with Sparklines)", value=True)

    st.subheader("Auto-refresh")
    auto_refresh = st.toggle("Auto-refresh every 60s (soft)", value=True)
    if st.button("🔁 Soft refresh now"):
        st.experimental_rerun()

    st.markdown("---")
    st.subheader("Filter")
    bias_filter = st.multiselect(
        "Show biases",
        ["🟢 LONG", "⚪ NEUTRAL", "🔴 SHORT"],
        default=["🟢 LONG", "⚪ NEUTRAL", "🔴 SHORT"]
    )
    min_weight_filter = st.slider("Min Weight (summary filter)", 0.0, 1.0, 0.00, 0.05)
    sort_choice = st.selectbox(
        "Sort summary table by",
        ["Default (BTC→ETH→BNB→SOL→XRP + TF order)", "Weight (desc)", "Score (desc)", "Asset A→Z"],
        index=0
    )

    st.markdown("---")
    st.subheader("Telegram Alerts")
    alerts_enabled = st.toggle("Enable Telegram alerts on bias change (UI runtime)", value=True)
    min_weight_for_alert = st.slider("Min Weight for alert", 0.0, 1.0, 0.30, 0.05)
    cooldown_min = st.slider("Cooldown per signal (min)", 0, 120, 15, 5)

    # Public status only (no public test button)
    if TG_TOKEN and TG_CHAT_ID:
        st.success("Telegram configured via env ✅")
    else:
        st.warning("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in environment to send alerts.", icon="⚠️")

    # Admin-only maintenance/test (hidden unless ?admin=PIN)
    if IS_ADMIN:
        with st.expander("🛠 Maintenance (admin)"):
            st.caption("Only visible in admin mode (?admin=PIN).")
            if st.button("Send private test message (admin)"):
                ok = False
                try:
                    ok = requests.post(
                        f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                        json={"chat_id": TG_CHAT_ID, "text": "✅ Admin test: Streamlit can send Telegram messages.", "parse_mode": "HTML"},
                        timeout=10
                    ).ok
                except Exception:
                    ok = False
                st.toast("Sent." if ok else "Failed.", icon="✅" if ok else "⚠️")

    st.caption(f"Running file: `{__file__}`")

# Soft auto-refresh (no white flash)
if auto_refresh:
    st_autorefresh(interval=60_000, key="soft-refresh")

# -------------------- PLACEHOLDERS (smooth swap) --------------------
kpi_ph = st.empty()
summary_ph = st.empty()
insight_ph = st.empty()
asset_insights_ph = st.empty()
heat_ph = st.empty()
cards_ph = st.empty()
footer_ph = st.empty()

# -------------------- DATA HELPERS --------------------
BINANCE_SPOT = [
    "https://api.binance.com", "https://api1.binance.com",
    "https://api2.binance.com", "https://api3.binance.com",
    "https://data-api.binance.vision"
]
BINANCE_FUT = [
    "https://fapi.binance.com", "https://fapi1.binance.com",
    "https://fapi2.binance.com", "https://fapi3.binance.com"
]
KRAKEN_PAIR = {
    "BTC/USDT": "XBTUSDT", "ETH/USDT": "ETHUSDT",
    "BNB/USDT": "BNBUSDT", "SOL/USDT": "SOLUSDT", "XRP/USDT": "XRPUSDT"
}

def sym_to_perp(sym): return sym.replace("/", "")
def tf_to_kraken(tf): return {"5m": 5, "1h": 60, "4h": 240}.get(tf, 60)

def send_telegram(text: str) -> bool:
    if not (TG_TOKEN and TG_CHAT_ID): return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=10
        )
        r.raise_for_status()
        return True
    except Exception:
        return False

@st.cache_data(ttl=60)
def fetch_binance(perp, interval, limit=1000):
    for base in BINANCE_SPOT:
        try:
            r = requests.get(f"{base}/api/v3/klines",
                             params={"symbol": perp, "interval": interval, "limit": limit}, timeout=15)
            r.raise_for_status()
            data = r.json()
            cols = ["t","o","h","l","c","v","ct","qa","nt","tb","tq","ig"]
            df = pd.DataFrame(data, columns=cols)
            df["time"] = pd.to_datetime(df["ct"], unit="ms", utc=True)
            out = df.set_index("time")[["o","h","l","c","v"]].astype(float)
            out.columns = ["open","high","low","close","volume"]
            return out
        except:
            continue
    raise RuntimeError("Binance unavailable")

@st.cache_data(ttl=60)
def fetch_kraken(sym, tf, limit=1000):
    kp = KRAKEN_PAIR.get(sym)
    r = requests.get("https://api.kraken.com/0/public/OHLC",
                     params={"pair": kp, "interval": tf_to_kraken(tf)}, timeout=15)
    r.raise_for_status()
    data = r.json()
    key = next(iter(data["result"]))
    df = pd.DataFrame(data["result"][key],
                      columns=["t","o","h","l","c","vwap","vol","count"])
    df["time"] = pd.to_datetime(df["t"], unit="s", utc=True)
    out = df.set_index("time")[["o","h","l","c","vol"]].astype(float)
    out.columns = ["open","high","low","close","volume"]
    return out.tail(limit)

@st.cache_data(ttl=60)
def fetch_ohlcv(sym, tf, limit=1000):
    try:
        return fetch_binance(sym_to_perp(sym), tf, limit)
    except:
        return fetch_kraken(sym, tf, limit)

@st.cache_data(ttl=60)
def fetch_funding(perp):
    for base in BINANCE_FUT:
        try:
            r = requests.get(f"{base}/fapi/v1/fundingRate", params={"symbol": perp, "limit": 1}, timeout=12)
            r.raise_for_status()
            j = r.json()
            if isinstance(j, list) and j:
                return float(j[-1]["fundingRate"])
        except:
            continue
    return np.nan

@st.cache_data(ttl=10)
def fetch_last_price_binance(perp: str) -> float:
    r = requests.get("https://api.binance.com/api/v3/ticker/price",
                     params={"symbol": perp}, timeout=10)
    r.raise_for_status()
    return float(r.json()["price"])

@st.cache_data(ttl=10)
def fetch_last_price_kraken(sym: str) -> float:
    kp = KRAKEN_PAIR.get(sym)
    r = requests.get("https://api.kraken.com/0/public/Ticker", params={"pair": kp}, timeout=10)
    r.raise_for_status()
    data = r.json()["result"]
    key = next(iter(data))
    return float(data[key]["c"][0])

@st.cache_data(ttl=10)
def fetch_spot_last(sym: str) -> float:
    perp = sym.replace("/", "")
    try:
        return fetch_last_price_binance(perp)
    except:
        try:
            return fetch_last_price_kraken(sym)
        except:
            return float("nan")

# -------------------- INDICATORS --------------------
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def atr(df, n=20):
    pc = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - pc).abs(),
        (df["low"] - pc).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()
def zscore(s, n=90):
    m, v = s.rolling(n).mean(), s.rolling(n).std(ddof=0)
    return (s - m) / v.replace(0, np.nan)
def donchian(df, n=20): return df["high"].rolling(n).max(), df["low"].rolling(n).min()

def compute_score(df):
    ema50, ema200 = ema(df["close"], 50), ema(df["close"], 200)
    trend = (df["close"] > ema50) & (ema50 > ema200)
    dc_hi, _ = donchian(df, 20)
    breakout = df["close"] > dc_hi.shift(1)
    volZ = zscore(df["volume"], 90).fillna(0)
    vol_conf = volZ > 0.5
    a = atr(df, 20)
    vol_regime = ((a/df["close"]) > (a/df["close"]).rolling(90).median()).fillna(False)
    raw = 0.3*trend + 0.3*breakout + 0.2*vol_conf + 0.2*vol_regime
    return (100*raw.clip(0, 1)).fillna(0)

def bias_and_weight(score, long_t, short_t):
    if score >= long_t:
        w = min(1.0, (score - long_t) / (100 - long_t + 1e-9))
        return "🟢 LONG", round(w, 2)
    if score <= short_t:
        w = min(1.0, (short_t - score) / (short_t + 1e-9))
        return "🔴 SHORT", round(w, 2)
    return "⚪ NEUTRAL", 0.0

def funding_alignment(bias, fr):
    if pd.isna(fr): return "–"
    if bias == "🟢 LONG" and fr <= 0: return "✅ aligned"
    if bias == "🔴 SHORT" and fr >= 0: return "✅ aligned"
    if bias == "⚪ NEUTRAL": return "•"
    return "⚠️"

TF_ORDER = {"5m": 0, "1h": 1, "4h": 2}
ASSET_ORDER = {"BTC/USDT": 0, "ETH/USDT": 1, "BNB/USDT": 2, "SOL/USDT": 3, "XRP/USDT": 4}

def tf_consensus(rows_for_asset: pd.DataFrame):
    if rows_for_asset.empty:
        return "⚪ NEUTRAL", 0, 0, 0.0
    vc = rows_for_asset["Bias"].value_counts()
    dom = vc.idxmax()
    agree = int(vc.max())
    total = len(rows_for_asset)
    avg_w = float(rows_for_asset["Weight"].mean())
    return dom, agree, total, avg_w

# -------------------- BUILD SNAPSHOT --------------------
snapshot = None
errors = []

with st.spinner("Updating…"):
    try:
        # 1) Build table rows
        rows = []
        now_epoch = time.time()
        for sym in assets:
            perp = sym_to_perp(sym)
            live_price = fetch_spot_last(sym)
            fr = fetch_funding(perp)
            funding_pct = "-" if pd.isna(fr) else round(fr*100, 3)
            for tf in tfs:
                try:
                    df = fetch_ohlcv(sym, tf, 1000)
                    sc = compute_score(df)
                    s = float(sc.iloc[-1])
                    price_fallback = float(df["close"].iloc[-1])
                    price = live_price if not np.isnan(live_price) else price_fallback
                    last_time = sc.index[-1]
                    bias, w = bias_and_weight(s, entry_long, entry_short)
                    align = funding_alignment(bias, fr)

                    # UI alerts (session-scoped, not CI)
                    if alerts_enabled and TG_TOKEN and TG_CHAT_ID:
                        key = f"{sym}|{tf}"
                        prev = st.session_state.prev_bias.get(key)
                        if prev is None:
                            st.session_state.prev_bias[key] = bias
                        else:
                            cooldown_ok = (now_epoch - st.session_state.last_alert_ts.get(key, 0)) >= cooldown_min*60
                            if bias != prev and w >= min_weight_for_alert and cooldown_ok:
                                msg = (f"⚡ Momentum change on <b>{sym}</b> ({tf})\n"
                                       f"{prev} → <b>{bias}</b>\n"
                                       f"Weight: <b>{w:.2f}</b> | Score: <b>{s:.1f}</b>\n"
                                       f"Price: <b>{price:.2f}</b>\n"
                                       f"Funding: <b>{'-' if funding_pct == '-' else str(funding_pct)+'%'}</b>\n"
                                       f"Time: {last_time.tz_convert('Europe/Berlin').strftime('%Y-%m-%d %H:%M')}")
                                if send_telegram(msg):
                                    st.session_state.last_alert_ts[key] = now_epoch
                                    st.session_state.prev_bias[key] = bias

                    spark_vals = df["close"].tail(spark_len).tolist()
                    delta = spark_vals[-1] - spark_vals[0] if len(spark_vals) > 1 else 0
                    spark_color = "green" if delta > 0 else "red" if delta < 0 else "gray"

                    rows.append({
                        "Asset": sym, "TF": tf, "Price": round(price, 2),
                        "Funding %": funding_pct, "Score": round(s, 1),
                        "Bias": bias, "Weight": w, "Alignment": align,
                        "TrendColor": spark_color, "Spark": spark_vals,
                        "Last (Berlin)": last_time.tz_convert("Europe/Berlin").strftime("%Y-%m-%d %H:%M")
                    })
                except Exception as e:
                    errors.append((sym, tf, str(e)))
                    rows.append({
                        "Asset": sym, "TF": tf, "Price": "-", "Funding %": "-", "Score": "-",
                        "Bias": "ERR", "Weight": 0, "Alignment": "–", "TrendColor": "gray", "Spark": [],
                        "Last (Berlin)": str(e)
                    })

        table = pd.DataFrame(rows)

        # 2) KPI
        kpis = None
        if not table.empty:
            kpis = {
                "long": int((table["Bias"] == "🟢 LONG").sum()),
                "short": int((table["Bias"] == "🔴 SHORT").sum()),
                "neutral": int((table["Bias"] == "⚪ NEUTRAL").sum())
            }

        # 3) Summary w/ consensus
        summary = None
        if show_summary and not table.empty:
            tbl = table[table["Bias"].isin(bias_filter)].copy()
            tbl = tbl[tbl["Weight"] >= min_weight_filter]
            tbl["TF_sort"] = tbl["TF"].map(TF_ORDER).fillna(999)
            tbl["Asset_sort"] = tbl["Asset"].map(ASSET_ORDER).fillna(999)

            cons_rows = []
            for asset in tbl["Asset"].unique():
                sub = tbl[tbl["Asset"] == asset]
                dom, agree, total, avg_w = tf_consensus(sub)
                cons_rows.append({"Asset": asset, "Consensus": f"{agree}/{total} {dom}", "AvgW": round(avg_w, 2)})
            cons_df = pd.DataFrame(cons_rows) if cons_rows else pd.DataFrame(columns=["Asset","Consensus","AvgW"])
            tbl = tbl.merge(cons_df, on="Asset", how="left")

            if sort_choice == "Weight (desc)":
                tbl = tbl.sort_values(["Weight", "Asset", "TF"], ascending=[False, True, True])
            elif sort_choice == "Score (desc)":
                tbl = tbl.sort_values(["Score", "Asset", "TF"], ascending=[False, True, True])
            elif sort_choice == "Asset A→Z":
                tbl = tbl.sort_values(["Asset", "TF"])
            else:
                tbl = tbl.sort_values(["Asset_sort", "TF_sort"])

            view_cols = ["Asset", "Consensus", "TF", "Price", "Score", "Bias", "Weight", "Funding %", "Alignment", "Last (Berlin)"]
            summary = (tbl[view_cols], {
                "Price": st.column_config.NumberColumn(format="%.2f"),
                "Score": st.column_config.NumberColumn(format="%.1f"),
                "Weight": st.column_config.ProgressColumn("Weight", min_value=0.0, max_value=1.0, format="%.2f"),
                "Funding %": st.column_config.NumberColumn(format="%.3f%%"),
                "Last (Berlin)": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
            })

        # 4) Global insight
        global_insight = ""
        if not table.empty:
            avg_weight = table["Weight"].mean()
            bias_counts = table["Bias"].value_counts()
            funding_vals = table["Funding %"].replace("-", np.nan).dropna().astype(float)
            avg_funding = funding_vals.mean() if not funding_vals.empty else np.nan
            main_bias = bias_counts.idxmax() if not bias_counts.empty else "⚪ NEUTRAL"

            global_insight = f"**🧠 Signal Insight (Overall)**\n\n"
            global_insight += f"Dominant Bias: **{main_bias}**\n"
            global_insight += f"Average Weight: **{avg_weight:.2f}**\n"
            if not np.isnan(avg_funding):
                global_insight += f"Average Funding: **{avg_funding:.3f}%**\n\n"
            if "SHORT" in main_bias and avg_weight > 0.7:
                global_insight += "→ Strong bearish regime across timeframes. Trend mature; avoid chasing new shorts."
            elif "LONG" in main_bias and avg_weight > 0.7:
                global_insight += "→ Strong bullish momentum. Stay with trend; manage trailing stops."
            elif avg_weight < 0.3:
                global_insight += "→ Weak or choppy momentum. Stand aside until structure builds."
            else:
                global_insight += "→ Mixed signals. Wait for confirmation before acting."

        # 5) Per-asset insights
        per_asset_md = []
        if show_asset_insights and not table.empty:
            order_assets = [a for a in ["BTC/USDT","ETH/USDT","BNB/USDT","SOL/USDT","XRP/USDT"] if a in assets]
            for asset in order_assets:
                sub = table[table["Asset"] == asset].copy()
                if sub.empty: continue
                counts = sub["Bias"].value_counts()
                dom = counts.idxmax()
                agree = int(counts.max()); total = len(sub)
                a_weight = sub["Weight"].mean()
                fvals = sub["Funding %"].replace("-", np.nan).dropna().astype(float)
                a_funding = fvals.mean() if not fvals.empty else np.nan
                lead_row = sub.sort_values("Weight", ascending=False).iloc[0]
                lead_tf, lead_w = lead_row["TF"], lead_row["Weight"]
                if "SHORT" in dom and a_weight >= 0.7:
                    hint = "Trend strong & mature — avoid chasing; consider rallies to re-short."
                elif "LONG" in dom and a_weight >= 0.7:
                    hint = "Strong bullish impulse — stay with trend; use trailing stops."
                elif a_weight < 0.3:
                    hint = "Low conviction — stand aside, wait for structure."
                else:
                    hint = "Mixed conviction — prefer confirmation before entries."
                funding_txt = "–" if np.isnan(a_funding) else f"{a_funding:.3f}%"
                per_asset_md.append(
                    f"**{asset}** — {agree}/{total} TFs **{dom}**, avg Weight **{a_weight:.2f}**, "
                    f"avg Funding **{funding_txt}**. Lead TF: **{lead_tf}** (Weight {lead_w:.2f}).\n\n→ {hint}"
                )

        # 6) Heat grid (rebuild only if biases changed)
        fig_heat = None
        if show_heatgrid and not table.empty:
            bias_sig = "|".join(table.sort_values(["Asset","TF"])["Bias"].astype(str).tolist())
            if bias_sig != st.session_state.last_bias_sig or st.session_state.heat_fig is None:
                xs, ys, colors, hovers = [], [], [], []
                cmap = {"🟢 LONG": "#16a34a", "🔴 SHORT": "#b91c1c", "⚪ NEUTRAL": "#6b7280"}
                for i, a in enumerate(assets):
                    for j, tf in enumerate(tfs):
                        b = table[(table["Asset"] == a) & (table["TF"] == tf)]
                        bias = b.iloc[0]["Bias"] if len(b) == 1 else "⚪ NEUTRAL"
                        xs.append(j); ys.append(i)
                        colors.append(cmap.get(bias, "#6b7280"))
                        hovers.append(f"{a} | {tf} | {bias}")
                marker_size = 18
                row_height = 24
                base_height = 70
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode="markers",
                    marker=dict(size=marker_size, color=colors, line=dict(color="#111", width=1), symbol="circle"),
                    text=hovers, hovertemplate="%{text}<extra></extra>"
                ))
                fig.update_xaxes(tickvals=list(range(len(tfs))), ticktext=tfs, side="top", color="white", showgrid=False, zeroline=False)
                fig.update_yaxes(tickvals=list(range(len(assets))), ticktext=assets, autorange="reversed", color="white", showgrid=False, zeroline=False)
                fig.update_layout(height=base_height + row_height * max(1, len(assets)),
                                  margin=dict(l=4, r=4, t=0, b=0),
                                  plot_bgcolor="#000000", paper_bgcolor="#000000")
                st.session_state.heat_fig = fig
                st.session_state.last_bias_sig = bias_sig
            fig_heat = st.session_state.heat_fig

        # 7) Live sparkline figs
        card_figs = []
        if show_sparklines and not table.empty:
            for _, r in table.iterrows():
                if len(r["Spark"]) > 1:
                    fig2 = go.Figure()
                    fig2.add_trace(go.Spline(y=r["Spark"])) if hasattr(go, "Spline") else \
                        fig2.add_trace(go.Scatter(y=r["Spark"], mode="lines", line=dict(color=r["TrendColor"], width=2), hoverinfo="skip"))
                    fig2.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=80,
                                       paper_bgcolor="#000", plot_bgcolor="#000",
                                       xaxis_visible=False, yaxis_visible=False)
                    card_figs.append((r, fig2))

        # 8) Footer
        berlin = pytz.timezone("Europe/Berlin")
        ts = datetime.utcnow().replace(tzinfo=pytz.utc).astimezone(berlin).strftime("%Y-%m-%d %H:%M:%S")
        footer_text = f"Last update (Berlin): {ts} • Build: {BUILD}"

        # 9) Snapshot
        snapshot = {
            "table": table,
            "kpis": kpis,
            "summary": summary,
            "global_insight": global_insight,
            "per_asset_md": per_asset_md,
            "fig_heat": fig_heat,
            "card_figs": card_figs,
            "footer": footer_text,
            "errors": errors
        }

    except Exception as e:
        if st.session_state.last_snapshot is None:
            st.error(f"Update failed: {e}")
        snapshot = st.session_state.last_snapshot

# -------------------- RENDER (swap once) --------------------
if snapshot is not None:
    if snapshot["kpis"] is not None:
        with kpi_ph.container():
            c1, c2, c3 = st.columns(3)
            c1.metric("🟢 LONG", snapshot["kpis"]["long"])
            c2.metric("🔴 SHORT", snapshot["kpis"]["short"])
            c3.metric("⚪ NEUTRAL", snapshot["kpis"]["neutral"])

    if show_summary and snapshot["summary"] is not None:
        df_view, column_cfg = snapshot["summary"]
        with summary_ph.container():
            st.subheader("Summary")
            st.dataframe(df_view, use_container_width=True, hide_index=True, column_config=column_cfg)
            st.caption("Price source: Spot (10 s cache). Scores & bias by timeframe OHLCV.")

    if snapshot["global_insight"]:
        with insight_ph.container():
            st.markdown(snapshot["global_insight"])

    if show_asset_insights and snapshot["per_asset_md"]:
        with asset_insights_ph.container():
            st.subheader("Per-Asset Insights")
            for md in snapshot["per_asset_md"]:
                st.markdown(md)

    if show_heatgrid and snapshot["fig_heat"] is not None:
        with heat_ph.container():
            st.subheader("Bias Heat Grid")
            st.plotly_chart(snapshot["fig_heat"], use_container_width=True)
            st.markdown("**Legend:** 🟢 Long • ⚪ Neutral • 🔴 Short")

    if show_sparklines and snapshot["card_figs"]:
        with cards_ph.container():
            st.subheader("Live Signals Overview")
            for r, fig2 in snapshot["card_figs"]:
                c_info, c_chart = st.columns([1, 4])
                c_info.markdown(f"**{r['Asset']} — {r['TF']}**")
                c_info.markdown(
                    f"Bias: {r['Bias']}  |  Weight: `{r['Weight']}`  |  Alignment: {r['Alignment']}`  |  Price: `{r['Price']}`"
                )
                c_info.caption(f"Funding: {r['Funding %']}% • Last (Berlin): {r['Last (Berlin)']}")
                c_chart.plotly_chart(fig2, use_container_width=True)

    with footer_ph.container():
        st.caption(snapshot["footer"])

    if snapshot["errors"]:
        with st.expander("🛠️ Diagnostics / Errors"):
            st.dataframe(pd.DataFrame(snapshot["errors"], columns=["Asset", "TF", "Error"]), hide_index=True)

    st.session_state.last_snapshot = snapshot
else:
    st.error("No snapshot available.")

# -------------------- INTERPRETATION --------------------
with st.expander("ℹ️ Interpretation & Rules"):
    st.markdown(f"""
**Bias (trend direction)**
- 🟢 **LONG**: Score > **{entry_long}** → upward momentum
- ⚪ **NEUTRAL**: {entry_short} ≤ Score ≤ **{entry_long}**
- 🔴 **SHORT**: Score < **{entry_short}** → downward momentum

**Weight (trend strength 0–1)**
- **≈ 1.00** strong trend • **≈ 0.50** moderate • **≈ 0.25** early/weak

**Funding alignment**
- **✅ aligned**: LONG with funding ≤ 0% • SHORT with funding ≥ 0%
- **⚠️ divergence**: momentum vs. funding disagree → reduce size or wait

**Multi-timeframe reading**
- Agreement across TFs & Weight > 0.7 → robust trend
- 5m flips first → early warning / potential reversal
- 1h & 4h dominate → higher-timeframe context

**Signal triggers (UI alerts)**
- **🟢 Long setup:** Score crosses up above **{entry_long}**
- **🔴 Short setup:** Score crosses down below **{entry_short}**
- **⚠️ Exit warning:** Score < **55**
- **🚪 Hard exit:** Score < **45**
""")
