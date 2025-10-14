import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime
import pytz, sys, os

BUILD = "HG-markers-v8"

# ----------------- PAGE SETUP -----------------
st.set_page_config(page_title="Momentum Signals", layout="wide")
st.title("📈 Momentum Signals & Market Bias")

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.header("Settings")
    st.caption(f"Running file: `{__file__}`")

    default_assets = ["BTC/USDT", "ETH/USDT"]
    assets = st.multiselect("Assets", default_assets, default=default_assets)
    tfs = st.multiselect("Timeframes", ["5m", "1h", "4h"], default=["5m", "1h", "4h"])

    entry_long = st.slider("Long threshold", 50, 80, 60, 1)
    entry_short = st.slider("Short threshold", 20, 60, 40, 1)
    spark_len = st.slider("Sparkline length", 50, 200, 100, 10)

    show_summary = st.toggle("Show Summary Table", value=True)
    show_heatgrid = st.toggle("Show Heat Grid", value=True)
    show_sparklines = st.toggle("Show Live Cards (with Sparklines)", value=True)
    auto_refresh = st.toggle("Auto-refresh every 60s", value=True)

    st.markdown("—")
    st.subheader("Filter")
    bias_filter = st.multiselect(
        "Show biases",
        ["🟢 LONG", "⚪ NEUTRAL", "🔴 SHORT"],
        default=["🟢 LONG", "⚪ NEUTRAL", "🔴 SHORT"]
    )
    sort_choice = st.selectbox(
        "Sort summary table by",
        ["Default (BTC→ETH + TF order)", "Weight (desc)", "Score (desc)", "Asset A→Z"],
        index=0
    )

    st.divider()
    st.subheader("🧹 Maintenance")
    if st.button("Clear cache (once)"):
        st.cache_data.clear()
        st.experimental_rerun()

    with st.expander("Debug info"):
        st.write({"python": sys.version, "cwd": os.getcwd()})
    st.caption("Tip: press 'R' to rerun manually.")

if auto_refresh:
    st.markdown("<meta http-equiv='refresh' content='60'>", unsafe_allow_html=True)

# ----------------- DATA HELPERS -----------------
BINANCE_SPOT = [
    "https://api.binance.com", "https://api1.binance.com",
    "https://api2.binance.com", "https://api3.binance.com",
    "https://data-api.binance.vision"
]
BINANCE_FUT = [
    "https://fapi.binance.com", "https://fapi1.binance.com",
    "https://fapi2.binance.com", "https://fapi3.binance.com"
]

def sym_to_perp(sym): return sym.replace("/", "")
def tf_to_kraken(tf): return {"5m": 5, "1h": 60, "4h": 240}.get(tf, 60)

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
    pair_map = {"BTC/USDT": "XBTUSDT", "ETH/USDT": "ETHUSDT"}
    r = requests.get("https://api.kraken.com/0/public/OHLC",
                     params={"pair": pair_map.get(sym), "interval": tf_to_kraken(tf)}, timeout=15)
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
            r = requests.get(f"{base}/fapi/v1/fundingRate",
                             params={"symbol": perp, "limit": 1}, timeout=15)
            r.raise_for_status()
            j = r.json()
            if isinstance(j, list) and j:
                return float(j[-1]["fundingRate"])
        except:
            continue
    return np.nan

# --------- NEW: unified spot price (same across TFs), short cache (10s) ---------
@st.cache_data(ttl=10)
def fetch_last_price_binance(perp: str) -> float:
    r = requests.get("https://api.binance.com/api/v3/ticker/price",
                     params={"symbol": perp}, timeout=10)
    r.raise_for_status()
    return float(r.json()["price"])

@st.cache_data(ttl=10)
def fetch_last_price_kraken(sym: str) -> float:
    pair_map = {"BTC/USDT": "XBTUSDT", "ETH/USDT": "ETHUSDT"}
    kp = pair_map.get(sym)
    r = requests.get("https://api.kraken.com/0/public/Ticker",
                     params={"pair": kp}, timeout=10)
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

# ----------------- CALCULATIONS -----------------
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
    vol_regime = ((a/df["close"]) >
                  (a/df["close"]).rolling(90).median()).fillna(False)
    raw = 0.3*trend + 0.3*breakout + 0.2*vol_conf + 0.2*vol_regime
    return (100*raw.clip(0, 1)).fillna(0)

def bias_and_weight(score, long_t, short_t):
    if score >= long_t:
        w = min(1.0, (score - long_t) / (100 - long_t + 1e-9))
        return "LONG", round(w, 2)
    if score <= short_t:
        w = min(1.0, (short_t - score) / (short_t + 1e-9))
        return "SHORT", round(w, 2)
    return "NEUTRAL", 0.0

def bias_badge(b): return {"LONG": "🟢 LONG", "SHORT": "🔴 SHORT", "NEUTRAL": "⚪ NEUTRAL"}[b]
def funding_alignment(bias, fr):
    if pd.isna(fr): return "–"
    if bias == "LONG" and fr <= 0: return "✅ aligned"
    if bias == "SHORT" and fr >= 0: return "✅ aligned"
    if bias == "NEUTRAL": return "•"
    return "⚠️"

# ----------------- BUILD TABLE -----------------
rows, errors = [], []
for sym in assets:
    perp = sym_to_perp(sym)
    live_price = fetch_spot_last(sym)  # unified live price per asset (10s cache)
    for tf in tfs:
        try:
            df = fetch_ohlcv(sym, tf, 1000)
            sc = compute_score(df)
            s = float(sc.iloc[-1])
            price_fallback = float(df["close"].iloc[-1])
            price = live_price if not np.isnan(live_price) else price_fallback
            last_time = sc.index[-1]
            fr = fetch_funding(perp)
            funding_pct = "-" if pd.isna(fr) else round(fr * 100, 3)
            bias, w = bias_and_weight(s, entry_long, entry_short)
            align = funding_alignment(bias, fr)
            spark_vals = df["close"].tail(spark_len).tolist()
            delta = spark_vals[-1] - spark_vals[0] if len(spark_vals) > 1 else 0
            spark_color = "green" if delta > 0 else "red" if delta < 0 else "gray"
            rows.append({
                "Asset": sym, "TF": tf, "Price": round(price, 2),
                "Funding %": funding_pct, "Score": round(s, 1),
                "Bias": bias_badge(bias), "Weight": w, "Alignment": align,
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

# ----------------- KPI HEADER -----------------
if not table.empty:
    c1, c2, c3 = st.columns(3)
    c1.metric("🟢 LONG", (table["Bias"] == "🟢 LONG").sum())
    c2.metric("🔴 SHORT", (table["Bias"] == "🔴 SHORT").sum())
    c3.metric("⚪ NEUTRAL", (table["Bias"] == "⚪ NEUTRAL").sum())

# ----------------- SUMMARY TABLE -----------------
if show_summary and not table.empty:
    st.subheader("Summary")
    tbl = table[table["Bias"].isin(bias_filter)].copy()

    # Default sort: BTC→ETH and TF 5m→1h→4h
    tf_order = {"5m": 0, "1h": 1, "4h": 2}
    asset_order = {"BTC/USDT": 0, "ETH/USDT": 1}
    tbl["TF_sort"] = tbl["TF"].map(tf_order).fillna(999)
    tbl["Asset_sort"] = tbl["Asset"].map(asset_order).fillna(999)

    if sort_choice == "Weight (desc)":
        tbl = tbl.sort_values(["Weight", "Asset", "TF"], ascending=[False, True, True])
    elif sort_choice == "Score (desc)":
        tbl = tbl.sort_values(["Score", "Asset", "TF"], ascending=[False, True, True])
    elif sort_choice == "Asset A→Z":
        tbl = tbl.sort_values(["Asset", "TF"])
    else:
        tbl = tbl.sort_values(["Asset_sort", "TF_sort"])

    tbl = tbl.drop(columns=["Asset_sort", "TF_sort"], errors="ignore")

    view_cols = ["Asset", "TF", "Price", "Score", "Bias", "Weight", "Funding %", "Alignment", "Last (Berlin)"]
    tbl_view = tbl[view_cols]

    st.dataframe(
        tbl_view,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Price": st.column_config.NumberColumn(format="%.2f"),
            "Score": st.column_config.NumberColumn(format="%.1f"),
            "Weight": st.column_config.ProgressColumn("Weight", min_value=0.0, max_value=1.0, format="%.2f"),
            "Funding %": st.column_config.NumberColumn(format="%.3f%%"),
            "Last (Berlin)": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
        },
    )
    st.caption("Price source: Spot (10 s cache). Scores & bias by timeframe OHLCV.")

    # ----------------- SIGNAL INSIGHT BOX -----------------
    avg_weight = tbl["Weight"].mean()
    bias_counts = tbl["Bias"].value_counts()
    funding_vals = tbl["Funding %"].replace("-", np.nan).dropna().astype(float)
    avg_funding = funding_vals.mean() if not funding_vals.empty else np.nan
    main_bias = bias_counts.idxmax() if not bias_counts.empty else "⚪ NEUTRAL"

    insight = f"**🧠 Signal Insight**\n\n"
    insight += f"Dominant Bias: **{main_bias}**\n"
    insight += f"Average Weight: **{avg_weight:.2f}**\n"
    if not np.isnan(avg_funding):
        insight += f"Average Funding: **{avg_funding:.3f}%**\n\n"

    if "SHORT" in main_bias and avg_weight > 0.7:
        insight += "→ Strong bearish regime across timeframes. Trend mature; avoid chasing new shorts."
    elif "LONG" in main_bias and avg_weight > 0.7:
        insight += "→ Strong bullish momentum. Stay with trend, manage trailing stops."
    elif avg_weight < 0.3:
        insight += "→ Weak or choppy momentum. Stand aside until structure builds."
    else:
        insight += "→ Mixed signals. Wait for confirmation before acting."

    st.markdown(insight)

# ----------------- HEAT GRID (COMPACT) -----------------
if show_heatgrid and not table.empty:
    st.subheader("Bias Heat Grid")
    assets_order, tfs_order = assets[:], tfs[:]
    xs, ys, colors, hovers = [], [], [], []
    cmap = {"🟢 LONG": "#16a34a", "🔴 SHORT": "#b91c1c", "⚪ NEUTRAL": "#6b7280"}
    for i, a in enumerate(assets_order):
        for j, tf in enumerate(tfs_order):
            b = table[(table["Asset"] == a) & (table["TF"] == tf)]
            bias = b.iloc[0]["Bias"] if len(b) == 1 else "⚪ NEUTRAL"
            xs.append(j); ys.append(i)
            colors.append(cmap.get(bias, "#6b7280"))
            hovers.append(f"{a} | {tf} | {bias}")

    # Smaller markers + tighter layout
    marker_size = 22  # was 42
    row_height = 30   # was 48
    base_height = 80  # was 140

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(size=marker_size, color=colors,
                    line=dict(color="#111111", width=1), symbol="circle"),
        text=hovers, hovertemplate="%{text}<extra></extra>"
    ))
    fig.update_xaxes(
        tickvals=list(range(len(tfs_order))), ticktext=tfs_order,
        side="top", color="white", showgrid=False, zeroline=False
    )
    fig.update_yaxes(
        tickvals=list(range(len(assets_order))), ticktext=assets_order,
        autorange="reversed", color="white", showgrid=False, zeroline=False
    )
    fig.update_layout(
        height=base_height + row_height * max(1, len(assets_order)),
        margin=dict(l=4, r=4, t=0, b=0),
        plot_bgcolor="#000000", paper_bgcolor="#000000"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Legend:** 🟢 Long • ⚪ Neutral • 🔴 Short")

# ----------------- LIVE CARDS (SPARKLINES) -----------------
if show_sparklines and not table.empty:
    st.subheader("Live Signals Overview")
    for _, r in table.iterrows():
        c_info, c_chart = st.columns([1, 4])
        c_info.markdown(f"**{r['Asset']} — {r['TF']}**")
        c_info.markdown(
            f"Bias: {r['Bias']}  |  Weight: `{r['Weight']}`  |  Alignment: {r['Alignment']}  |  Price: `{r['Price']}`"
        )
        c_info.caption(f"Funding: {r['Funding %']}% • Last (Berlin): {r['Last (Berlin)']}")
        if len(r["Spark"]) > 1:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                y=r["Spark"], mode="lines",
                line=dict(color=r["TrendColor"], width=2),
                hoverinfo="skip"
            ))
            fig2.update_layout(
                margin=dict(l=0, r=0, t=0, b=0), height=80,
                paper_bgcolor="#000", plot_bgcolor="#000",
                xaxis_visible=False, yaxis_visible=False
            )
            c_chart.plotly_chart(fig2, use_container_width=True)

# ----------------- INTERPRETATION & RULES -----------------
with st.expander("ℹ️ Interpretation & Rules"):
    st.markdown(f"""
**Bias (trend direction)**
- 🟢 **LONG**: Score > **{entry_long}** → upward momentum
- ⚪ **NEUTRAL**: {entry_short} ≤ Score ≤ **{entry_long}**
- 🔴 **SHORT**: Score < **{entry_short}** → downward momentum

**Weight (trend strength 0–1)**
- **≈ 1.00** → strong trend
- **≈ 0.50** → moderate / pausing
- **≈ 0.25** → weak / early momentum (caution)

**Funding alignment**
- **✅ aligned**: LONG with funding ≤ 0% or SHORT with funding ≥ 0% (confirmation)
- **⚠️ divergence**: momentum vs. funding disagree → reduce size or wait for confirmation

**Multi-timeframe reading**
- All selected TFs agree & Weight > 0.7 → robust trend
- 5m flips first → early warning / potential reversal
- 1h & 4h dominate → higher-timeframe context

**Signal triggers (alerts)**
- **🟢 Long setup:** Score **crosses up** above **{entry_long}**
- **🔴 Short setup:** Score **crosses down** below **{entry_short}**
- **⚠️ Exit warning:** Score drops **below 55**
- **🚪 Hard exit:** Score drops **below 45**
""")

# ----------------- FOOTER -----------------
berlin = pytz.timezone("Europe/Berlin")
ts = datetime.utcnow().replace(tzinfo=pytz.utc).astimezone(berlin).strftime("%Y-%m-%d %H:%M:%S")
st.caption(f"Last update (Berlin): {ts} • Build: {BUILD}")

if errors:
    with st.expander("🛠️ Diagnostics / Errors"):
        st.dataframe(pd.DataFrame(errors, columns=["Asset", "TF", "Error"]), hide_index=True)
