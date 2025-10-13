import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import pytz
import plotly.graph_objects as go

st.set_page_config(page_title="Momentum Signals & Market Bias", layout="wide")
st.title("📈 Momentum Signals & Market Bias")

# ================= Sidebar / UI =================
with st.sidebar:
    st.header("Settings")
    default_assets = ["BTC/USDT", "ETH/USDT"]
    assets = st.multiselect("Assets", default_assets, default=default_assets)
    tfs = st.multiselect("Timeframes", ["5m", "1h", "4h"], default=["5m", "1h", "4h"])
    entry_long = st.slider("Long threshold", 50, 80, 60, 1)
    entry_short = st.slider("Short threshold", 20, 60, 40, 1)
    spark_len = st.slider("Sparkline length (last candles)", 50, 200, 100, 10)
    show_sparklines = st.toggle("Show sparklines", value=True)
    show_heatgrid = st.toggle("Show bias heat grid", value=True)
    auto_refresh = st.toggle("Auto-refresh every 60s", value=True)
    st.caption("Tip: Use the Rerun button for an immediate refresh (data cache ≈60s).")

# Lightweight auto-refresh
if auto_refresh:
    st.markdown("<meta http-equiv='refresh' content='60'>", unsafe_allow_html=True)

# =============== Helpers / Fallbacks ===============
BINANCE_SPOT_BASES = [
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://data-api.binance.vision",
]
BINANCE_FUT_BASES = [
    "https://fapi.binance.com",
    "https://fapi1.binance.com",
    "https://fapi2.binance.com",
    "https://fapi3.binance.com",
]

def sym_to_perp(sym: str) -> str:
    return sym.replace("/", "")

def tf_to_kraken(tf: str) -> int:
    return {"5m": 5, "1h": 60, "4h": 240}.get(tf, 60)

@st.cache_data(ttl=60)
def fetch_klines_binance_any(perp: str, interval: str, limit: int = 1000) -> pd.DataFrame:
    params = {"symbol": perp, "interval": interval, "limit": limit}
    last_exc = None
    for base in BINANCE_SPOT_BASES:
        try:
            r = requests.get(f"{base}/api/v3/klines", params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
            cols = ["open_time","open","high","low","close","volume",
                    "close_time","qav","num_trades","taker_base","taker_quote","ignore"]
            df = pd.DataFrame(data, columns=cols)
            df["time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
            out = df.set_index("time")[["open","high","low","close","volume"]].astype(float)
            return out
        except Exception as e:
            last_exc = e
            continue
    raise last_exc if last_exc else RuntimeError("Binance klines failed")

@st.cache_data(ttl=60)
def fetch_klines_kraken(sym: str, tf: str, limit: int = 1000) -> pd.DataFrame:
    pair_map = {"BTC/USDT": "XBTUSDT", "ETH/USDT": "ETHUSDT"}
    kr_pair = pair_map.get(sym)
    if not kr_pair:
        raise ValueError("Unsupported symbol for Kraken fallback")
    interval = tf_to_kraken(tf)
    r = requests.get("https://api.kraken.com/0/public/OHLC",
                     params={"pair": kr_pair, "interval": interval}, timeout=15)
    r.raise_for_status()
    data = r.json()
    k = next(iter(data["result"]))
    rows = data["result"][k]
    df = pd.DataFrame(rows, columns=["time","open","high","low","close","vwap","volume","count"])
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    out = df.set_index("time")[["open","high","low","close","volume"]].astype(float)
    return out.tail(limit)

@st.cache_data(ttl=60)
def fetch_ohlcv(symbol: str, tf: str, limit: int = 1000) -> pd.DataFrame:
    perp = sym_to_perp(symbol)
    try:
        return fetch_klines_binance_any(perp, tf, limit)
    except Exception:
        return fetch_klines_kraken(symbol, tf, limit)

@st.cache_data(ttl=60)
def fetch_funding(perp: str) -> float:
    for base in BINANCE_FUT_BASES:
        try:
            r = requests.get(f"{base}/fapi/v1/fundingRate",
                             params={"symbol": perp, "limit": 1}, timeout=15)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list) and data:
                return float(data[-1]["fundingRate"])
        except Exception:
            continue
    return np.nan

# =============== Indicators / Score =================
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def donchian(df, n=20): return df["high"].rolling(n).max(), df["low"].rolling(n).min()
def atr(df, n=20):
    pc = df["close"].shift(1)
    tr = pd.concat([
        (df["high"]-df["low"]).abs(),
        (df["high"]-pc).abs(),
        (df["low"]-pc).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()
def zscore(s, n=90):
    m, v = s.rolling(n).mean(), s.rolling(n).std(ddof=0)
    return (s - m) / v.replace(0, np.nan)

def compute_score(df):
    ema50, ema200 = ema(df["close"], 50), ema(df["close"], 200)
    trend_up = (df["close"] > ema50) & (ema50 > ema200)
    dc_hi, dc_lo = donchian(df, 20)
    breakout_up = df["close"] > dc_hi.shift(1)
    volZ = zscore(df["volume"], 90).fillna(0)
    vol_confirm = volZ > 0.5
    a = atr(df, 20)
    vol_regime = ((a/df["close"]) > (a/df["close"]).rolling(90).median()).fillna(False)
    raw_long = (0.3*trend_up + 0.3*breakout_up + 0.2*vol_confirm + 0.2*vol_regime)
    return (100*raw_long.clip(0, 1)).fillna(0)

def bias_and_weight(score, long_th, short_th):
    if score >= long_th:
        w = min(1.0, (score - long_th) / (100 - long_th + 1e-9))
        return "LONG", round(w, 2)
    if score <= short_th:
        w = min(1.0, (short_th - score) / (short_th + 1e-9))
        return "SHORT", round(w, 2)
    return "NEUTRAL", 0.0

def bias_badge(bias):
    return {"LONG": "🟢 LONG", "SHORT": "🔴 SHORT", "NEUTRAL": "⚪ NEUTRAL"}[bias]

def funding_alignment(bias, fr):
    if pd.isna(fr): return "–"
    if bias == "LONG" and fr <= 0: return "✅ aligned"
    if bias == "SHORT" and fr >= 0: return "✅ aligned"
    if bias == "NEUTRAL": return "•"
    return "⚠️"

# ================= Build rows =================
rows, errors = [], []
for sym in assets:
    perp = sym_to_perp(sym)
    for tf in tfs:
        try:
            df = fetch_ohlcv(sym, tf, limit=1000)
            sc = compute_score(df)
            last_sc = float(sc.iloc[-1])
            price = float(df["close"].iloc[-1])
            last_time = sc.index[-1]
            fr = fetch_funding(perp)
            funding_pct = "-" if pd.isna(fr) else round(fr * 100, 3)
            bias, w = bias_and_weight(last_sc, entry_long, entry_short)
            align = funding_alignment(bias, fr)

            spark_vals = df["close"].tail(spark_len).astype(float).tolist()
            if len(spark_vals) > 1:
                delta = spark_vals[-1] - spark_vals[0]
                spark_color = "green" if delta > 0 else "red" if delta < 0 else "gray"
            else:
                spark_color = "gray"

            rows.append({
                "Asset": sym,
                "TF": tf,
                "Price": round(price, 2),
                "Funding %": funding_pct,
                "Score": round(last_sc, 1),
                "Bias": bias_badge(bias),
                "Weight": w,
                "Alignment": align,
                "TrendColor": spark_color,
                "Spark": spark_vals if show_sparklines else [],
                "Last (Berlin)": last_time.tz_convert("Europe/Berlin").strftime("%Y-%m-%d %H:%M")
            })
        except Exception as e:
            errors.append((sym, tf, str(e)))
            rows.append({
                "Asset": sym, "TF": tf, "Price": "-", "Funding %": "-", "Score": "-",
                "Bias": "ERR", "Weight": 0.0, "Alignment": "–",
                "TrendColor": "gray", "Spark": [],
                "Last (Berlin)": str(e)
            })

table = pd.DataFrame(rows)

# ================= Bias Heat Grid (markers-only, no text, dark) — HG-markers-v1 =================
if show_heatgrid and not table.empty:
    st.subheader("Bias Heat Grid")

    # Order
    assets_order = assets[:]
    tfs_order = tfs[:]

    # Build points
    xs, ys, colors, hovers = [], [], [], []
    color_map = {"🟢 LONG": "#16a34a", "🔴 SHORT": "#b91c1c", "⚪ NEUTRAL": "#374151"}

    for i, a in enumerate(assets_order):
        for j, tf in enumerate(tfs_order):
            match = table[(table["Asset"] == a) & (table["TF"] == tf)]
            bias = match.iloc[0]["Bias"] if len(match) == 1 else "⚪ NEUTRAL"
            xs.append(j)
            ys.append(i)
            colors.append(color_map.get(bias, "#374151"))
            hovers.append(f"Asset: {a}<br>TF: {tf}<br>Bias: {bias}")

    fig_grid = go.Figure()
    fig_grid.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="markers",                # <-- markers only (NO text)
        marker=dict(
            size=42,
            color=colors,
            line=dict(color="#111111", width=2),
            symbol="circle"
        ),
        hovertemplate="%{text}<extra></extra>",
        text=hovers
    ))

    fig_grid.update_xaxes(
        tickvals=list(range(len(tfs_order))), ticktext=tfs_order,
        side="top", color="white", showgrid=False, zeroline=False
    )
    fig_grid.update_yaxes(
        tickvals=list(range(len(assets_order))), ticktext=assets_order,
        autorange="reversed", color="white", showgrid=False, zeroline=False
    )
    fig_grid.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=140 + 48 * max(1, len(assets_order)),
        plot_bgcolor="#000000", paper_bgcolor="#000000"
    )

    st.plotly_chart(fig_grid, use_container_width=True)

# ================= Live Cards with Sparklines =================
if not table.empty:
    st.subheader("Live Signals Overview")
    for _, row in table.iterrows():
        c_info, c_chart = st.columns([1, 4])
        c_info.markdown(f"**{row['Asset']} — {row['TF']}**")
        c_info.markdown(
            f"Bias: {row['Bias']}  |  Weight: `{row['Weight']}`  |  Alignment: {row['Alignment']}  |  Price: `{row['Price']}`"
        )
        c_info.caption(f"Last (Berlin): {row['Last (Berlin)']}  •  Funding: {row['Funding %']}%")
        if show_sparklines and len(row["Spark"]) > 1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=row["Spark"], mode="lines",
                line=dict(color=row["TrendColor"], width=2),
                hoverinfo="skip"
            ))
            mid = (max(row["Spark"]) + min(row["Spark"])) / 2.0
            fig.add_hline(y=mid, line_width=1, line_dash="dot", line_color="lightgray")
            fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0), height=80,
                xaxis_visible=False, yaxis_visible=False,
                paper_bgcolor="#000000", plot_bgcolor="#000000"
            )
            c_chart.plotly_chart(fig, use_container_width=True)

# ================= Export / Legend / Diagnostics =================
berlin = pytz.timezone("Europe/Berlin")
ts = datetime.utcnow().replace(tzinfo=pytz.utc).astimezone(berlin).strftime("%Y-%m-%d %H:%M:%S")
st.caption(f"Last update (Berlin): {ts} • Build: HG-emoji-v3")

if not table.empty:
    csv_bytes = table.drop(columns=["TrendColor"]).to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download table as CSV", data=csv_bytes,
                       file_name="momentum_dashboard.csv", mime="text/csv")

with st.expander("ℹ️ Interpretation & Rules"):
    st.markdown(f"""
**Bias (trend direction)**  
🟢 LONG = upward momentum  
⚪ NEUTRAL = range / sideways  
🔴 SHORT = downward momentum  

**Weight (trend strength 0–1)**  
1.00 strong • 0.50 moderate • 0.25 weak  

**Funding alignment**  
✅ aligned = funding confirms bias  
⚠️ = divergence, be cautious  

**Signal triggers (Telegram)**  
🟢 Long setup: score crosses above {entry_long}  
🔴 Short setup: score crosses below {entry_short}  
⚠️ Exit warning: score < 55  
🚪 Hard exit: score < 45
""")

if errors:
    with st.expander("🛠️ Diagnostics / Errors"):
        err_df = pd.DataFrame(errors, columns=["Asset", "TF", "Error"])
        st.dataframe(err_df, use_container_width=True, hide_index=True)

st.caption("Data: Binance (multiple domains; fallback to Kraken if restricted). Funding from Binance Futures. Not financial advice.")
