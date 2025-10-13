import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import pytz

st.set_page_config(page_title="Momentum Dashboard", layout="wide")
st.title("Momentum • Live Bias & Funding")

# ================= Sidebar / UI =================
with st.sidebar:
    st.header("Settings")
    default_assets = ["BTC/USDT", "ETH/USDT"]
    assets = st.multiselect("Assets", default_assets, default=default_assets)
    tfs = st.multiselect("Timeframes", ["5m", "1h", "4h"], default=["5m", "1h", "4h"])
    entry_long = st.slider("Long threshold", 50, 80, 60, 1)
    entry_short = st.slider("Short threshold", 20, 60, 40, 1)
    spark_len = st.slider("Sparkline length", 50, 200, 100, 10)
    show_sparklines = st.toggle("Show sparklines", value=True)
    auto_refresh = st.toggle("Auto-refresh every 60s", value=True)
    st.caption("Tip: Use the Rerun button in the toolbar for an immediate refresh (data cache ≈60s).")

# Lightweight auto-refresh (browser meta refresh)
if auto_refresh:
    st.markdown("<meta http-equiv='refresh' content='60'>", unsafe_allow_html=True)

# =============== Helpers / Fallbacks ===============
BINANCE_SPOT_BASES = [
    "https://api.binance.com",
    "https://api-gcp.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://data-api.binance.vision",
]
BINANCE_FUT_BASES = [
    "https://fapi.binance.com",
    "https://fapi-gcp.binance.com",
    "https://fapi1.binance.com",
    "https://fapi2.binance.com",
    "https://fapi3.binance.com",
]

def sym_to_perp(sym: str) -> str:
    # "BTC/USDT" -> "BTCUSDT"
    return sym.replace("/", "")

def tf_to_kraken(tf: str) -> int:
    # Kraken OHLC intervals in minutes
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
    # Map only supported pairs for fallback
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
    # Try Binance → fallback to Kraken if blocked
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
                return float(data[-1]["fundingRate"])  # 0.0001 = 0.01%
        except Exception:
            continue
    return np.nan  # if all endpoints are blocked: hide it

# =============== Indicators / Score =================
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def donchian(df, n=20):
    hi = df["high"].rolling(n, min_periods=n).max()
    lo = df["low"].rolling(n, min_periods=n).min()
    return hi, lo

def atr(df, n=20):
    pc = df["close"].shift(1)
    tr = pd.concat([
        (df["high"]-df["low"]).abs(),
        (df["high"]-pc).abs(),
        (df["low"]-pc).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()

def zscore(s, n=90):
    m = s.rolling(n).mean()
    v = s.rolling(n).std(ddof=0)
    return (s - m) / v.replace(0, np.nan)

def compute_score(df: pd.DataFrame) -> pd.Series:
    ema50 = ema(df["close"], 50)
    ema200 = ema(df["close"], 200)
    trend_up = (df["close"] > ema50) & (ema50 > ema200)

    dc_hi, dc_lo = donchian(df, 20)
    breakout_up = df["close"] > dc_hi.shift(1)

    volZ = zscore(df["volume"], 90).fillna(0)
    vol_confirm = volZ > 0.5

    a = atr(df, 20)
    vol_regime = ((a/df["close"]) > (a/df["close"]).rolling(90).median()).fillna(False)

    raw_long = (0.30*trend_up.astype(float) +
                0.30*breakout_up.astype(float) +
                0.20*vol_confirm.astype(float) +
                0.20*vol_regime.astype(float))
    return (100*raw_long.clip(0, 1)).fillna(0)

def bias_and_weight(score: float, long_th: float, short_th: float):
    if score >= long_th:
        w = min(1.0, max(0.0, (score - long_th) / (100 - long_th + 1e-9)))
        return "LONG", round(w, 2)
    if score <= short_th:
        w = min(1.0, max(0.0, (short_th - score) / (short_th + 1e-9)))
        return "SHORT", round(w, 2)
    return "NEUTRAL", 0.0

def bias_badge(bias: str) -> str:
    return {"LONG": "🟢 LONG", "SHORT": "🔴 SHORT", "NEUTRAL": "⚪ NEUTRAL"}[bias]

def funding_alignment(bias: str, fr: float) -> str:
    # LONG is favored if fr <= 0; SHORT favored if fr >= 0
    if pd.isna(fr):
        return "–"
    if bias == "LONG" and fr <= 0:
        return "✅ aligned"
    if bias == "SHORT" and fr >= 0:
        return "✅ aligned"
    if bias == "NEUTRAL":
        return "•"
    return "⚠️"

# ================= Build rows =================
rows = []
sparks = {}
errors = []

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

            # Sparkline data (last N closes) with a safe fallback
            spark_vals = df["close"].tail(spark_len).astype(float).tolist()
            if len(spark_vals) < 2:
                spark_vals = [np.nan, np.nan]
            if show_sparklines:
                sparks[(sym, tf)] = spark_vals

            rows.append({
                "Asset": sym,
                "TF": tf,
                "Price": round(price, 2),
                "Funding %": funding_pct,
                "Score": round(last_sc, 1),
                "Bias": bias_badge(bias),
                "Weight": w,
                "Alignment": align,
                "Last (Berlin)": last_time.tz_convert("Europe/Berlin").strftime("%Y-%m-%d %H:%M"),
                **({"Spark": spark_vals} if show_sparklines else {})
            })
        except Exception as e:
            errors.append((sym, tf, str(e)))
            rows.append({
                "Asset": sym, "TF": tf, "Price": "-", "Funding %": "-", "Score": "-",
                "Bias": "ERR", "Weight": 0.0, "Alignment": "–",
                "Last (Berlin)": str(e), **({"Spark": []} if show_sparklines else {})
            })

table = pd.DataFrame(rows)

# ================= KPIs (top) =================
col_left, col_mid, col_right = st.columns([1,1,2])
if not table.empty:
    longs = (table["Bias"] == "🟢 LONG").sum()
    shorts = (table["Bias"] == "🔴 SHORT").sum()
    neuts = (table["Bias"] == "⚪ NEUTRAL").sum()
    col_left.metric("🟢 LONG", longs)
    col_mid.metric("🔴 SHORT", shorts)
    col_right.metric("⚪ NEUTRAL", neuts)

# Consensus per asset
if not table.empty:
    st.subheader("Consensus per Asset")
    cons_rows = []
    for asset, grp in table.groupby("Asset"):
        c_long = (grp["Bias"] == "🟢 LONG").sum()
        c_short = (grp["Bias"] == "🔴 SHORT").sum()
        c_neut = (grp["Bias"] == "⚪ NEUTRAL").sum()
        cons_rows.append({
            "Asset": asset,
            "Consensus": f"{c_long}×LONG, {c_short}×SHORT, {c_neut}×NEUTRAL"
        })
    st.table(pd.DataFrame(cons_rows))

# ================= Styled Table =================
if not table.empty:
    tf_order = {k: i for i, k in enumerate(["1m","3m","5m","15m","30m","1h","2h","4h","6h","8h","12h","1d"])}
    table["tf_sort"] = table["TF"].map(tf_order).fillna(999)
    table = table.sort_values(["Asset", "tf_sort", "Score"], ascending=[True, True, False]).drop(columns=["tf_sort"])

    st.subheader("Live Status")
    column_config = {
        "Price": st.column_config.NumberColumn(format="%.2f"),
        "Funding %": st.column_config.NumberColumn(format="%.3f"),
        "Score": st.column_config.NumberColumn(format="%.1f"),
        "Weight": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f"),
        "Alignment": st.column_config.TextColumn(),
    }
    if show_sparklines:
        column_config["Spark"] = st.column_config.LineChartColumn(width="small")

    st.dataframe(
        table,
        use_container_width=True,
        column_config=column_config,
        hide_index=True,
    )

# ===== Export / Legend / Diagnostics =====
# Last updated (Berlin)
berlin = pytz.timezone("Europe/Berlin")
ts = datetime.utcnow().replace(tzinfo=pytz.utc).astimezone(berlin).strftime("%Y-%m-%d %H:%M:%S")
st.caption(f"Last update (Berlin): {ts}")

# CSV Download
if not table.empty:
    csv_bytes = table.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download table as CSV", data=csv_bytes, file_name="momentum_dashboard.csv", mime="text/csv")

with st.expander("ℹ️ Interpretation & Rules"):
    st.markdown(f"""
**Bias (trend direction)**
- 🟢 **LONG**: Score > **{entry_long}** → upward momentum
- ⚪ **NEUTRAL**: {entry_short} ≤ Score ≤ {entry_long}
- 🔴 **SHORT**: Score < **{entry_short}** → downward momentum

**Weight (trend strength within zone, 0–1)**
- **≈ 1.00** → strong trend
- **≈ 0.50** → moderate strength / pausing
- **≈ 0.25** → weak/early momentum (caution)

**Funding alignment**
- **✅ aligned**: LONG with funding ≤ 0% or SHORT with funding ≥ 0% (confirmation)
- **⚠️**: momentum vs. funding diverge → be conservative

**Multi-timeframe reading**
- All TFs agree & Weight > 0.7 → robust trend
- 5m flips first → early warning for potential reversal
- 1h & 4h dominate → higher-timeframe context

**Signal triggers (for Telegram)**
- **🟢 Long setup:** Score **crosses up** above **{entry_long}** (prefer funding ≤ 0%)
- **🔴 Short setup:** Score **crosses down** below **{entry_short}** (prefer funding ≥ 0%)
- **⚠️ Exit warning:** Score drops **below 55**
- **🚪 Hard exit:** Score drops **below 45**
""")

if errors:
    with st.expander("🛠️ Diagnostics / Errors"):
        err_df = pd.DataFrame(errors, columns=["Asset", "TF", "Error"])
        st.dataframe(err_df, use_container_width=True, hide_index=True)

st.caption("Data: Binance (multiple domains) → fallback to Kraken OHLCV on 451; Funding from Binance Futures. Not financial advice.")
