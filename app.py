import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import timezone

st.set_page_config(page_title="Momentum Dashboard", layout="wide")
st.title("Momentum • Live Bias & Funding (Binance, direct API)")

# ---------------- UI ----------------
default_assets = ["BTC/USDT", "ETH/USDT"]
assets = st.multiselect("Assets", default_assets, default=default_assets)
tfs = st.multiselect("Timeframes", ["5m","1h","4h"], default=["5m","1h","4h"])
entry_long = st.slider("Long-Schwelle", 50, 80, 60, 1)
entry_short = st.slider("Short-Schwelle", 20, 60, 40, 1)

# ---------------- Helpers ----------------
def sym_to_perp(sym:str) -> str:
    # "BTC/USDT" -> "BTCUSDT"
    return sym.replace("/", "")

def interval_ok(tf:str) -> bool:
    return tf in {"1m","3m","5m","15m","30m","1h","2h","4h","6h","8h","12h","1d"}

@st.cache_data(ttl=60)
def fetch_klines_binance(symbol_perp:str, interval:str, limit:int=1000) -> pd.DataFrame:
    """Direct public endpoint: avoids exchangeInfo (which triggers 451)."""
    assert interval_ok(interval)
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol_perp, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()  # list of lists
    cols = ["open_time","open","high","low","close","volume",
            "close_time","qav","num_trades","taker_base","taker_quote","ignore"]
    df = pd.DataFrame(data, columns=cols)
    df["time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    out = df.set_index("time")[["open","high","low","close","volume"]].astype(float)
    return out

@st.cache_data(ttl=60)
def fetch_funding_binance(symbol_perp:str) -> float:
    """Binance futures funding rate (public). Returns fraction (e.g., 0.0001 = 0.01%)."""
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    r = requests.get(url, params={"symbol": symbol_perp, "limit": 1}, timeout=15)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, list) and data:
        return float(data[-1]["fundingRate"])
    return np.nan

# Indicators
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
    return (100*raw_long.clip(0,1)).fillna(0)

def bias_and_weight(score: float, long_th: float, short_th: float):
    if score >= long_th:
        w = min(1.0, max(0.0), (score - long_th) / (100-long_th + 1e-9))
        return "LONG", round(w,2)
    if score <= short_th:
        w = min(1.0, max(0.0), (short_th - score) / (short_th + 1e-9))
        return "SHORT", round(w,2)
    return "NEUTRAL", 0.0

# ---------------- Build table ----------------
rows = []
for sym in assets:
    perp = sym_to_perp(sym)
    for tf in tfs:
        try:
            df = fetch_klines_binance(perp, tf, limit=1000)
            sc = compute_score(df)
            last_sc = float(sc.iloc[-1])
            price = float(df["close"].iloc[-1])
            last_time = sc.index[-1]
            try:
                fr = fetch_funding_binance(perp)  # 0.0001 = 0.01%
                funding_pct = round(fr*100, 3)
            except Exception as e_f:
                funding_pct = np.nan
            bias, w = bias_and_weight(last_sc, entry_long, entry_short)
            rows.append({
                "Asset": sym,
                "TF": tf,
                "Price": round(price, 2),
                "Funding %": funding_pct if pd.notna(funding_pct) else "-",
                "Score": round(last_sc, 1),
                "Bias": bias,
                "Weight": w,
                "Last": last_time.tz_convert("Europe/Berlin").strftime("%Y-%m-%d %H:%M")
            })
        except Exception as e:
            rows.append({
                "Asset": sym, "TF": tf,
                "Price": "-", "Funding %": "-", "Score": "-",
                "Bias": "ERR", "Weight": "-",
                "Last": str(e)
            })

table = pd.DataFrame(rows)
if not table.empty:
    tf_order = {k:i for i,k in enumerate(["1m","3m","5m","15m","30m","1h","2h","4h","6h","8h","12h","1d"])}
    table["tf_sort"] = table["TF"].map(tf_order).fillna(999)
    table = table.sort_values(["Asset","tf_sort","Score"], ascending=[True,True,False]).drop(columns=["tf_sort"])

st.subheader("Live Status")
st.dataframe(table, use_container_width=True)
st.caption("Direkte Binance-Endpoints (klines/funding) – ohne exchangeInfo. Score kombiniert Trend (EMA), Donchian-Breakout, Volumen-Z, Volatilitätsregime. Keine Handelsempfehlung.")
