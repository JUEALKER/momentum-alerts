import streamlit as st
import pandas as pd
import numpy as np
import ccxt
from datetime import timedelta

st.set_page_config(page_title="Momentum Dashboard", layout="wide")
st.title("Momentum • Live Bias & Funding (Binance)")

# ---- UI ----
default_assets = ["BTC/USDT", "ETH/USDT"]
assets = st.multiselect("Assets", default_assets, default=default_assets)
tfs = st.multiselect("Timeframes", ["5m","1h","4h"], default=["5m","1h","4h"])
entry_long = st.slider("Long-Schwelle", 50, 80, 60, 1)
entry_short = st.slider("Short-Schwelle", 20, 60, 40, 1)

# ---- Helpers / Indicators ----
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
        (df["low"] -pc).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()

def zscore(s, n=90):
    m = s.rolling(n).mean()
    v = s.rolling(n).std(ddof=0)
    return (s - m) / v.replace(0, np.nan)

@st.cache_data(ttl=60)  # 60s Cache, schont die API
def fetch_ohlcv(symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
    ex = ccxt.binance()
    raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["ts","open","high","low","close","volume"])
    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("time")[["open","high","low","close","volume"]].astype(float)
    return df

@st.cache_data(ttl=60)
def fetch_funding_rate_binance(perp_symbol: str) -> float:
    # perp_symbol: "BTCUSDT", "ETHUSDT" ...
    ex = ccxt.binance()
    data = ex.fapiPublicGetFundingRate({"symbol": perp_symbol, "limit": 1})
    return float(data[-1]["fundingRate"])  # z.B. 0.0001 = 0.01%

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
        w = min(1.0, max(0.0, (score - long_th)/(100-long_th+1e-9)))
        return "LONG", round(w,2)
    if score <= short_th:
        w = min(1.0, max(0.0, (short_th - score)/(short_th+1e-9)))
        return "SHORT", round(w,2)
    return "NEUTRAL", 0.0

# ---- Build table ----
rows = []
for sym in assets:
    perp = sym.replace("/", "")  # BTC/USDT -> BTCUSDT (für Funding)
    for tf in tfs:
        try:
            df = fetch_ohlcv(sym, tf, limit=1000)
            sc = compute_score(df)
            last_sc = float(sc.iloc[-1])
            price = float(df["close"].iloc[-1])
            last_time = sc.index[-1]
            fr = fetch_funding_rate_binance(perp)  # 0.0001 = 0.01%
            bias, w = bias_and_weight(last_sc, entry_long, entry_short)
            rows.append({
                "Asset": sym,
                "TF": tf,
                "Price": price,
                "Funding %": round(fr*100, 3),
                "Score": round(last_sc, 1),
                "Bias": bias,
                "Weight": w,
                "Last": last_time.tz_convert("Europe/Berlin").strftime("%Y-%m-%d %H:%M")
            })
        except Exception as e:
            rows.append({"Asset": sym, "TF": tf, "Price": "-", "Funding %": "-", "Score": "-", "Bias": "ERR", "Weight": "-", "Last": str(e)})

table = pd.DataFrame(rows)
if not table.empty:
    # sortiere: erst TF, dann Score absteigend
    tf_order = {k:i for i,k in enumerate(["5m","1h","4h","1d"])}
    table["tf_sort"] = table["TF"].map(tf_order).fillna(99)
    table = table.sort_values(["Asset","tf_sort","Score"], ascending=[True,True,False]).drop(columns=["tf_sort"])

st.subheader("Live Status")
st.dataframe(table, use_container_width=True)
st.caption("Hinweis: Funding von Binance Futures; OHLCV von Binance Spot. Score kombiniert Trend (EMA), Donchian-Breakout, Volumen-Z, Volatilitätsregime. Keine Handelsempfehlung.")
