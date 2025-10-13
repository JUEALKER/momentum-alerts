import streamlit as st
import pandas as pd
import numpy as np
import requests

st.set_page_config(page_title="Momentum Dashboard", layout="wide")
st.title("Momentum • Live Bias & Funding")

# ---------------- UI ----------------
default_assets = ["BTC/USDT", "ETH/USDT"]
assets = st.multiselect("Assets", default_assets, default=default_assets)
tfs = st.multiselect("Timeframes", ["5m","1h","4h"], default=["5m","1h","4h"])
entry_long = st.slider("Long-Schwelle", 50, 80, 60, 1)
entry_short = st.slider("Short-Schwelle", 20, 60, 40, 1)

# ---------------- Helpers ----------------
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

def sym_to_perp(sym:str) -> str:
    return sym.replace("/", "")  # BTC/USDT -> BTCUSDT

def tf_to_binance(tf:str) -> str:
    return tf  # unsere Auswahl passt zu Binance

def tf_to_kraken(tf:str) -> int:
    # Kraken unterstützt Minuten-Intervalle: 1,5,15,30,60,240,1440 ...
    return {"5m":5, "1h":60, "4h":240}.get(tf, 60)

def fetch_klines_binance_any(perp:str, interval:str, limit:int=1000) -> pd.DataFrame:
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

def fetch_klines_kraken(sym:str, tf:str, limit:int=1000) -> pd.DataFrame:
    # Mappe nur BTC/USDT / ETH/USDT -> Kraken-Paare
    pair_map = {"BTC/USDT": "XBTUSDT", "ETH/USDT": "ETHUSDT"}
    kr_pair = pair_map.get(sym)
    if not kr_pair:
        raise ValueError("Unsupported symbol for Kraken fallback")
    interval = tf_to_kraken(tf)
    r = requests.get("https://api.kraken.com/0/public/OHLC",
                     params={"pair": kr_pair, "interval": interval},
                     timeout=15)
    r.raise_for_status()
    data = r.json()
    # Kraken liefert unter result[PAIR] eine Liste: [time, open, high, low, close, vwap, volume, count]
    k = next(iter(data["result"]))  # Paar-Key
    rows = data["result"][k]
    df = pd.DataFrame(rows, columns=["time","open","high","low","close","vwap","volume","count"])
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    out = df.set_index("time")[["open","high","low","close","volume"]].astype(float)
    return out.tail(limit)

@st.cache_data(ttl=60)
def fetch_ohlcv(symbol:str, tf:str, limit:int=1000) -> pd.DataFrame:
    # Versuche Binance → sonst Kraken
    perp = sym_to_perp(symbol)
    try:
        return fetch_klines_binance_any(perp, tf_to_binance(tf), limit)
    except Exception:
        return fetch_klines_kraken(symbol, tf, limit)

@st.cache_data(ttl=60)
def fetch_funding(perp:str) -> float:
    last_exc = None
    for base in BINANCE_FUT_BASES:
        try:
            r = requests.get(f"{base}/fapi/v1/fundingRate",
                             params={"symbol": perp, "limit": 1}, timeout=15)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list) and data:
                return float(data[-1]["fundingRate"])  # 0.0001 = 0.01%
        except Exception as e:
            last_exc = e
            continue
    # Wenn alle blockiert sind: Funding ausblenden
    return np.nan

# ---- Indicators ----
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
        w = min(1.0, max(0.0, (score - long_th) / (100-long_th + 1e-9)))
        return "LONG", round(w,2)
    if score <= short_th:
        w = min(1.0, max(0.0, (short_th - score) / (short_th + 1e-9)))
        return "SHORT", round(w,2)
    return "NEUTRAL", 0.0

# ---------------- Build table ----------------
rows = []
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
            funding_pct = "-" if pd.isna(fr) else round(fr*100, 3)

            bias, w = bias_and_weight(last_sc, entry_long, entry_short)
            rows.append({
                "Asset": sym,
                "TF": tf,
                "Price": round(price, 2),
                "Funding %": funding_pct,
                "Score": round(last_sc, 1),
                "Bias": bias,
                "Weight": w,
                "Last": last_time.tz_convert("Europe/Berlin").strftime("%Y-%m-%d %H:%M")
            })
        except Exception as e:
            rows.append({"Asset": sym, "TF": tf, "Price": "-", "Funding %": "-",
                         "Score": "-", "Bias": "ERR", "Weight": "-", "Last": str(e)})

table = pd.DataFrame(rows)
if not table.empty:
    tf_order = {k:i for i,k in enumerate(["1m","3m","5m","15m","30m","1h","2h","4h","6h","8h","12h","1d"])}
    table["tf_sort"] = table["TF"].map(tf_order).fillna(999)
    table = table.sort_values(["Asset","tf_sort","Score"], ascending=[True,True,False]).drop(columns=["tf_sort"])

st.subheader("Live Status")
st.dataframe(table, use_container_width=True)
st.caption("Mehrere Binance-Domains mit Fallback; bei 451 → Kraken-OHLCV. Funding: Binance-Futures (Fallback-Domains). Score: EMA/Donchian/Vol-Z/ATR-Regime.")
