#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Momentum signal job with persisted state across CI runs.

- Fetches OHLCV from Binance (fallback Kraken)
- Computes momentum score (0..100), Bias (LONG/SHORT/NEUTRAL), Weight (0..1)
- Compares current Bias to last stored Bias per (asset, timeframe)
- Sends Telegram alert on Bias change if Weight >= min_weight and cooldown passed
- Persists state (prev_bias and last_alert_ts) to JSON at STATE_PATH

ENV (from GitHub Actions secrets / env):
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
  STATE_PATH (e.g., 'state/prev_bias.json')
  TZ_NAME (optional, default 'Europe/Berlin')

Example run (from workflow):
  python signal_job.py \
    --assets "BTC/USDT,ETH/USDT,BNB/USDT,SOL/USDT,XRP/USDT" \
    --timeframes "5m,1h,4h" \
    --limit 1000 \
    --entry_long 60 --entry_short 40 \
    --exit_warn 55 --exit_hard 45 \
    --min_weight 0.30 --cooldown 15
"""

import os, sys, json, pathlib, time
import argparse
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests

# -------------------- CONFIG / ENV --------------------
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
TZ_NAME = os.getenv("TZ_NAME", "Europe/Berlin")
STATE_PATH = os.getenv("STATE_PATH", "state/prev_bias.json").strip()

TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TG_CHAT  = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# --------------- SMALL UTILS ----------------
def log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{ts}] {msg}", flush=True)

def sym_to_perp(sym: str) -> str:
    return sym.replace("/", "")

def tf_to_kraken(tf: str) -> int:
    return {"5m": 5, "1h": 60, "4h": 240}.get(tf, 60)

def to_local(dt_utc: pd.Timestamp) -> str:
    try:
        return dt_utc.tz_convert(TZ_NAME).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return dt_utc.tz_localize("UTC").tz_convert(TZ_NAME).strftime("%Y-%m-%d %H:%M")

# --------------- STATE PERSISTENCE ---------------
def load_state(path=STATE_PATH) -> dict:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def save_state(state: dict, path=STATE_PATH):
    pathlib.Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)

# --------------- TELEGRAM ----------------
def tg_send(text: str) -> bool:
    if not TG_TOKEN or not TG_CHAT:
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT, "text": text, "parse_mode": "HTML"},
            timeout=12
        )
        r.raise_for_status()
        return True
    except Exception as e:
        log(f"Telegram send failed: {e}")
        return False

# --------------- DATA FETCH ----------------
def fetch_binance_ohlcv(perp: str, interval: str, limit=1000) -> pd.DataFrame:
    for base in BINANCE_SPOT:
        try:
            r = requests.get(f"{base}/api/v3/klines",
                             params={"symbol": perp, "interval": interval, "limit": limit},
                             timeout=15)
            r.raise_for_status()
            data = r.json()
            cols = ["t","o","h","l","c","v","ct","qa","nt","tb","tq","ig"]
            df = pd.DataFrame(data, columns=cols)
            df["time"] = pd.to_datetime(df["ct"], unit="ms", utc=True)
            out = df.set_index("time")[["o","h","l","c","v"]].astype(float)
            out.columns = ["open","high","low","close","volume"]
            return out
        except Exception as e:
            last = str(e)
            continue
    raise RuntimeError(f"Binance spot unavailable for {perp} ({last})")

def fetch_kraken_ohlcv(sym: str, tf: str, limit=1000) -> pd.DataFrame:
    kp = KRAKEN_PAIR.get(sym)
    if not kp:
        raise RuntimeError(f"No Kraken mapping for {sym}")
    r = requests.get("https://api.kraken.com/0/public/OHLC",
                     params={"pair": kp, "interval": tf_to_kraken(tf)},
                     timeout=15)
    r.raise_for_status()
    j = r.json()
    key = next(iter(j["result"]))
    df = pd.DataFrame(j["result"][key],
                      columns=["t","o","h","l","c","vwap","vol","count"])
    df["time"] = pd.to_datetime(df["t"], unit="s", utc=True)
    out = df.set_index("time")[["o","h","l","c","vol"]].astype(float)
    out.columns = ["open","high","low","close","volume"]
    return out.tail(limit)

def fetch_ohlcv(sym: str, tf: str, limit=1000) -> pd.DataFrame:
    perp = sym_to_perp(sym)
    try:
        return fetch_binance_ohlcv(perp, tf, limit)
    except Exception as e1:
        log(f"Binance fail for {sym} {tf}: {e1} → trying Kraken")
        return fetch_kraken_ohlcv(sym, tf, limit)

def fetch_funding_rate(perp: str) -> float:
    for base in BINANCE_FUT:
        try:
            r = requests.get(f"{base}/fapi/v1/fundingRate",
                             params={"symbol": perp, "limit": 1}, timeout=12)
            r.raise_for_status()
            j = r.json()
            if isinstance(j, list) and j:
                return float(j[-1]["fundingRate"])  # e.g., 0.0001 → 0.01%
        except Exception:
            continue
    return float("nan")

# --------------- INDICATORS / SCORE ----------------
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def atr(df: pd.DataFrame, n: int = 20) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - pc).abs(),
        (df["low"] - pc).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def zscore(s: pd.Series, n: int = 90) -> pd.Series:
    m, v = s.rolling(n).mean(), s.rolling(n).std(ddof=0)
    return (s - m) / v.replace(0, np.nan)

def donchian(df: pd.DataFrame, n: int = 20):
    return df["high"].rolling(n).max(), df["low"].rolling(n).min()

def compute_score(df: pd.DataFrame) -> pd.Series:
    ema50, ema200 = ema(df["close"], 50), ema(df["close"], 200)
    trend = (df["close"] > ema50) & (ema50 > ema200)
    dc_hi, _ = donchian(df, 20)
    breakout = df["close"] > dc_hi.shift(1)
    volZ = zscore(df["volume"], 90).fillna(0)
    vol_conf = volZ > 0.5
    a = atr(df, 20)
    vol_regime = ((a / df["close"]) >
                  (a / df["close"]).rolling(90).median()).fillna(False)
    raw = 0.3 * trend + 0.3 * breakout + 0.2 * vol_conf + 0.2 * vol_regime
    return (100 * raw.clip(0, 1)).fillna(0)

def bias_and_weight(score: float, long_t: int, short_t: int):
    if score >= long_t:
        w = min(1.0, (score - long_t) / (100 - long_t + 1e-9))
        return "🟢 LONG", round(w, 2)
    if score <= short_t:
        w = min(1.0, (short_t - score) / (short_t + 1e-9))
        return "🔴 SHORT", round(w, 2)
    return "⚪ NEUTRAL", 0.0

def funding_alignment(bias: str, fr: float) -> str:
    if np.isnan(fr): return "–"
    if bias == "🟢 LONG" and fr <= 0: return "✅ aligned"
    if bias == "🔴 SHORT" and fr >= 0: return "✅ aligned"
    if bias == "⚪ NEUTRAL": return "•"
    return "⚠️"

# --------------- MAIN JOB ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Momentum alert job with persisted state")
    p.add_argument("--assets", type=str, required=True,
                   help='Comma-separated, e.g. "BTC/USDT,ETH/USDT,BNB/USDT"')
    p.add_argument("--timeframes", type=str, required=True,
                   help='Comma-separated, e.g. "5m,1h,4h"')
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--entry_long", type=int, default=60)
    p.add_argument("--entry_short", type=int, default=40)
    p.add_argument("--exit_warn", type=int, default=55)
    p.add_argument("--exit_hard", type=int, default=45)
    p.add_argument("--min_weight", type=float, default=0.30,
                   help="Min weight to alert on bias change")
    p.add_argument("--cooldown", type=int, default=15,
                   help="Cooldown minutes per (asset, timeframe)")
    return p.parse_args()

def main():
    args = parse_args()
    assets = [a.strip() for a in args.assets.split(",") if a.strip()]
    tfs = [t.strip() for t in args.timeframes.split(",") if t.strip()]
    limit = args.limit

    log(f"Start job | assets={assets} | tfs={tfs} | limit={limit} | thresholds L={args.entry_long}/S={args.entry_short} | min_weight={args.min_weight} | cooldown={args.cooldown}m")
    state = load_state()
    prev_bias = state.get("prev_bias", {})           # key: "ASSET|TF" -> "🟢 LONG"/"🔴 SHORT"/"⚪ NEUTRAL"
    last_alert_ts = state.get("last_alert_ts", {})   # key: epoch seconds

    alerts_sent = 0
    now = time.time()

    for sym in assets:
        perp = sym_to_perp(sym)
        try:
            fr = fetch_funding_rate(perp)
        except Exception as e:
            fr = float("nan")
            log(f"Funding fetch failed for {sym}: {e}")

        for tf in tfs:
            key = f"{sym}|{tf}"
            try:
                df = fetch_ohlcv(sym, tf, limit)
                sc = compute_score(df)
                s = float(sc.iloc[-1])
                price = float(df["close"].iloc[-1])
                ts_utc = sc.index[-1]  # pandas Timestamp (UTC)

                bias, w = bias_and_weight(s, args.entry_long, args.entry_short)
                align = funding_alignment(bias, fr)

                # Compare with previous bias
                prev = prev_bias.get(key)
                cooldown_ok = (now - float(last_alert_ts.get(key, 0))) >= args.cooldown * 60

                # Decide alert
                if (prev is not None) and (bias != prev) and (w >= args.min_weight) and cooldown_ok:
                    text = (
                        f"⚡ Momentum change on <b>{sym}</b> ({tf})\n"
                        f"{prev} → <b>{bias}</b>\n"
                        f"Weight: <b>{w:.2f}</b> | Score: <b>{s:.1f}</b>\n"
                        f"Price: <b>{price:.2f}</b>\n"
                        f"Funding: <b>{'-' if np.isnan(fr) else str(round(fr*100,3))+'%'}</b> ({align})\n"
                        f"Time: {to_local(ts_utc)} ({TZ_NAME})"
                    )
                    if tg_send(text):
                        alerts_sent += 1
                        last_alert_ts[key] = now
                        log(f"ALERT sent for {key} | w={w:.2f} s={s:.1f} bias {prev}→{bias}")
                    else:
                        log(f"ALERT failed send for {key}")

                # Update stored bias (always)
                prev_bias[key] = bias

            except Exception as e:
                log(f"Data/compute error for {key}: {e}")
                # Do not change prev bias on failure; continue

    # Save state
    new_state = {"prev_bias": prev_bias, "last_alert_ts": last_alert_ts}
    save_state(new_state)
    log(f"Done. Alerts sent: {alerts_sent}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
