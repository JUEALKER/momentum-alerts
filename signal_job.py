#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Momentum signal job with persisted state, higher-TF alignment, persistence, and exit logic.

ENV (GitHub Actions Secrets / local):
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
  STATE_PATH (e.g., 'state/prev_bias.json')
  TZ_NAME (default 'Europe/Berlin')
  PERSIST_N (default '2')

Example:
  python signal_job.py \
    --assets "BTC/USDT,ETH/USDT,BNB/USDT,SOL/USDT,XRP/USDT" \
    --timeframes "5m,1h,4h" \
    --limit 1000 \
    --entry_long 60 --entry_short 40 \
    --exit_warn 55 --exit_hard 45 \
    --min_weight 0.30 --cooldown 15
"""

import os, sys, json, time, math, pathlib, argparse, random
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests

# -------------------- ENV/CONFIG --------------------
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

TZ_NAME   = os.getenv("TZ_NAME", "Europe/Berlin")
STATE_PATH = os.getenv("STATE_PATH", "state/prev_bias.json").strip()
PERSIST_N  = int(os.getenv("PERSIST_N", "2"))

TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TG_CHAT  = os.getenv("TELEGRAM_CHAT_ID", "").strip()

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "momentum-alerts/1.0"})

# -------------------- UTIL --------------------
def log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{ts}] {msg}", flush=True)

def jitter_sleep(ms_low=120, ms_high=420):
    time.sleep(random.uniform(ms_low, ms_high)/1000.0)

def sym_to_perp(sym: str) -> str:
    return sym.replace("/", "")

def tf_to_kraken(tf: str) -> int:
    return {"5m": 5, "1h": 60, "4h": 240}.get(tf, 60)

def to_local(dt_utc: pd.Timestamp) -> str:
    try:
        return dt_utc.tz_convert(TZ_NAME).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return dt_utc.tz_localize("UTC").tz_convert(TZ_NAME).strftime("%Y-%m-%d %H:%M")

# -------------------- STATE --------------------
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

# -------------------- TELEGRAM --------------------
def tg_send(text: str) -> bool:
    if not TG_TOKEN or not TG_CHAT:
        return False
    try:
        r = SESSION.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT, "text": text, "parse_mode": "HTML"},
            timeout=12
        )
        r.raise_for_status()
        return True
    except Exception as e:
        log(f"Telegram send failed: {e}")
        return False

# -------------------- HTTP HELPERS --------------------
def _get_json(url, params=None, timeout=15, retries=2):
    last_err = None
    for i in range(retries + 1):
        try:
            r = SESSION.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            jitter_sleep()
    if last_err:
        raise last_err

# -------------------- MARKET DATA --------------------
def fetch_binance_ohlcv(perp: str, interval: str, limit=1000) -> pd.DataFrame:
    last = None
    for base in BINANCE_SPOT:
        try:
            j = _get_json(f"{base}/api/v3/klines",
                          params={"symbol": perp, "interval": interval, "limit": limit}, timeout=20)
            cols = ["t","o","h","l","c","v","ct","qa","nt","tb","tq","ig"]
            df = pd.DataFrame(j, columns=cols)
            df["time"] = pd.to_datetime(df["ct"], unit="ms", utc=True)
            out = df.set_index("time")[["o","h","l","c","v"]].astype(float)
            out.columns = ["open","high","low","close","volume"]
            return out
        except Exception as e:
            last = e
            jitter_sleep()
            continue
    raise RuntimeError(f"Binance spot unavailable for {perp} ({last})")

def fetch_kraken_ohlcv(sym: str, tf: str, limit=1000) -> pd.DataFrame:
    kp = KRAKEN_PAIR.get(sym)
    if not kp:
        raise RuntimeError(f"No Kraken mapping for {sym}")
    j = _get_json("https://api.kraken.com/0/public/OHLC",
                  params={"pair": kp, "interval": tf_to_kraken(tf)}, timeout=20)
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
            j = _get_json(f"{base}/fapi/v1/fundingRate", params={"symbol": perp, "limit": 1}, timeout=15)
            if isinstance(j, list) and j:
                return float(j[-1]["fundingRate"])  # e.g., 0.0001 -> 0.01%
        except Exception:
            jitter_sleep()
            continue
    return float("nan")

# -------------------- INDICATORS / SCORE --------------------
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

# -------------------- CLI --------------------
def parse_args():
    p = argparse.ArgumentParser(description="Momentum alert job with persisted state & exits")
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

# -------------------- MAIN --------------------
def main():
    args = parse_args()
    assets = [a.strip() for a in args.assets.split(",") if a.strip()]
    tfs    = [t.strip() for t in args.timeframes.split(",") if t.strip()]
    limit  = args.limit

    # small random delay to avoid thundering herd on schedule
    jitter_sleep(200, 900)

    log(f"Start job | assets={assets} | tfs={tfs} | limit={limit} | L/S={args.entry_long}/{args.entry_short} | minW={args.min_weight} | cd={args.cooldown}m | persist={PERSIST_N}")

    state = load_state()
    prev_bias     = state.get("prev_bias", {})         # "ASSET|TF" -> "🟢 LONG"/"🔴 SHORT"/"⚪ NEUTRAL"
    last_alert_ts = state.get("last_alert_ts", {})     # "ASSET|TF" -> epoch seconds
    consec        = state.get("consec", {})            # "ASSET|TF|BIAS" -> consecutive bars
    positions     = state.get("positions", {})         # "ASSET" -> {"side","entry","stop"}

    curr_bias = {}  # current run biases for alignment checks
    alerts_sent = 0
    now = time.time()

    for sym in assets:
        perp = sym_to_perp(sym)
        try:
            fr = fetch_funding_rate(perp)
        except Exception as e:
            fr = float("nan")
            log(f"Funding fetch failed for {sym}: {e}")

        # process each TF
        for tf in tfs:
            key = f"{sym}|{tf}"
            try:
                df = fetch_ohlcv(sym, tf, limit)
                sc = compute_score(df)
                s = float(sc.iloc[-1])
                price = float(df["close"].iloc[-1])
                ts_utc = sc.index[-1]

                # indicators for exits
                a20 = float(atr(df, 20).iloc[-1])

                # bias & weight
                curr_badge, w = bias_and_weight(s, args.entry_long, args.entry_short)
                align = funding_alignment(curr_badge, fr)
                curr_bias[key] = curr_badge

                # update persistence counters
                c_key = f"{key}|{curr_badge}"
                consec[c_key] = int(consec.get(c_key, 0)) + 1
                for other in ("🟢 LONG", "🔴 SHORT", "⚪ NEUTRAL"):
                    if other != curr_badge:
                        o_key = f"{key}|{other}"
                        if o_key in consec:  # reset opposite streaks
                            consec[o_key] = 0

                # --- Lead TF entry rule & persistence (alerts only on 5m entries) ---
                prev = prev_bias.get(key)
                cooldown_ok = (now - float(last_alert_ts.get(key, 0))) >= args.cooldown * 60

                def higher_tf_agree(sym_: str, desired: str) -> bool:
                    k1 = f"{sym_}|1h"; k2 = f"{sym_}|4h"
                    b1 = curr_bias.get(k1) or prev_bias.get(k1)
                    b2 = curr_bias.get(k2) or prev_bias.get(k2)
                    return (b1 == desired) and (b2 == desired)

                should_alert_entry = False
                if tf == "5m":
                    if (prev is not None) and (curr_badge != prev) and (w >= args.min_weight) and cooldown_ok:
                        if (curr_badge in ("🟢 LONG", "🔴 SHORT")) and higher_tf_agree(sym, curr_badge):
                            if consec.get(c_key, 0) >= PERSIST_N:
                                should_alert_entry = True

                if should_alert_entry:
                    text = (
                        f"⚡ Momentum change on <b>{sym}</b> ({tf})\n"
                        f"{prev} → <b>{curr_badge}</b>\n"
                        f"Weight: <b>{w:.2f}</b> | Score: <b>{s:.1f}</b>\n"
                        f"Price: <b>{price:.2f}</b>\n"
                        f"Funding: <b>{'-' if np.isnan(fr) else str(round(fr*100,3))+'%'}</b> ({align})\n"
                        f"Time: {to_local(ts_utc)} ({TZ_NAME})"
                    )
                    if tg_send(text):
                        alerts_sent += 1
                        last_alert_ts[key] = now
                        log(f"ALERT ENTRY {key} | w={w:.2f} s={s:.1f} {prev}→{curr_badge}")

                        # open/update virtual position with ATR stop
                        side = "LONG" if "🟢" in curr_badge else "SHORT"
                        entry = price
                        stop = entry - 1.5*a20 if side == "LONG" else entry + 1.5*a20
                        positions[sym] = {"side": side, "entry": entry, "stop": stop}

                # --- Manage exits / trailing (evaluate once per asset on 5m) ---
                if tf == "5m" and sym in positions:
                    pos = positions[sym]
                    # Trail stop (2*ATR)
                    if pos["side"] == "LONG":
                        new_stop = max(pos["stop"], price - 2.0*a20)
                    else:
                        new_stop = min(pos["stop"], price + 2.0*a20)
                    positions[sym]["stop"] = new_stop

                    # Score-based exits
                    # LONG side exits when score degrades; SHORT mirrored around 100
                    if pos["side"] == "LONG":
                        if s <= args.exit_hard:
                            tg_send(f"🚪 HARD EXIT {sym} LONG — Score {s:.1f} | Price {price:.2f}")
                            positions.pop(sym, None)
                        elif s <= args.exit_warn:
                            tg_send(f"⚠️ EXIT WARN {sym} LONG — Score {s:.1f} | Price {price:.2f}")
                    else:  # SHORT
                        if s >= (100 - args.exit_hard):
                            tg_send(f"🚪 HARD EXIT {sym} SHORT — Score {s:.1f} | Price {price:.2f}")
                            positions.pop(sym, None)
                        elif s >= (100 - args.exit_warn):
                            tg_send(f"⚠️ EXIT WARN {sym} SHORT — Score {s:.1f} | Price {price:.2f}")

                    # ATR stop breach
                    if sym in positions:
                        pos = positions[sym]
                        if (pos["side"] == "LONG" and price <= pos["stop"]) or \
                           (pos["side"] == "SHORT" and price >= pos["stop"]):
                            tg_send(f"🛑 ATR STOP {sym} {pos['side']} hit at {pos['stop']:.2f} (Price {price:.2f})")
                            positions.pop(sym, None)

                # Update stored bias every run
                prev_bias[key] = curr_badge

            except Exception as e:
                log(f"Data/compute error for {key}: {e}")
                # Keep previous bias on failure; continue

    # Save state
    new_state = {
        "prev_bias": prev_bias,
        "last_alert_ts": last_alert_ts,
        "consec": consec,
        "positions": positions
    }
    save_state(new_state)
    log(f"Done. Alerts sent: {alerts_sent}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
