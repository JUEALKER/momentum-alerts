# signal_job.py — Build: SJ-v5 (GH Actions friendly)
import os
import json
import time
import math
import argparse
from datetime import datetime, timezone
import pytz
import requests
import numpy as np
import pandas as pd

BERLIN = pytz.timezone("Europe/Berlin")

# ------------- EXCHANGE ENDPOINTS -------------
BINANCE_SPOT = [
    "https://api.binance.com", "https://api1.binance.com",
    "https://api2.binance.com", "https://api3.binance.com",
    "https://data-api.binance.vision",
]
BINANCE_FUT = [
    "https://fapi.binance.com", "https://fapi1.binance.com",
    "https://fapi2.binance.com", "https://fapi3.binance.com",
]
KRAKEN_PAIR = {
    "BTC/USDT": "XBTUSDT", "ETH/USDT": "ETHUSDT",
    "BNB/USDT": "BNBUSDT", "SOL/USDT": "SOLUSDT", "XRP/USDT": "XRPUSDT",
}

def sym_to_perp(sym: str) -> str:
    return sym.replace("/", "")

def tf_to_kraken(tf: str) -> int:
    return {"5m": 5, "1h": 60, "4h": 240}.get(tf, 60)

# ------------- TELEGRAM -------------
def send_telegram(token: str, chat_id: str, text: str) -> bool:
    if not (token and chat_id):
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=12,
        )
        r.raise_for_status()
        return True
    except Exception as e:
        print(f"[WARN] Telegram send failed: {e}")
        return False

# ------------- DATA FETCHERS (Binance, fallback Kraken) -------------
def fetch_binance_klines(perp: str, interval: str, limit: int = 1000) -> pd.DataFrame:
    for base in BINANCE_SPOT:
        try:
            r = requests.get(
                f"{base}/api/v3/klines",
                params={"symbol": perp, "interval": interval, "limit": limit},
                timeout=15,
            )
            r.raise_for_status()
            data = r.json()
            cols = ["t","o","h","l","c","v","ct","qa","nt","tb","tq","ig"]
            df = pd.DataFrame(data, columns=cols)
            df["time"] = pd.to_datetime(df["ct"], unit="ms", utc=True)
            out = df.set_index("time")[["o","h","l","c","v"]].astype(float)
            out.columns = ["open","high","low","close","volume"]
            return out
        except Exception as e:
            # try next mirror
            last_err = str(e)
            continue
    raise RuntimeError(f"Binance klines unavailable for {perp}@{interval}: {last_err}")

def fetch_kraken_ohlc(sym: str, tf: str, limit: int = 1000) -> pd.DataFrame:
    kp = KRAKEN_PAIR.get(sym)
    r = requests.get(
        "https://api.kraken.com/0/public/OHLC",
        params={"pair": kp, "interval": tf_to_kraken(tf)},
        timeout=15,
    )
    r.raise_for_status()
    j = r.json()
    key = next(iter(j["result"]))
    df = pd.DataFrame(j["result"][key], columns=["t","o","h","l","c","vwap","vol","count"])
    df["time"] = pd.to_datetime(df["t"], unit="s", utc=True)
    out = df.set_index("time")[["o","h","l","c","vol"]].astype(float)
    out.columns = ["open","high","low","close","volume"]
    return out.tail(limit)

def fetch_ohlcv(sym: str, tf: str, limit: int = 1000) -> pd.DataFrame:
    perp = sym_to_perp(sym)
    try:
        return fetch_binance_klines(perp, tf, limit)
    except Exception as e:
        print(f"[INFO] Falling back to Kraken for {sym}@{tf}: {e}")
        return fetch_kraken_ohlc(sym, tf, limit)

def fetch_funding(perp: str) -> float:
    for base in BINANCE_FUT:
        try:
            r = requests.get(
                f"{base}/fapi/v1/fundingRate",
                params={"symbol": perp, "limit": 1},
                timeout=12,
            )
            r.raise_for_status()
            j = r.json()
            if isinstance(j, list) and j:
                return float(j[-1]["fundingRate"])
        except Exception:
            continue
    return float("nan")

def fetch_spot_last_binance(perp: str) -> float:
    r = requests.get(
        "https://api.binance.com/api/v3/ticker/price",
        params={"symbol": perp},
        timeout=10,
    )
    r.raise_for_status()
    return float(r.json()["price"])

def fetch_spot_last_kraken(sym: str) -> float:
    kp = KRAKEN_PAIR.get(sym)
    r = requests.get("https://api.kraken.com/0/public/Ticker", params={"pair": kp}, timeout=10)
    r.raise_for_status()
    data = r.json()["result"]
    key = next(iter(data))
    return float(data[key]["c"][0])

def fetch_spot_last(sym: str) -> float:
    perp = sym_to_perp(sym)
    try:
        return fetch_spot_last_binance(perp)
    except Exception:
        try:
            return fetch_spot_last_kraken(sym)
        except Exception:
            return float("nan")

# ------------- INDICATORS & SCORE -------------
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
    m = s.rolling(n).mean()
    v = s.rolling(n).std(ddof=0)
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
    vol_regime = ((a / df["close"]) > (a / df["close"]).rolling(90).median()).fillna(False)
    raw = 0.3*trend + 0.3*breakout + 0.2*vol_conf + 0.2*vol_regime
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
    if math.isnan(fr): return "–"
    if bias == "🟢 LONG" and fr <= 0: return "✅ aligned"
    if bias == "🔴 SHORT" and fr >= 0: return "✅ aligned"
    if bias == "⚪ NEUTRAL": return "•"
    return "⚠️"

# ------------- STATE (JSON on disk) -------------
def load_state(path: str) -> dict:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def save_state(path: str, state: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    os.replace(tmp, path)

# ------------- MAIN LOGIC -------------
def main():
    parser = argparse.ArgumentParser(description="Momentum alert job")
    parser.add_argument("--assets", type=str, default="BTC/USDT,ETH/USDT",
                        help="Comma-separated assets, e.g. 'BTC/USDT,ETH/USDT,BNB/USDT'")
    parser.add_argument("--timeframes", type=str, default="5m,1h,4h",
                        help="Comma-separated TFs, e.g. '5m,1h,4h'")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--entry_long", type=int, default=60)
    parser.add_argument("--entry_short", type=int, default=40)
    parser.add_argument("--exit_warn", type=int, default=55)
    parser.add_argument("--exit_hard", type=int, default=45)
    parser.add_argument("--min_weight", type=float, default=0.30,
                        help="Min weight to alert on bias change")
    parser.add_argument("--cooldown", type=int, default=15,
                        help="Cooldown minutes per asset|tf")
    args = parser.parse_args()

    assets = [a.strip() for a in args.assets.split(",") if a.strip()]
    tfs = [t.strip() for t in args.timeframes.split(",") if t.strip()]

    TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    TG_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    STATE_PATH = os.getenv("STATE_PATH", "state/prev_bias.json")

    print(f"[INFO] Running SJ-v5 for {assets} @ {tfs}; state: {STATE_PATH}")

    state = load_state(STATE_PATH)  # dict: key -> {bias, score, weight, last_alert_epoch}

    now_epoch = time.time()
    alerts_sent = 0

    for sym in assets:
        perp = sym_to_perp(sym)
        fr = fetch_funding(perp)
        live_price = fetch_spot_last(sym)

        for tf in tfs:
            key = f"{sym}|{tf}"
            try:
                df = fetch_ohlcv(sym, tf, args.limit)
                sc = compute_score(df)
                s = float(sc.iloc[-1])
                last_idx = sc.index[-1]
                price_fallback = float(df["close"].iloc[-1])
                price = live_price if not np.isnan(live_price) else price_fallback

                bias, w = bias_and_weight(s, args.entry_long, args.entry_short)
                align = funding_alignment(bias, fr)

                prev = state.get(key, {})
                prev_bias = prev.get("bias")
                last_alert_epoch = float(prev.get("last_alert_epoch", 0))
                cooldown_ok = (now_epoch - last_alert_epoch) >= args.cooldown * 60

                # 1) Bias flip alert (primary)
                if prev_bias is not None and bias != prev_bias and w >= args.min_weight and cooldown_ok:
                    msg = (
                        f"⚡ <b>Momentum flip</b> on <b>{sym}</b> ({tf})\n"
                        f"{prev_bias} → <b>{bias}</b>\n"
                        f"Weight: <b>{w:.2f}</b> | Score: <b>{s:.1f}</b>\n"
                        f"Price: <b>{price:.2f}</b>\n"
                        f"Funding: <b>{'-' if math.isnan(fr) else f'{fr*100:.3f}%'}</b> ({align})\n"
                        f"Time: {last_idx.tz_convert(BERLIN).strftime('%Y-%m-%d %H:%M')}"
                    )
                    if send_telegram(TG_TOKEN, TG_CHAT_ID, msg):
                        alerts_sent += 1
                        last_alert_epoch = now_epoch

                # 2) Exit warnings / hard exit (stateful, symmetric)
                # If you were LONG and momentum deteriorates…
                if prev_bias == "🟢 LONG":
                    if s < args.exit_hard and cooldown_ok:
                        msg = (
                            f"🚪 <b>Hard exit (LONG weakening)</b> — {sym} ({tf})\n"
                            f"Score < {args.exit_hard} → {s:.1f} | Price {price:.2f}"
                        )
                        if send_telegram(TG_TOKEN, TG_CHAT_ID, msg):
                            alerts_sent += 1
                            last_alert_epoch = now_epoch
                    elif s < args.exit_warn and cooldown_ok:
                        msg = (
                            f"⚠️ <b>Exit warning (LONG)</b> — {sym} ({tf})\n"
                            f"Score < {args.exit_warn} → {s:.1f} | Price {price:.2f}"
                        )
                        if send_telegram(TG_TOKEN, TG_CHAT_ID, msg):
                            alerts_sent += 1
                            last_alert_epoch = now_epoch

                # If you were SHORT and momentum deteriorates for shorts (mirror thresholds)
                if prev_bias == "🔴 SHORT":
                    mirror_warn = 100 - args.exit_warn
                    mirror_hard = 100 - args.exit_hard
                    if s > mirror_hard and cooldown_ok:
                        msg = (
                            f"🚪 <b>Hard exit (SHORT weakening)</b> — {sym} ({tf})\n"
                            f"Score > {mirror_hard} → {s:.1f} | Price {price:.2f}"
                        )
                        if send_telegram(TG_TOKEN, TG_CHAT_ID, msg):
                            alerts_sent += 1
                            last_alert_epoch = now_epoch
                    elif s > mirror_warn and cooldown_ok:
                        msg = (
                            f"⚠️ <b>Exit warning (SHORT)</b> — {sym} ({tf})\n"
                            f"Score > {mirror_warn} → {s:.1f} | Price {price:.2f}"
                        )
                        if send_telegram(TG_TOKEN, TG_CHAT_ID, msg):
                            alerts_sent += 1
                            last_alert_epoch = now_epoch

                # Update state
                state[key] = {
                    "bias": bias, "score": round(s, 2), "weight": w,
                    "last_alert_epoch": last_alert_epoch,
                    "last_seen_epoch": now_epoch,
                }

                print(f"[OK] {sym} {tf} price={price:.2f} score={s:.1f} bias={bias} w={w} fr={fr if not math.isnan(fr) else 'nan'}")

                # polite pacing for APIs
                time.sleep(0.2)

            except Exception as e:
                print(f"[ERR] {sym} {tf}: {e}")
                # keep previous state untouched on error; continue
                continue

    # Persist state
    save_state(STATE_PATH, state)

    ts = datetime.utcnow().replace(tzinfo=timezone.utc).astimezone(BERLIN).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[DONE] {alerts_sent} alerts sent • {ts} Berlin")

if __name__ == "__main__":
    main()
