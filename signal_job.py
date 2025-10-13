import os, argparse, time, requests
import pandas as pd
import numpy as np

# --- Binance via ccxt (nur Public-Reads) ---
import ccxt

# --------- Indikatoren ---------
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def donchian(df, n=20):
    hi = df['high'].rolling(n, min_periods=n).max()
    lo = df['low' ].rolling(n, min_periods=n).min()
    return hi, lo

def atr(df, n=20):
    pc = df['close'].shift(1)
    tr = pd.concat([
        (df['high']-df['low']).abs(),
        (df['high']-pc).abs(),
        (df['low'] -pc).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()

def zscore(s, n=90):
    m = s.rolling(n).mean()
    v = s.rolling(n).std(ddof=0)
    return (s-m)/v.replace(0, np.nan)

# --------- Funding (Binance Futures public) ---------
def fetch_funding_rate_binance(symbol_perp: str):
    """
    symbol_perp z.B. 'BTCUSDT' oder 'ETHUSDT'
    """
    ex = ccxt.binance()
    # ccxt raw endpoint for fapi public funding rate:
    data = ex.fapiPublicGetFundingRate({'symbol': symbol_perp, 'limit': 1})
    # data ist Liste mit dicts; nimm die letzte
    fr = float(data[-1]['fundingRate'])
    return fr  # z.B. 0.0001 = 0.01%

# --------- Daten ---------
def fetch_ohlcv_binance_spot(symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
    ex = ccxt.binance()
    raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=['ts','open','high','low','close','volume'])
    df['time'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    df = df.set_index('time')[['open','high','low','close','volume']].astype(float)
    return df

def compute_score(df: pd.DataFrame) -> pd.Series:
    # Komponenten: Trend (EMA50>EMA200), Breakout (Donchian20), Volumen-Z, Vol-Regime (ATR/Price > Median)
    ema50 = ema(df['close'], 50)
    ema200 = ema(df['close'], 200)
    trend_up = (df['close'] > ema50) & (ema50 > ema200)

    dc_hi, dc_lo = donchian(df, 20)
    breakout_up   = df['close'] > dc_hi.shift(1)
    breakout_down = df['close'] < dc_lo.shift(1)

    volZ = zscore(df['volume'], 90).fillna(0)
    vol_confirm = volZ > 0.5

    a = atr(df, 20)
    vol_regime = ((a/df['close']) > (a/df['close']).rolling(90).median()).fillna(False)

    # simple gewichtete Mischung -> 0..100
    raw_long = (0.30*trend_up.astype(float) +
                0.30*breakout_up.astype(float) +
                0.20*vol_confirm.astype(float) +
                0.20*vol_regime.astype(float))
    # für MVP genügt long-orientierter Score; Exits/Shorts kommen über Schwellen/Flip
    score = (100*raw_long.clip(0,1)).fillna(0)
    return score

def bias_weight(score, long_th=60, short_th=40):
    if score >= long_th:
        w = min(1.0, max(0.0, (score - long_th)/(100-long_th+1e-9)))
        return "LONG", round(w,2)
    if score <= short_th:
        w = min(1.0, max(0.0, (short_th - score)/(short_th+1e-9)))
        return "SHORT", round(w,2)
    return "NEUTRAL", 0.0

# --------- Alerts ---------
def send_telegram(msg: str, token: str, chat_id: str):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    r = requests.post(url, json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"})
    if not r.ok:
        print("Telegram error:", r.status_code, r.text)

# --------- Main ---------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default="BTC/USDT,ETH/USDT")
    ap.add_argument("--timeframes", default="5m,1h,4h")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--entry_long", type=float, default=60.0)
    ap.add_argument("--entry_short", type=float, default=40.0)
    ap.add_argument("--exit_warn", type=float, default=55.0)
    ap.add_argument("--exit_hard", type=float, default=45.0)
    args = ap.parse_args()

    token = os.getenv("TELEGRAM_BOT_TOKEN","")
    chat_id = os.getenv("TELEGRAM_CHAT_ID","")
    if not token or not chat_id:
        print("Missing TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID (add in GitHub Secrets).")
        return

    assets = [a.strip() for a in args.assets.split(",") if a.strip()]
    tfs    = [t.strip() for t in args.timeframes.split(",") if t.strip()]

    for sym in assets:
        # Perp-Symbol für Funding bauen (BTC/USDT -> BTCUSDT)
        perp = sym.replace("/", "")
        for tf in tfs:
            try:
                # 1) Daten + Score
                df = fetch_ohlcv_binance_spot(sym, tf, args.limit)
                sc = compute_score(df)
                last = float(sc.iloc[-1])
                prev = float(sc.iloc[-2]) if len(sc) > 1 else last
                price = float(df['close'].iloc[-1])

                # 2) Funding
                fr = fetch_funding_rate_binance(perp)  # z.B. 0.0001 = 0.01%
                # Filterlogik:
                funding_ok_long  = fr <= 0.0          # günstiger/neutral
                funding_ok_short = fr >= 0.0          # neutral/positiv -> Short begünstigt
                funding_note = f"{fr*100:.3f}%"

                # 3) Entry Crossings
                crossed_up   = (prev < args.entry_long) and (last >= args.entry_long)
                crossed_down = (prev > args.entry_short) and (last <= args.entry_short)

                if crossed_up and funding_ok_long:
                    msg = (f"🟢 *{sym}* • {tf}\n"
                           f"Momentum {prev:.0f} → *{last:.0f}*  (cross_up {args.entry_long:.0f})\n"
                           f"Funding: {funding_note}  ✅\n"
                           f"Preis: {price:.2f}")
                    send_telegram(msg, token, chat_id)

                if crossed_down and funding_ok_short:
                    msg = (f"🔴 *{sym}* • {tf}\n"
                           f"Momentum {prev:.0f} → *{last:.0f}*  (cross_down {args.entry_short:.0f})\n"
                           f"Funding: {funding_note}  ✅\n"
                           f"Preis: {price:.2f}")
                    send_telegram(msg, token, chat_id)

                # 4) Exit-Alerts (Score-Flip)
                # Warn-Exit: Long verliert Momentum
                if (prev >= args.exit_warn) and (last < args.exit_warn):
                    send_telegram(f"⚠️ *{sym}* • {tf}\nExit-Warnung (Score < {args.exit_warn:.0f})\nScore: {last:.0f}  • Funding: {funding_note}\nPreis: {price:.2f}",
                                  token, chat_id)
                # Hard-Exit
                if (prev >= args.exit_hard) and (last < args.exit_hard):
                    send_telegram(f"🚪 *{sym}* • {tf}\nExit (Score < {args.exit_hard:.0f})\nScore: {last:.0f}  • Funding: {funding_note}\nPreis: {price:.2f}",
                                  token, chat_id)

                # Optional: Short-Exit (wenn du Short nutzt): Flip zurück >45/55 etc.

            except Exception as e:
                print("Error", sym, tf, e)
                continue

if __name__ == "__main__":
    main()
