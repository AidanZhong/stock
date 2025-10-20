# bot_template.py
# Multi-instrument OANDA bot for ONE timeframe (TAG), with multi-trade handling
# and per-timeframe stats CSV: data/<TAG>/stats_<TAG>.csv

import os, sys, time, requests, pandas as pd, numpy as np, datetime as dt, uuid
from notifier import send_telegram_message

# --------- PARAMETERS PASSED FROM RUNNER ----------
GRANULARITY = sys.argv[1] if len(sys.argv) > 1 else "M1"   # e.g. S5, M1, M5
TAG = GRANULARITY                                          # label used in logs/paths

# --------- OANDA + INSTRUMENTS ----------
OANDA_URL  = "https://api-fxpractice.oanda.com/v3"
ACCOUNT_ID = "101-004-37405580-001"
API_KEY = "6cae6190e1ce41e05724528b6dc3179e-ed6d4e7bcaaa1b3f8f8f6cac933b7bc6"
INSTRUMENTS = ["XAU_USD", "XAG_USD", "UK100_GBP"]          # edit as needed

# --------- DIRECTORIES & FILES ----------
BASE_DIR     = "data"
DATA_DIR     = os.path.join(BASE_DIR, TAG)
os.makedirs(DATA_DIR, exist_ok=True)
OPEN_FILE    = os.path.join(DATA_DIR, f"open_trades_{TAG}.csv")
CLOSED_FILE  = os.path.join(DATA_DIR, f"closed_trades_{TAG}.csv")
STATS_FILE_TF= os.path.join(DATA_DIR, f"stats_{TAG}.csv")   # <-- per-timeframe stats file

# --------- STRATEGY ----------
CANDLE_COUNT = 1200
POLL_SECONDS = 1
RSI_PERIOD = 14
RSI_ENTRY_THRESHOLD = 30
RSI_ADD_THRESHOLD = 40
EMA_FAST = 20
EMA_TREND = 200
ATR_PERIOD = 14
SWING_LOOKBACK = 10
RISK_REWARD = 2.0
ATR_SL_MULTIPLIER = 1.5

# =================== INDICATORS ===================
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi(series, n=14):
    d  = series.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / (dn + 1e-12)
    return 100 - (100 / (1 + rs))

def atr(h, l, c, n=14):
    pc = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def add_indicators(df):
    df["ema_fast"]  = ema(df["close"], EMA_FAST)
    df["ema_trend"] = ema(df["close"], EMA_TREND)
    df["rsi"]       = rsi(df["close"], RSI_PERIOD)
    df["atr"]       = atr(df["high"], df["low"], df["close"], ATR_PERIOD)
    return df

# =================== DATA ===================
def fetch_oanda(symbol):
    url = f"{OANDA_URL}/instruments/{symbol}/candles"
    h   = {"Authorization": f"Bearer {API_KEY}"}
    p   = {"count": CANDLE_COUNT, "granularity": GRANULARITY, "price": "M"}
    r   = requests.get(url, headers=h, params=p, timeout=10)
    d   = r.json()
    if "candles" not in d:
        raise RuntimeError(f"Bad response for {symbol} [{TAG}]: {d}")
    rec = [(c["time"], float(c["mid"]["o"]), float(c["mid"]["h"]),
            float(c["mid"]["l"]), float(c["mid"]["c"]))
           for c in d["candles"] if c.get("complete", False)]
    df = pd.DataFrame(rec, columns=["time","open","high","low","close"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df.set_index("time", inplace=True)
    return df

# =================== SIGNALS ===================
def compute_levels(entry, df, direction="long"):
    recent = df.iloc[-SWING_LOOKBACK:]
    a = recent["atr"].iloc[-1]
    if direction == "long":
        sl = max(recent["low"].min(), entry - ATR_SL_MULTIPLIER * a)
        tp = entry + RISK_REWARD * (entry - sl)
    else:
        sl = min(recent["high"].max(), entry + ATR_SL_MULTIPLIER * a)
        tp = entry - RISK_REWARD * (sl - entry)
    return round(tp,2), round(sl,2)

def generate_signal(df):
    prev, last = df.iloc[-2], df.iloc[-1]
    in_up   = last.close > last.ema_trend
    in_down = last.close < last.ema_trend
    # Longs
    if in_up and prev.rsi <= RSI_ENTRY_THRESHOLD and last.rsi > RSI_ENTRY_THRESHOLD:
        tp, sl = compute_levels(last.close, df, "long")
        return {"type":"BUY", "price":last.close, "tp":tp, "sl":sl,
                "why":f"RSI↑{RSI_ENTRY_THRESHOLD} + price>EMA200"}
    if in_up and abs(last.close-last.ema_fast)/last.ema_fast < 0.002 and prev.rsi <= RSI_ADD_THRESHOLD and last.rsi > RSI_ADD_THRESHOLD:
        tp, sl = compute_levels(last.close, df, "long")
        return {"type":"ADD_LONG", "price":last.close, "tp":tp, "sl":sl,
                "why":f"Pullback≈EMA20 + RSI↑{RSI_ADD_THRESHOLD}"}
    # Shorts
    if in_down and prev.rsi >= 70 and last.rsi < 70:
        tp, sl = compute_levels(last.close, df, "short")
        return {"type":"SELL", "price":last.close, "tp":tp, "sl":sl,
                "why":"RSI↓70 + price<EMA200"}
    if in_down and abs(last.close-last.ema_fast)/last.ema_fast < 0.002 and prev.rsi >= 60 and last.rsi < 60:
        tp, sl = compute_levels(last.close, df, "short")
        return {"type":"ADD_SHORT", "price":last.close, "tp":tp, "sl":sl,
                "why":"Pullback≈EMA20 + RSI↓60"}
    return None

# =================== TRADE TRACKING ===================
# allow multiple open trades per symbol
open_trades = {s: [] for s in INSTRUMENTS}

def _append_csv(path, rowdict):
    pd.DataFrame([rowdict]).to_csv(path, mode="a", index=False, header=not os.path.exists(path))

def record_trade_open(symbol, sig):
    trade = {
        "trade_id": str(uuid.uuid4()),
        "timeframe": TAG,
        "symbol": symbol,
        "type": sig["type"],
        "entry_time": dt.datetime.utcnow().isoformat(),
        "entry_price": float(sig["price"]),
        "tp": float(sig["tp"]),
        "sl": float(sig["sl"]),
        "status": "OPEN"
    }
    open_trades.setdefault(symbol, []).append(trade)
    _append_csv(OPEN_FILE, trade)

def record_trade_close(symbol, trade, outcome, exit_price):
    # remove from memory
    lst = open_trades.get(symbol, [])
    open_trades[symbol] = [t for t in lst if t["trade_id"] != trade["trade_id"]]

    closed = dict(trade)
    closed.update({
        "exit_time": dt.datetime.utcnow().isoformat(),
        "exit_price": float(exit_price),
        "outcome": outcome,
        "status": "CLOSED"
    })
    _append_csv(CLOSED_FILE, closed)
    # update this timeframe's stats file
    update_stats_summary_tag(TAG, symbol)

def check_trade_outcomes_for_symbol(symbol, last_price):
    """Traverse ALL open trades for this symbol; close whichever hits TP/SL first."""
    for trade in list(open_trades.get(symbol, [])):
        ttype = trade["type"]
        long_side  = ttype.startswith("BUY") or ttype.startswith("ADD_LONG")
        short_side = ttype.startswith("SELL") or ttype.startswith("ADD_SHORT")

        if long_side:
            if last_price >= trade["tp"]:
                record_trade_close(symbol, trade, "WIN", last_price)
            elif last_price <= trade["sl"]:
                record_trade_close(symbol, trade, "LOSS", last_price)
        elif short_side:
            if last_price <= trade["tp"]:
                record_trade_close(symbol, trade, "WIN", last_price)
            elif last_price >= trade["sl"]:
                record_trade_close(symbol, trade, "LOSS", last_price)

# =================== PER-TIMEFRAME STATS ===================
def compute_win_stats_for_symbol(symbol):
    """Compute total/wins/losses/winrate for THIS timeframe (TAG) only, per symbol."""
    if not os.path.exists(CLOSED_FILE): return 0,0,0,0.0
    df = pd.read_csv(CLOSED_FILE)
    if df.empty: return 0,0,0,0.0
    # ensure we filter to this timeframe TAG (in case files are reused)
    if "timeframe" in df.columns:
        sdf = df[(df["timeframe"] == TAG) & (df["symbol"] == symbol)]
    else:
        sdf = df[df["symbol"] == symbol]
    total  = len(sdf)
    if total == 0: return 0,0,0,0.0
    wins   = (sdf["outcome"] == "WIN").sum()
    losses = (sdf["outcome"] == "LOSS").sum()
    rate   = round(100.0 * wins / total, 2)
    return total, wins, losses, rate

def update_stats_summary_tag(timeframe, symbol):
    """Upsert stats row for (timeframe, symbol) into data/<TAG>/stats_<TAG>.csv."""
    total, wins, losses, rate = compute_win_stats_for_symbol(symbol)
    row = {
        "timeframe": timeframe,
        "symbol": symbol,
        "total": total,
        "wins": wins,
        "losses": losses,
        "winrate": rate,
        "last_updated": dt.datetime.utcnow().isoformat()
    }
    if os.path.exists(STATS_FILE_TF):
        df = pd.read_csv(STATS_FILE_TF)
        mask = (df["timeframe"] == timeframe) & (df["symbol"] == symbol)
        if mask.any():
            df.loc[mask, ["total","wins","losses","winrate","last_updated"]] = \
                [total, wins, losses, rate, row["last_updated"]]
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(STATS_FILE_TF, index=False)
    else:
        pd.DataFrame([row]).to_csv(STATS_FILE_TF, index=False)

# =================== MAIN LOOP ===================
def run_bot():
    print(f"🤖 Starting OANDA bot [{TAG}] with {GRANULARITY} candles — monitoring {len(INSTRUMENTS)} instruments")
    last_signal_time = {s: None for s in INSTRUMENTS}

    while True:
        try:
            for sym in INSTRUMENTS:
                df = add_indicators(fetch_oanda(sym))
                last, t = df.iloc[-1], df.index[-1]

                # 1) Check ALL open trades for this symbol (may close multiple)
                check_trade_outcomes_for_symbol(sym, float(last.close))

                # 2) Generate possible new signal
                sig = generate_signal(df)

                # 3) Log latest candle (per timeframe/symbol)
                df.tail(1).assign(time=t).to_csv(
                    os.path.join(DATA_DIR, f"{sym}.csv"),
                    mode="a", index=False,
                    header=not os.path.exists(os.path.join(DATA_DIR, f"{sym}.csv"))
                )

                # 4) New signal (de-dup per bar) → alert & record open
                if sig and str(t) != last_signal_time[sym]:
                    last_signal_time[sym] = str(t)
                    send_telegram_message(
                        f"🚀 {sig['type']} on {sym} [{TAG}]\nPrice: {sig['price']:.2f}\nTP:{sig['tp']} | SL:{sig['sl']}\nReason: {sig['why']}"
                    )
                    record_trade_open(sym, sig)

                # 5) Console status
                print(f"[{t.strftime('%H:%M:%S')}] {sym} [{TAG}] {last.close:.2f} RSI={last.rsi:.1f}")

            # Optional: quick snapshot per timeframe
            if os.path.exists(STATS_FILE_TF):
                sdf = pd.read_csv(STATS_FILE_TF)
                # Aggregate across symbols for display only
                total = int(sdf["total"].sum())
                wins  = int(sdf["wins"].sum())
                rate  = round(100.0*wins/max(total,1), 2)
                print(f"📊 [{TAG}] aggregate wins: {wins}/{total} ({rate}%)")

        except Exception as e:
            print("❌ Error:", e)

        time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    run_bot()