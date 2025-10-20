# -*- coding: utf-8 -*-
"""
OANDA Multi-Instrument Bot (5s candles)
RSI + Pullback Strategy with Win Rate Tracking (No Telegram trade results)
Author: Aidan
"""
import os, time, json, requests, pandas as pd, numpy as np, datetime, platform, matplotlib
from matplotlib import pyplot as plt
from notifier import send_telegram_message

# -------- Cross-platform backend fix --------
if platform.system() == "Darwin":
    try: matplotlib.use("Qt5Agg")
    except: matplotlib.use("MacOSX")
else:
    matplotlib.use("TkAgg")

# ================== CONFIG ==================
OANDA_URL = "https://api-fxpractice.oanda.com/v3"
ACCOUNT_ID = "101-004-37405580-001"
API_KEY = "6cae6190e1ce41e05724528b6dc3179e-ed6d4e7bcaaa1b3f8f8f6cac933b7bc6"
# Instruments to monitor
INSTRUMENTS = [
    "XAU_USD",  # Gold
    "XAG_USD",  # Silver
    "UK100_GBP"  # FTSE 100
]

GRANULARITY = "S5"      # 5-second candles
CANDLE_COUNT = 1200
POLL_SECONDS = 1

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# --- Strategy parameters ---
RSI_PERIOD = 14
RSI_ENTRY_THRESHOLD = 30
RSI_ADD_THRESHOLD = 40
EMA_FAST = 20
EMA_TREND = 200
ATR_PERIOD = 14
SWING_LOOKBACK = 10
RISK_REWARD = 2.0
ATR_SL_MULTIPLIER = 1.5
# ===============================================================

# ---------------- INDICATORS ----------------
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi(series, n=14):
    d = series.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    down = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / (down + 1e-12)
    return 100 - (100 / (1 + rs))

def atr(h, l, c, n=14):
    prev_c = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def add_indicators(df):
    df["ema_fast"] = ema(df["close"], EMA_FAST)
    df["ema_trend"] = ema(df["close"], EMA_TREND)
    df["rsi"] = rsi(df["close"], RSI_PERIOD)
    df["atr"] = atr(df["high"], df["low"], df["close"], ATR_PERIOD)
    return df

# ---------------- FETCH DATA ----------------
def fetch_oanda_candles(symbol):
    url = f"{OANDA_URL}/instruments/{symbol}/candles"
    h = {"Authorization": f"Bearer {API_KEY}"}
    p = {"count": CANDLE_COUNT, "granularity": GRANULARITY, "price": "M"}
    r = requests.get(url, headers=h, params=p, timeout=10)
    d = r.json()
    if "candles" not in d:
        raise RuntimeError(f"Bad response for {symbol}: {d}")
    rec = [(c["time"], float(c["mid"]["o"]), float(c["mid"]["h"]),
            float(c["mid"]["l"]), float(c["mid"]["c"]))
           for c in d["candles"] if c["complete"]]
    df = pd.DataFrame(rec, columns=["time","open","high","low","close"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df.set_index("time", inplace=True)
    return df

# ---------------- SIGNAL LOGIC ----------------
def compute_levels(entry, df, direction="long"):
    r = df.iloc[-SWING_LOOKBACK:]
    a = r["atr"].iloc[-1]
    if direction == "long":
        sl = max(r["low"].min(), entry - ATR_SL_MULTIPLIER * a)
        tp = entry + RISK_REWARD * (entry - sl)
    else:
        sl = min(r["high"].max(), entry + ATR_SL_MULTIPLIER * a)
        tp = entry - RISK_REWARD * (sl - entry)
    return round(tp,2), round(sl,2)

def generate_signal(df):
    prev, last = df.iloc[-2], df.iloc[-1]
    in_up = last.close > last.ema_trend
    in_down = last.close < last.ema_trend
    # Long signals
    if in_up and prev.rsi <= RSI_ENTRY_THRESHOLD and last.rsi > RSI_ENTRY_THRESHOLD:
        tp, sl = compute_levels(last.close, df, "long")
        return {"type":"BUY","price":last.close,"tp":tp,"sl":sl,"why":f"RSI↑{RSI_ENTRY_THRESHOLD}"}
    if in_up and abs(last.close-last.ema_fast)/last.ema_fast<0.002 and prev.rsi<=RSI_ADD_THRESHOLD and last.rsi>RSI_ADD_THRESHOLD:
        tp, sl = compute_levels(last.close, df, "long")
        return {"type":"ADD_LONG","price":last.close,"tp":tp,"sl":sl,"why":"RSI↑40 near EMA20"}
    # Short signals
    if in_down and prev.rsi>=70 and last.rsi<70:
        tp, sl = compute_levels(last.close, df, "short")
        return {"type":"SELL","price":last.close,"tp":tp,"sl":sl,"why":"RSI↓70"}
    if in_down and abs(last.close-last.ema_fast)/last.ema_fast<0.002 and prev.rsi>=60 and last.rsi<60:
        tp, sl = compute_levels(last.close, df, "short")
        return {"type":"ADD_SHORT","price":last.close,"tp":tp,"sl":sl,"why":"RSI↓60 near EMA20"}
    return None

# ---------------- TRADE TRACKING ----------------
open_trades = {}
closed_trades_file = os.path.join(DATA_DIR, "closed_trades.csv")

def record_trade_open(symbol, sig):
    trade = {
        "symbol": symbol,
        "type": sig["type"],
        "entry_time": datetime.datetime.utcnow().isoformat(),
        "entry_price": sig["price"],
        "tp": sig["tp"],
        "sl": sig["sl"],
        "status": "OPEN"
    }
    open_trades[symbol] = trade
    pd.DataFrame([trade]).to_csv(os.path.join(DATA_DIR, "open_trades.csv"), mode="a",
                                 index=False, header=not os.path.exists(os.path.join(DATA_DIR,"open_trades.csv")))
    # Only signal alert — no trade result notifications
    send_telegram_message(f"🚀 {symbol} {sig['type']} @ {sig['price']:.2f}\nTP:{sig['tp']} SL:{sig['sl']}")

def record_trade_close(symbol, outcome, price):
    if symbol not in open_trades: return
    t = open_trades.pop(symbol)
    t["exit_time"] = datetime.datetime.utcnow().isoformat()
    t["exit_price"] = price
    t["outcome"] = outcome
    t["status"] = "CLOSED"
    pd.DataFrame([t]).to_csv(closed_trades_file, mode="a",
                             index=False, header=not os.path.exists(closed_trades_file))
    print(f"✅ Trade closed: {symbol} {t['type']} {outcome}")

def check_trade_outcome(symbol, last_price):
    if symbol not in open_trades: return
    t = open_trades[symbol]
    if t["type"].startswith("BUY") and last_price >= t["tp"]:
        record_trade_close(symbol, "WIN", last_price)
    elif t["type"].startswith("BUY") and last_price <= t["sl"]:
        record_trade_close(symbol, "LOSS", last_price)
    elif t["type"].startswith("SELL") and last_price <= t["tp"]:
        record_trade_close(symbol, "WIN", last_price)
    elif t["type"].startswith("SELL") and last_price >= t["sl"]:
        record_trade_close(symbol, "LOSS", last_price)

def compute_winrate():
    if not os.path.exists(closed_trades_file): return 0,0,0
    df = pd.read_csv(closed_trades_file)
    if df.empty: return 0,0,0
    total = len(df)
    wins = (df["outcome"]=="WIN").sum()
    win_rate = 100 * wins / total
    return total, wins, round(win_rate,2)

# ---------------- MAIN LOOP ----------------
def run_bot():
    print(f"🤖 OANDA Multi-Bot running ({GRANULARITY}) | {len(INSTRUMENTS)} instruments")
    plt.ion()
    fig, axes = plt.subplots(len(INSTRUMENTS), 1, figsize=(10,3*len(INSTRUMENTS)), sharex=False)
    if len(INSTRUMENTS)==1: axes=[axes]
    last_signal_time = {s:None for s in INSTRUMENTS}

    while True:
        try:
            for i,sym in enumerate(INSTRUMENTS):
                df = add_indicators(fetch_oanda_candles(sym))
                last = df.iloc[-1]; t = df.index[-1]
                sig = generate_signal(df)
                check_trade_outcome(sym, last.close)

                print(f"[{t.strftime('%H:%M:%S')}] {sym}  {last.close:.2f}  RSI={last.rsi:.1f}")

                # open trade + alert
                if sig and str(t)!=last_signal_time[sym]:
                    record_trade_open(sym, sig)
                    last_signal_time[sym]=str(t)

                # log latest prices for reference
                df.tail(1).assign(time=t).to_csv(f"{DATA_DIR}/{sym}.csv", mode="a", index=False,
                                                header=not os.path.exists(f"{DATA_DIR}/{sym}.csv"))

                # Plot
                ax = axes[i]; ax.clear()
                ax.plot(df.index, df["close"], color="black")
                ax.plot(df.index, df["ema_fast"], color="blue")
                ax.plot(df.index, df["ema_trend"], color="orange")
                ax.set_title(f"{sym} | {last.close:.2f} | RSI={last.rsi:.1f}")
                ax.grid(True)

            fig.tight_layout(); fig.canvas.draw(); fig.canvas.flush_events()

            total, wins, winrate = compute_winrate()
            print(f"📊 Stats: {wins}/{total} wins ({winrate}%)")

        except Exception as e:
            print("❌ Error:", e)

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    run_bot()