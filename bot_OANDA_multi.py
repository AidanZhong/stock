# -*- coding: utf-8 -*-
"""
OANDA Multi-Instrument Bot (5-second RSI + Pullback)
Author: Aidan
"""
import time, requests, pandas as pd, numpy as np, json, os, datetime, matplotlib
from matplotlib import pyplot as plt
from notifier import send_telegram_message

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

GRANULARITY = "S5"  # 5-second candles
CANDLE_COUNT = 1200
POLL_SECONDS = 1  # update frequency

# Strategy hyperparameters
RSI_PERIOD = 14
RSI_ENTRY_THRESHOLD = 30
RSI_ADD_THRESHOLD = 40
EMA_FAST = 20
EMA_TREND = 200
ATR_PERIOD = 14
SWING_LOOKBACK = 10
RISK_REWARD = 2.0
ATR_SL_MULTIPLIER = 1.5


# ================== CORE ==================

def fetch_oanda_candles(symbol):
    url = f"{OANDA_URL}/instruments/{symbol}/candles"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    params = {"count": CANDLE_COUNT, "granularity": GRANULARITY, "price": "M"}
    r = requests.get(url, headers=headers, params=params, timeout=10)
    data = r.json()
    if "candles" not in data:
        raise RuntimeError(f"Bad response for {symbol}: {data}")
    records = [
        (c["time"], float(c["mid"]["o"]), float(c["mid"]["h"]),
         float(c["mid"]["l"]), float(c["mid"]["c"]))
        for c in data["candles"] if c["complete"]
    ]
    df = pd.DataFrame(records, columns=["time", "open", "high", "low", "close"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df.set_index("time", inplace=True)
    return df


def ema(s, n): return s.ewm(span=n, adjust=False).mean()


def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = up / (down + 1e-12)
    return 100 - (100 / (1 + rs))


def atr(high, low, close, period=14):
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def add_indicators(df):
    df["ema_fast"] = ema(df["close"], EMA_FAST)
    df["ema_trend"] = ema(df["close"], EMA_TREND)
    df["rsi"] = rsi(df["close"], RSI_PERIOD)
    df["atr"] = atr(df["high"], df["low"], df["close"], ATR_PERIOD)
    return df


def compute_levels(entry, df, direction="long"):
    recent = df.iloc[-SWING_LOOKBACK:]
    atr_val = recent["atr"].iloc[-1]
    if direction == "long":
        swing_low = recent["low"].min()
        sl = max(swing_low, entry - ATR_SL_MULTIPLIER * atr_val)
        tp = entry + RISK_REWARD * (entry - sl)
    else:
        swing_high = recent["high"].max()
        sl = min(swing_high, entry + ATR_SL_MULTIPLIER * atr_val)
        tp = entry - RISK_REWARD * (sl - entry)
    return round(tp, 2), round(sl, 2)


def generate_signals(df):
    last, prev = df.iloc[-1], df.iloc[-2]
    # long setup
    in_up = last.close > last.ema_trend
    if in_up and prev.rsi <= RSI_ENTRY_THRESHOLD and last.rsi > RSI_ENTRY_THRESHOLD:
        tp, sl = compute_levels(last.close, df, "long")
        return {"action": "BUY", "price": last.close, "tp": tp, "sl": sl,
                "why": f"RSI cross↑{RSI_ENTRY_THRESHOLD} + price > EMA200"}
    if in_up and abs(
            last.close - last.ema_fast) / last.ema_fast < 0.002 and prev.rsi <= RSI_ADD_THRESHOLD and last.rsi > RSI_ADD_THRESHOLD:
        tp, sl = compute_levels(last.close, df, "long")
        return {"action": "ADD_LONG", "price": last.close, "tp": tp, "sl": sl,
                "why": f"Pullback near EMA20 + RSI bounce↑{RSI_ADD_THRESHOLD}"}
    # short setup
    in_down = last.close < last.ema_trend
    if in_down and prev.rsi >= 70 and last.rsi < 70:
        tp, sl = compute_levels(last.close, df, "short")
        return {"action": "SELL", "price": last.close, "tp": tp, "sl": sl,
                "why": "RSI cross↓70 + price < EMA200"}
    if in_down and abs(last.close - last.ema_fast) / last.ema_fast < 0.002 and prev.rsi >= 60 and last.rsi < 60:
        tp, sl = compute_levels(last.close, df, "short")
        return {"action": "ADD_SHORT", "price": last.close, "tp": tp, "sl": sl,
                "why": "Pullback near EMA20 + RSI cross↓60"}
    return None


# ---------- MAIN ----------
def run_multi_bot():
    print(f"🤖 Multi-Instrument OANDA bot started ({GRANULARITY} candles)")
    os.makedirs("data", exist_ok=True)

    plt.ion()
    fig, axes = plt.subplots(len(INSTRUMENTS), 1, figsize=(10, 3 * len(INSTRUMENTS)), sharex=False)
    if len(INSTRUMENTS) == 1:
        axes = [axes]

    state = {inst: None for inst in INSTRUMENTS}

    while True:
        try:
            for i, inst in enumerate(INSTRUMENTS):
                df = add_indicators(fetch_oanda_candles(inst))
                last = df.iloc[-1];
                t = df.index[-1]
                sig = generate_signals(df)
                print(f"[{t.strftime('%H:%M:%S')}] {inst}  {last.close:.2f}  RSI={last.rsi:.1f}")

                # Log to CSV
                df.tail(1).assign(time=t).to_csv(f"data/{inst}.csv", mode="a",
                                                 header=not os.path.exists(f"data/{inst}.csv"))

                # Telegram alert
                if sig and state[inst] != str(t):
                    state[inst] = str(t)
                    msg = (f"⚡ {sig['action']} signal on {inst}\n"
                           f"Price: {sig['price']:.2f}\nTP: {sig['tp']} | SL: {sig['sl']}\nReason: {sig['why']}")
                    print(msg);
                    send_telegram_message(msg)

                # Plotting
                ax = axes[i];
                ax.clear()
                ax.plot(df.index, df["close"], color="black", label="Close")
                ax.plot(df.index, df["ema_fast"], color="blue", label="EMA20")
                ax.plot(df.index, df["ema_trend"], color="orange", label="EMA200")
                ax.set_title(f"{inst} | {last.close:.2f} ({t.strftime('%H:%M:%S')}) | RSI={last.rsi:.1f}")
                ax.legend(loc="upper left");
                ax.grid(True)

            fig.tight_layout();
            fig.canvas.draw();
            fig.canvas.flush_events()

        except Exception as e:
            print("❌ Error:", e)

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    run_multi_bot()
