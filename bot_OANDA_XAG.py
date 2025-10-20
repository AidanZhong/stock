# -*- coding: utf-8 -*-
"""
Created on 2025/10/19 21:56

@author: Aidan
@project: stock
@filename: bot_OANDA
"""
# ===========================================
# OANDA v20 Real-time XAU/USD Bot
# Strategy: RSI + Pullback
# ===========================================

import time, requests, pandas as pd, numpy as np, json, os, datetime

import matplotlib
from matplotlib import pyplot as plt

from notifier import send_telegram_message

matplotlib.use("TkAgg")
# --- CONFIG ---
OANDA_URL = "https://api-fxpractice.oanda.com/v3"
ACCOUNT_ID = "101-004-37405580-001"
API_KEY = "6cae6190e1ce41e05724528b6dc3179e-ed6d4e7bcaaa1b3f8f8f6cac933b7bc6"

INSTRUMENT = "XAG_USD"  # Gold spot
GRANULARITY = "M1"  # 1-minute candles
CANDLE_COUNT = 500

RSI_PERIOD = 14
RSI_ENTRY_THRESHOLD = 30
RSI_ADD_THRESHOLD = 40
EMA_FAST = 20
EMA_TREND = 200
ATR_PERIOD = 14
SWING_LOOKBACK = 10
RISK_REWARD = 2.0
ATR_SL_MULTIPLIER = 1.5

STATE_FILE = "state_oanda.json"
POLL_SECONDS = 5
MAX_ADDS = 3


# --- HELPERS ---
def fetch_oanda_candles():
    url = f"{OANDA_URL}/instruments/{INSTRUMENT}/candles"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    params = {"count": CANDLE_COUNT, "granularity": GRANULARITY, "price": "M"}
    r = requests.get(url, headers=headers, params=params, timeout=10)
    data = r.json()
    if "candles" not in data:
        raise RuntimeError(f"Bad response: {data}")
    records = [
        (c["time"], float(c["mid"]["o"]), float(c["mid"]["h"]),
         float(c["mid"]["l"]), float(c["mid"]["c"]))
        for c in data["candles"] if c["complete"]
    ]
    df = pd.DataFrame(records, columns=["time", "open", "high", "low", "close"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df.set_index("time", inplace=True)
    return df


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


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


def compute_levels(entry, df):
    recent = df.iloc[-SWING_LOOKBACK:]
    swing_low = recent["low"].min()
    atr_val = recent["atr"].iloc[-1]
    sl = max(swing_low, entry - ATR_SL_MULTIPLIER * atr_val)
    tp = entry + RISK_REWARD * (entry - sl)
    return round(tp, 2), round(sl, 2)


def generate_signals(df):
    last, prev = df.iloc[-1], df.iloc[-2]

    # === LONG SETUPS ===
    in_uptrend = last["close"] > last["ema_trend"]
    crossed_up = prev["rsi"] <= RSI_ENTRY_THRESHOLD and last["rsi"] > RSI_ENTRY_THRESHOLD
    if in_uptrend and crossed_up:
        tp, sl = compute_levels(last["close"], df)
        return {"action": "buy", "price": last["close"], "tp": tp, "sl": sl,
                "why": f"RSI cross↑{RSI_ENTRY_THRESHOLD} + price above EMA200"}

    near_ema = abs(last["close"] - last["ema_fast"]) / last["ema_fast"] < 0.002
    rsi_bounce = prev["rsi"] <= RSI_ADD_THRESHOLD and last["rsi"] > RSI_ADD_THRESHOLD
    if in_uptrend and near_ema and rsi_bounce:
        tp, sl = compute_levels(last["close"], df)
        return {"action": "add", "price": last["close"], "tp": tp, "sl": sl,
                "why": f"Pullback near EMA20 + RSI bounce↑{RSI_ADD_THRESHOLD}"}

    # === SHORT SETUPS ===
    in_downtrend = last["close"] < last["ema_trend"]
    crossed_down = prev["rsi"] >= 70 and last["rsi"] < 70
    if in_downtrend and crossed_down:
        # mirror of compute_levels for short
        recent = df.iloc[-SWING_LOOKBACK:]
        swing_high = recent["high"].max()
        atr_val = recent["atr"].iloc[-1]
        tp = last["close"] - RISK_REWARD * (swing_high - last["close"])
        sl = min(swing_high, last["close"] + ATR_SL_MULTIPLIER * atr_val)
        return {"action": "sell", "price": last["close"], "tp": round(tp, 2), "sl": round(sl, 2),
                "why": f"RSI cross↓70 + price below EMA200"}

    near_ema_short = abs(last["close"] - last["ema_fast"]) / last["ema_fast"] < 0.002
    rsi_fall = prev["rsi"] >= 60 and last["rsi"] < 60
    if in_downtrend and near_ema_short and rsi_fall:
        recent = df.iloc[-SWING_LOOKBACK:]
        swing_high = recent["high"].max()
        atr_val = recent["atr"].iloc[-1]
        tp = last["close"] - RISK_REWARD * (swing_high - last["close"])
        sl = min(swing_high, last["close"] + ATR_SL_MULTIPLIER * atr_val)
        return {"action": "add_short", "price": last["close"], "tp": round(tp, 2), "sl": round(sl, 2),
                "why": f"Pullback near EMA20 + RSI cross↓60"}

    return {"action": None}


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            return json.load(open(STATE_FILE))
        except:
            pass
    return {"position": None, "last_signal_ts": None}


def save_state(state):
    json.dump(state, open(STATE_FILE, "w"), indent=2)


def find_historical_signals(df):
    """Scan through past candles and record every Buy/Add opportunity."""
    entries = []
    for i in range(2, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]

        # Trend filter
        in_uptrend = curr["close"] > curr["ema_trend"]

        # --- BUY signal (RSI crosses up 30) ---
        crossed_up = prev["rsi"] <= RSI_ENTRY_THRESHOLD and curr["rsi"] > RSI_ENTRY_THRESHOLD
        if in_uptrend and crossed_up:
            tp, sl = compute_levels(curr["close"], df.iloc[:i])
            entries.append({
                "time": df.index[i],
                "price": curr["close"],
                "tp": tp,
                "sl": sl,
                "type": "buy"
            })
            continue

        # --- ADD signal (RSI crosses up 40 near EMA20) ---
        near_ema = abs(curr["close"] - curr["ema_fast"]) / curr["ema_fast"] < 0.002
        rsi_bounce = prev["rsi"] <= RSI_ADD_THRESHOLD and curr["rsi"] > RSI_ADD_THRESHOLD
        if in_uptrend and near_ema and rsi_bounce:
            tp, sl = compute_levels(curr["close"], df.iloc[:i])
            entries.append({
                "time": df.index[i],
                "price": curr["close"],
                "tp": tp,
                "sl": sl,
                "type": "add"
            })
    return entries


def plot_historical_signals(df):
    """Draw chart with all detected Buy/Add signals."""
    entries = find_historical_signals(df)

    plt.figure(figsize=(12, 8))
    plt.title(f"{INSTRUMENT} — Historical Entry Opportunities", fontsize=14)

    # --- Price + EMAs ---
    plt.plot(df.index, df["close"], color="black", label="Close")
    plt.plot(df.index, df["ema_fast"], color="blue", label="EMA20")
    plt.plot(df.index, df["ema_trend"], color="orange", label="EMA200")

    # --- Mark entries with TP/SL ---
    for e in entries:
        color = "green" if e["type"] == "buy" else "cyan"
        label = "BUY" if e["type"] == "buy" else "ADD"
        plt.scatter(e["time"], e["price"], color=color, marker="^", s=100, label=label)
        plt.axhline(e["tp"], color="lime", linestyle="--", linewidth=0.8, alpha=0.7)
        plt.axhline(e["sl"], color="red", linestyle="--", linewidth=0.8, alpha=0.7)
        plt.text(df.index[-1], e["tp"], f"TP {e['tp']}", color="lime", fontsize=8, va="bottom")
        plt.text(df.index[-1], e["sl"], f"SL {e['sl']}", color="red", fontsize=8, va="top")

    plt.legend(loc="upper left")
    plt.grid(True)
    plt.show()


# --- MAIN LOOP ---
# keep this list outside the loop (global to persist markers)
entries = []


def run_loop():
    state = load_state()
    print(f"🤖 OANDA bot started on {INSTRUMENT}")

    plt.ion()
    fig, (ax_price, ax_rsi, ax_atr) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"{INSTRUMENT} — Live RSI + Pullback Strategy", fontsize=14)
    while True:
        try:
            df = add_indicators(fetch_oanda_candles())
            latest_time = df.index[-1]
            latest = df.iloc[-1]
            delay = datetime.datetime.now(datetime.timezone.utc) - latest_time.to_pydatetime()

            print("=" * 80)
            print(f"[{latest_time}] {INSTRUMENT}")
            print(f"   Price: {latest['close']:.2f} (delay {delay.total_seconds():.1f}s)")
            print(f"   EMA20: {latest['ema_fast']:.2f}  EMA200: {latest['ema_trend']:.2f}")
            print(f"   RSI: {latest['rsi']:.2f}  ATR: {latest['atr']:.2f}")

            sig = generate_signals(df)
            ts = str(latest_time)
            if ts != state.get("last_signal_ts") and sig["action"]:
                state["last_signal_ts"] = ts
                state["position"] = sig
                save_state(state)
                message = f"🚀 SIGNAL: {sig['action'].upper()} at {sig['price']:.2f} | TP {sig['tp']} | SL {sig['sl']} \n Reason: {sig['why']}"
                print(message)
                # store for chart
                entries.append({
                    "time": latest_time,
                    "price": sig["price"],
                    "tp": sig["tp"],
                    "sl": sig["sl"],
                    "type": sig["action"]
                })
                send_telegram_message(message)
            else:
                print("No new signal.")

        except Exception as e:
            print("❌ Error:", e)

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    print("📈 Generating historical signal chart...")
    df = add_indicators(fetch_oanda_candles())
    plot_historical_signals(df)

    print("✅ Chart done. Starting live monitoring...")
    run_loop()
