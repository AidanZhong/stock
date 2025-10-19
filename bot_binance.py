# -*- coding: utf-8 -*-
"""
Created on 2025/10/19 21:50

@author: Aidan
@project: stock
@filename: bot_binance
"""
# ===========================================
# Real-time Binance Trading Bot Example
# Strategy: RSI + Pullback
# ===========================================

import time, requests, pandas as pd, numpy as np, datetime, json, os

import matplotlib
from matplotlib import pyplot as plt

matplotlib.use("TkAgg")
# ---------------- CONFIG ----------------
SYMBOL = "BTCUSDT"  # Binance symbol, e.g. BTCUSDT, ETHUSDT, etc.
INTERVAL = "1m"  # Available: 1m,3m,5m,15m,30m,1h,4h
LIMIT = 500  # Number of candles to request

RSI_PERIOD = 14
RSI_ENTRY_THRESHOLD = 30
RSI_ADD_THRESHOLD = 40
EMA_FAST = 20
EMA_TREND = 200
ATR_PERIOD = 14
SWING_LOOKBACK = 10
RISK_REWARD = 2.0
ATR_SL_MULTIPLIER = 1.5

STATE_FILE = "state.json"
POLL_SECONDS = 5
MAX_ADDS = 3


# ---------------- HELPERS ----------------
def fetch_binance_klines():
    url = f"https://api.binance.com/api/v3/klines?symbol={SYMBOL}&interval={INTERVAL}&limit={LIMIT}"
    data = requests.get(url, timeout=10).json()
    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "qav", "trades", "tb_base", "tb_quote", "ignore"]
    df = pd.DataFrame(data, columns=cols)
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.set_index("time", inplace=True)
    return df[["open", "high", "low", "close", "volume"]]


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = up / (down + 1e-12)
    return 100 - (100 / (1 + rs))


def atr(high, low, close, period=14):
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(),
                    (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
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
    return {"action": None}


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            return json.load(open(STATE_FILE))
        except Exception:
            pass
    return {"position": None, "last_signal_ts": None}


def save_state(state):
    json.dump(state, open(STATE_FILE, "w"), indent=2)


def run_loop():
    state = load_state()
    print(f"🤖 Bot started on {SYMBOL} (Binance live)")
    plt.ion()
    fig, (ax_price, ax_rsi, ax_atr) = plt.subplots(3, 1, figsize=(10, 8), sharex=True,
                                                   gridspec_kw={'height_ratios': [3, 1, 1]})
    fig.canvas.manager.window.attributes('-topmost', True)
    fig.suptitle(f"{SYMBOL} — Live RSI + Pullback Strategy", fontsize=14)
    while True:
        try:
            df = add_indicators(fetch_binance_klines())
            latest_time = df.index[-1]
            latest = df.iloc[-1]
            # 4️⃣ Print full indicator snapshot
            print("=" * 80)
            print(f"[{latest_time}]  {SYMBOL}")
            print(f"   Price: {latest['close']:.2f})")
            print(f"   EMA20: {latest['ema_fast']:.2f}   EMA200: {latest['ema_trend']:.2f}")
            print(f"   RSI({RSI_PERIOD}): {latest['rsi']:.2f}   ATR({ATR_PERIOD}): {latest['atr']:.2f}")
            trend = "⬆️ UP" if latest['close'] > latest['ema_trend'] else "⬇️ DOWN"
            print(f"   Trend Direction: {trend}")
            print("-" * 80)

            sig = generate_signals(df)
            ts = str(latest_time)
            if ts != state.get("last_signal_ts") and sig["action"]:
                state["last_signal_ts"] = ts
                state["position"] = sig
                save_state(state)
                print(
                    f"🚀 {sig['action'].upper()} at {sig['price']:.2f} | TP {sig['tp']} | SL {sig['sl']} ({sig['why']})")

            # --- PLOT SECTION (Live Refresh) ---
            ax_price.clear()
            ax_rsi.clear()
            ax_atr.clear()

            # Plot price + EMAs
            ax_price.plot(df.index, df["close"], label="Close", color="black", linewidth=1)
            ax_price.plot(df.index, df["ema_fast"], label="EMA20", color="blue", linewidth=1)
            ax_price.plot(df.index, df["ema_trend"], label="EMA200", color="orange", linewidth=1)
            ax_price.set_title(f"{SYMBOL} Price with EMA20/200", fontsize=10)
            ax_price.legend(loc="upper left")
            ax_price.grid(True)

            # 💬 --- Add live text label for latest price/time ---
            latest_time_str = latest_time.strftime("%Y-%m-%d %H:%M:%S")
            latest_price = latest["close"]
            text_str = f"Latest: {latest_price:.2f} USD  ({latest_time_str} UTC)"
            ax_price.text(
                0.01, 0.95, text_str,
                transform=ax_price.transAxes,
                fontsize=10, color="green",
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
            )

            # RSI plot
            ax_rsi.plot(df.index, df["rsi"], color="purple", label="RSI(14)")
            ax_rsi.axhline(70, color="red", linestyle="--", linewidth=0.8)
            ax_rsi.axhline(30, color="green", linestyle="--", linewidth=0.8)
            ax_rsi.set_ylim(0, 100)
            ax_rsi.legend(loc="upper left")
            ax_rsi.grid(True)

            # ATR plot
            ax_atr.plot(df.index, df["atr"], color="brown", label="ATR(14)")
            ax_atr.legend(loc="upper left")
            ax_atr.grid(True)

            # 🔄 Refresh the chart
            fig.canvas.draw()
            fig.canvas.flush_events()



        except Exception as e:
            print("Error:", e)

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    run_loop()
