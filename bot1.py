# -*- coding: utf-8 -*-
"""
Created on 2025/10/19 21:19

@author: Aidan
@project: stock
@filename: bot1
"""
# ==============================================
# XAUUSD RSI + Pullback Strategy Bot
# (single-file version)
# ==============================================

import time, json, os, requests
import pandas as pd
import yfinance as yf
import numpy as np

# ---------------- CONFIG ----------------
SYMBOL = "BTC-USD"  # Yahoo symbol for spot gold (USD)
INTERVAL = "1m"
LOOKBACK_DAYS = 2

RSI_PERIOD = 14
RSI_ENTRY_THRESHOLD = 30
RSI_ADD_THRESHOLD = 40
EMA_FAST = 20
EMA_TREND = 200
ATR_PERIOD = 14
SWING_LOOKBACK = 10
RISK_REWARD = 2.0
ATR_SL_MULTIPLIER = 1.5

TELEGRAM_BOT_TOKEN = ""  # fill if you want Telegram alerts
TELEGRAM_CHAT_ID = ""  # fill if you want Telegram alerts

STATE_FILE = "state.json"
POLL_SECONDS = 60
MAX_ADDS = 3


# ---------------- HELPERS ----------------
def send(msg: str):
    """Send message to Telegram or print to console"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[NOTIFY]", msg)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg})
    except Exception as e:
        print("[NOTIFY-ERR]", e)


def ema(series: pd.Series, span: int):
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = up / (down + 1e-12)
    return 100 - (100 / (1 + rs))


def atr(high, low, close, period: int = 14):
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(),
         (high - prev_close).abs(),
         (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def fetch_candles():
    df = yf.download(SYMBOL, period=f"{LOOKBACK_DAYS}d", interval=INTERVAL, group_by="ticker",
                     auto_adjust=False, progress=False)
    df = df.droplevel(0, axis=1)
    df = df.rename(columns={"Open": "open", "High": "high",
                            "Low": "low", "Close": "close",
                            "Volume": "volume"}).dropna()
    return df


def add_indicators(df: pd.DataFrame):
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
        return {"action": "buy", "price": round(last["close"], 2), "tp": tp, "sl": sl,
                "why": f"RSI cross↑{RSI_ENTRY_THRESHOLD} + price above EMA200"}

    near_ema = abs(last["close"] - last["ema_fast"]) / last["ema_fast"] < 0.002
    rsi_bounce = prev["rsi"] <= RSI_ADD_THRESHOLD and last["rsi"] > RSI_ADD_THRESHOLD
    if in_uptrend and near_ema and rsi_bounce:
        tp, sl = compute_levels(last["close"], df)
        return {"action": "add", "price": round(last["close"], 2), "tp": tp, "sl": sl,
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


def fmt_signal(sig):
    return (f"*{SYMBOL}* — {sig['action'].upper()}\n"
            f"Price: {sig['price']}\nTP: {sig['tp']} | SL: {sig['sl']}\nReason: {sig['why']}")


# ---------------- MAIN LOOP ----------------
def run_loop():
    state = load_state()
    send("🤖 XAUUSD RSI+Pullback bot started.")
    while True:
        try:
            df = add_indicators(fetch_candles())
            latest_time = df.index[-1]
            latest_price = float(df["close"].iloc[-1])
            print(f"[{latest_time}] Latest {SYMBOL} close price: {latest_price:.2f}")
            sig = generate_signals(df)
            ts = str(df.index[-1])

            if ts != state.get("last_signal_ts") and sig["action"]:
                last_close = float(df["close"].iloc[-1])
                pos = state.get("position")
                if sig["action"] == "buy":
                    state["position"] = {"side": "long", "entry": last_close, "size": 1, "adds": 0}
                elif sig["action"] == "add" and pos and pos.get("adds", 0) < MAX_ADDS:
                    new_size = pos["size"] + 1
                    pos["entry"] = (pos["entry"] * pos["size"] + last_close) / new_size
                    pos.update({"size": new_size, "adds": pos["adds"] + 1})
                    state["position"] = pos

                state["last_signal_ts"] = ts
                save_state(state)
                send(fmt_signal(sig))

        except Exception as e:
            send(f"[Error] {e}")

        time.sleep(2)


if __name__ == "__main__":
    run_loop()
