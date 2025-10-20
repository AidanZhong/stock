# -*- coding: utf-8 -*-
"""
Created on 2025/10/20 1:06

@author: Aidan
@project: stock
@filename: notifier
"""
import requests

TELEGRAM_BOT_TOKEN = "8288063161:AAEWkF-gkAaVT0fCU5YeHtI8N24DOfgTd5Y"
TELEGRAM_CHAT_ID = "7747333112"

def send_telegram_message(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
    try:
        requests.post(url, data=payload, timeout=5)
        print(f"📩 Sent Telegram alert: {text[:60]}...")
    except Exception as e:
        print("❌ Telegram send failed:", e)
