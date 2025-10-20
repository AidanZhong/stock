"""
Author: Aidan Zhong
Date: 2025/10/20 01:57
Description:
"""
# run_multi_timeframes.py
import subprocess
import sys

timeframes = ["S5", "M1", "M5"]

processes = []
for tf in timeframes:
    print(f"🚀 Launching bot for {tf}")
    p = subprocess.Popen([sys.executable, "bot_template.py", tf])
    processes.append(p)

try:
    for p in processes:
        p.wait()
except KeyboardInterrupt:
    print("🛑 Stopping all bots...")
    for p in processes:
        p.terminate()
