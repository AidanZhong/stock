# -*- coding: utf-8 -*-
"""
Created on 2025/10/19 22:14

@author: Aidan
@project: stock
@filename: test
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np, time

from notifier import send_telegram_message

# matplotlib.use("TkAgg")
# plt.ion()
# fig, ax = plt.subplots()
# x, y = [], []
#
# for i in range(100):
#     x.append(i)
#     y.append(np.sin(i / 10))
#     ax.clear()
#     ax.plot(x, y, color='blue')
#     ax.set_title("Live updating test")
#     fig.canvas.draw()
#     fig.canvas.flush_events()
#     time.sleep(0.1)

send_telegram_message("test")