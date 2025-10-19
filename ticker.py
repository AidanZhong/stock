# -*- coding: utf-8 -*-
"""
Created on 2025/10/19 21:27

@author: Aidan
@project: stock
@filename: ticker
"""
import yfinance as yf

ticker = yf.Ticker("GC=F")
print(ticker.info)
