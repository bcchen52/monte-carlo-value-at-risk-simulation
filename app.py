import streamlit as sl
import requests
import pandas as pd
import os
from tickers.get_tickers import fetch_nasdaq_tickers
from io import StringIO
from utils.shared_lock import FILE_LOCK

base_dir = os.path.dirname(os.path.abspath(__file__))
tickers_path = os.path.join(base_dir, "tickers", "ticker_list.txt")
time_path = os.path.join(base_dir, "tickers", "timestamp.txt")

sl.title("Monte Carlo Value at Risk Simulator")
sl.write("Hello")

sl.header("Portfolio Specifcations")

with open(time_path, "r") as f:
    time = f.readline()
sl.success(f"Tickers last updated {time}")

with FILE_LOCK:
    with open(tickers_path, "r") as f:
        tickers = [line.strip() for line in f.readlines()]
    selected_ticker = sl.selectbox("Selectn a Ticker", options=tickers, placeholder="Search Ticker")

sl.header("Simulation Specifications")

simulation_time = sl.slider("How many days do you want to simulate?", min_value=1, max_value=252, value=30)
simulation_confidence_level = sl.slider("Pick your confidence level", min_value=0.90, max_value=0.99, value=0.95)
simulation_number = sl.number_input("Number of Monte Carlo simulations", min_value=1000, max_value=10000, step=1000)

sl.header("Test Ticker Fetch from NASDAQ")

if sl.button("Fetch Tickers"):
    df = fetch_nasdaq_tickers()
    if df is not None:
        sl.success(f"✅ Fetched {len(df)} tickers")
        sl.dataframe(df.head(10))