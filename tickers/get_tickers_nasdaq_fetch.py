import pandas as pandas
import os
import datetime as dt 
import yfinance as yf
import requests


def fetch_nasdaq_tickers(save_path="ticker_list.txt"):
    url_nasdaq = "https://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
    url_other = "https://ftp.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
    
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url_nasdaq, headers=headers, timeout=10)

    with open("nasdaqlisted.txt", "w") as f:
        f.write(response.text)

fetch_nasdaq_tickers()