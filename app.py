import streamlit as sl
import requests

sl.title("Monte Carlo Value at Risk Simulator")
sl.write("Hello")

sl.header("Portfolio Specifcations")

sl.header("Simulation Specifications")

simulation_time = sl.slider("How many days do you want to simulate?", min_value=1, max_value=252, value=30)
simulation_confidence_level = sl.slider("Pick your confidence level", min_value=0.90, max_value=0.99, value=0.95)
simulation_number = sl.number_input("Number of Monte Carlo simulations", min_value=1000, max_value=10000, step=1000)

def fetch_nasdaq_tickers():
    url = "https://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        response.raise_for_status()

        lines = response.text.strip().splitlines()
        df = pd.read_csv(pd.compat.StringIO("\n".join(lines[:-1])), sep="|")
        df = df[df['Test Issue'] == 'N']
        df = df[['Symbol', 'Security Name']].drop_duplicates()
        df.columns = ['Ticker', 'Name']
        return df

    except Exception as e:
        sl.error(f"Failed to fetch NASDAQ tickers: {e}")
        return None

sl.header("Test Ticker Fetch from NASDAQ")

if sl.button("Fetch Tickers"):
    df = fetch_nasdaq_tickers()
    if df is not None:
        sl.success(f"✅ Fetched {len(df)} tickers")
        sl.dataframe(df.head(10))