import streamlit as sl
import pandas as pd
import os
import plotly.graph_objects as go
from tickers.get_tickers import fetch_nasdaq_tickers
from utils.shared_lock import FILE_LOCK
from utils.monte_carlo import simulate_monte_carlo

base_dir = os.path.dirname(os.path.abspath(__file__))
tickers_path = os.path.join(base_dir, "tickers", "ticker_list.txt")
time_path = os.path.join(base_dir, "tickers", "timestamp.txt")

if "main" not in sl.session_state:
    sl.session_state.main = True

if sl.session_state["main"]:
    sl.title("Monte Carlo Value at Risk Simulator")

    sl.text("To simulate the Value at Risk of a portfolio, historical data is used to get the sample mean and sample standard deviation of logarithmic returns to simulate monte carlo trials of the Geometric Brownian Motion equation.")

    sl.text("These trials are used to get information about the spread and likelihood of potential results.")

    sl.markdown("To read more about the math, explanation in [repo](https://github.com/bcchen52/monte-carlo-value-at-risk-simulation)")

    #we use session_state to locally cache values based on session to prevent rereading unecessarily, as streamlit reloads on any event
    if "tickers" not in sl.session_state:
        with open(time_path, "r") as f:
            sl.session_state.time = f.readline()
        with FILE_LOCK:
            with open(tickers_path, "r") as f:
                sl.session_state.tickers = [line.strip() for line in f.readlines()]

    if "num_portfolio_entries" not in sl.session_state:
        sl.session_state.num_portfolio_entries = 1

    sl.header("Portfolio Specifations")
    sl.markdown("**Add up to 15 securites below by searching for ticker symbols.**")
    sl.markdown('''**To leave an entry empty, keep the security as the default value, "-"**''')
    sl.markdown("*Securities are updated daily at 4AM EST and chosen from nasdaqtrader's [nasdaqlisted.txt](https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt) and [otherlisted.txt](https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt) files and crossreferenced on yfinance to cover a broad portion of commonly traded securities that have available trading data.*")
    sl.success(f"Tickers last updated {sl.session_state.time}")

    portfolio = []

    for i in range(0, sl.session_state.num_portfolio_entries):
        ticker_col, value_col = sl.columns([2, 1])
        with ticker_col:
            ticker = sl.selectbox(f"Security {i+1}", ["-"] + sl.session_state.tickers, key=f"security_{i}")
        with value_col:
            #price = sl.number_input(f"Price {i+1}", min_value=0.00, max_value=1000000000.00, format="%.2f", step=10.0, key=f"price_{i}")   
            price_str = sl.text_input(f"Price {i+1}", placeholder="Enter price", key=f"price_{i}", max_chars=15)
            try:
                if price_str:
                    price = float(price_str)
                    if price <= 0:
                        sl.error("Enter a valid positive number.")
                        price = None
                    else:
                        price = round(price, 2)
                else:
                    price = None
            except ValueError:
                sl.error("Enter a valid positive number.")
                price = None
        if ticker != "-" and price is not None and price > 0:
            cleaned_name = ticker.split(" - ", 1)
            portfolio.append((f"{cleaned_name[1]} ({cleaned_name[0]})", price, cleaned_name[0]))

    if sl.button("Add another security"):
        if sl.session_state.num_portfolio_entries < 15:
            sl.session_state.num_portfolio_entries += 1
            sl.rerun() #force rerun so we see the newly added ticker; directly adding another ticker here puts it under the button
        else:
            sl.error("Max 15 different securities")

    #if portfolio, show visualization
    if len(portfolio) > 0:
        labels = [item[0] for item in portfolio]
        values = [item[1] for item in portfolio]
        total = sum(values)

        #create pie chart with plotly
        fig = go.Figure(
            data = [go.Pie(labels=labels, values=values, hole=0.6)]
        )

        fig.update_layout(
            title = {
                'text': f"Portfolio Breakdown",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24}
            }
        )

        sl.markdown(f"### Portfolio Value: ${total:,.2f}")

        portfolio = sorted(portfolio, key=lambda x:x[1], reverse=True) #(TICKER, VAL), sort by val in decreasing order
        
        for name, value, ticker in portfolio:
            sl.markdown(f"{name}: **${value:,.2f}**")

        #display chart
        sl.plotly_chart(fig)

    sl.header("Simulation Specifications")

    simulation_time = sl.slider("How many days do you want to simulate?", min_value=1, max_value=252, value=30)
    sl.text("252 trading days is approximately a year")
    simulation_confidence_level = sl.slider("Pick your confidence level", min_value=0.90, max_value=0.99, value=0.95)
    simulation_number = sl.number_input("Number of Monte Carlo simulations", min_value=50000, max_value=200000, step=10000)
    sl.text("50,000 to 200,000")
    simulation_range = sl.slider("How many years of data do you want to use?", min_value=1, max_value=5, step=1)

    # submit button

    if sl.button("Simulate"):
        if len(portfolio) > 0:
            #sl.session_state["portfolio"] = portfolio
            sl.session_state.main=False
            altered_portfolio = {x[2]:x[1] for x in portfolio}
            sl.session_state.results=simulate_monte_carlo(altered_portfolio, simulation_number, simulation_time, simulation_range, simulation_confidence_level)
            sl.rerun()
        else:
            sl.error("Your current portfolio is empty.")

    #after submit, actually do stuff
else:
    sl.title("Portfolio Results")

    if sl.button("Simulate again"):
        sl.session_state.main=True
        sl.session_state.num_portfolio_entries = 1
        sl.rerun()

    for i in range(len(sl.session_state.results["tickers"])):
        sl.text(f'''({sl.session_state.results["tickers"][i]}): Average {sl.session_state.results["means"][i]*100:.2f}% daily change with average {sl.session_state.results["volatilities"][i]*100:.2f}% volatility''')

    sl.plotly_chart(sl.session_state.results["plots"][0], use_container_width=True)
    sl.text("The returns empirical CDF has the 0.50 and specificed VaR quantile marked.")
    sl.text("Below is a histogram scaled to probability, with a Gaussian kernel density estimation scaled similarly to show the distribution of returns.")
    
    sl.plotly_chart(sl.session_state.results["plots"][1], use_container_width=True)
    sl.plotly_chart(sl.session_state.results["plots"][2], use_container_width=True)
    sl.plotly_chart(sl.session_state.results["plots"][3], use_container_width=True)
    sl.text("VaR vs Simulations shows the VaR and the 95% confidence interval of the VaR as the number of simulations increase.")
    sl.text("The CI half-with stabilizing indicates an appropriate number of simulations.")

    sl.plotly_chart(sl.session_state.results["plots"][4], use_container_width=True)

    