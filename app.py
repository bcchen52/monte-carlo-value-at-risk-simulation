import streamlit as sl

sl.title("Monte Carlo Value at Risk Simulator")
sl.write("Hello")

sl.header("Portfolio Specifcations")

sl.header("Simulation Specifications")

simulation_time = sl.slider("How many days do you want to simulate?", min_value=1, max_value=252, value=30)
simulation_confidence_level = sl.slider("Pick your confidence level", min_value=0.90, max_value=0.99, value=0.95)
simulation_number = sl.number_input("Number of Monte Carlo simulations", min_value=1000, max_value=10000, step=1000)