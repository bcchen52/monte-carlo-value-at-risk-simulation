# monte-carlo-value-at-risk-simulation

Given a portfolio (tickers and values), and some simulation parameters (time, confidence interval, number of simulations), we return the VaR with a specified confidence.

Using the specific tickers' values from the last year, we find the mean and _, to figure out the geometric brownian motion equation, which is put in a monte carlo simulation for x times. 