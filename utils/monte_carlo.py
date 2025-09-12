import pandas as pd
import numpy as np
import yfinance as yf
from curl_cffi import requests
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.colors as pc
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
from scipy.special import ndtri

main_colors = ["#4363d8", "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#ff4500", "#00ced1", "#800000", "#ffd700", "#ff1493"]
secondary_colors = ["#7f8fa6", "#9fa8da", "#a8d5ba"]

def simulate_monte_carlo(portfolio, num_sims, time_frame, time_period, ci):
    # give tickers and dollar samounts
    session = requests.Session(impersonate="chrome")
    # portfolio = {"TICKER": XX.XX, ...}

    #portfolio = {"AAPL": 500000.23, "TSLA": 500000.23, "PYPL": 500000.23, "NVDA":500000.23, "PYPL":500000.23}
    
    tickers = list(portfolio.keys())

    #tickers = ["AAPL", "TSLA", "CHYM"]

    yf_data = yf.download(tickers, period=f"{time_period}y", group_by="ticker", auto_adjust=True, session=session)
    # downloading for multiple will have dates for all

    data = {ticker: yf_data[ticker].dropna()["Close"].to_numpy(dtype=np.float64) for ticker in tickers}

    # if one of the securities missing data, we cut to align them all
    min_len = min(len(v) for v in data.values())

    cleaned_data = {ticker: data[ticker][-min_len:] for ticker in tickers}

    log_returns = np.stack([get_log_returns(cleaned_data[ticker]) for ticker in tickers])

    mean_log_returns = log_returns.mean(axis=1)

    #covariance calculation with ddof=1 for unbiased sample covariance
    covariance = np.cov(log_returns, rowvar=True, ddof=1)

    #for 1 entry, cholesky breaks if not 2D, so make 2D
    if covariance.ndim == 0:
        covariance = np.array([[covariance]])

    weights = np.array([portfolio[ticker]/cleaned_data[ticker][-1] for ticker in tickers])

    # [tickers, time_frame, num_sims] array
    simulated_prices = run_GBM(time_frame, num_sims, mean_log_returns, covariance, np.array([cleaned_data[ticker][-1] for ticker in tickers]))

    initial_prices = np.array([portfolio[ticker] for ticker in tickers])

    weighted_prices = simulated_prices[:, :, :] * weights[:, None, None]

    final_portfolio_values = weighted_prices[:, -1, :].sum(axis=0)

    final_portfolio_returns = final_portfolio_values/(initial_prices.sum()) - 1

    sorted_final_returns = np.sort(final_portfolio_returns)

    final_wprices_returns = np.vstack([weighted_prices[:, -1, :], final_portfolio_values])

    means = (np.exp(mean_log_returns)-1).tolist()

    volatilities = [covariance[i,i] for i in range(len(covariance))]
    
    # sort by portfolio value
    sorted_order = np.argsort(final_portfolio_values)
    final_wprices_returns_sorted = final_wprices_returns[:, sorted_order] 

    plot1 = plot_visualization_1(initial_prices, sorted_final_returns, ci)

    plot2 = plot_visualization_2(initial_prices, weighted_prices, final_portfolio_values, tickers, ci)

    plot3 = plot_visualization_3(initial_prices, weighted_prices, final_portfolio_values, final_wprices_returns_sorted, tickers, ci, mean_log_returns, covariance)
    
    plot4 = plot_visualization_4(initial_prices, weighted_prices, ci)

    plot5 = plot_visualization_5(simulated_prices, time_frame, num_sims, tickers)

    return {"plots":[plot1, plot2, plot3, plot4, plot5], "means":means, "volatilities":volatilities, "tickers":tickers}

def get_log_returns(prices):
    return np.log(prices[1:]/prices[:-1])

def run_GBM(num_days, num_simulations, mean_log_returns, covariance, initial_prices):
    # S(t+1) = S(t) * e^(µ'*(dt) + L*Z*sqrt(dt)))
    # omit dt and sqrt(dt) since dt = 1
    # use len(m) for number of securities
    epsilons = np.random.normal(0, 1, (len(mean_log_returns), num_days, num_simulations))
    price_matrix = np.zeros((len(mean_log_returns), num_days + 1, num_simulations))

    price_matrix[:, 0, :] = initial_prices[:, None]
    # cholesky for factor of cov matrix
    L = np.linalg.cholesky(covariance)

    # we want to calculate the AZ for every day for every simulation, as these are correlated across securities
    # L*epsilon[:,i,j] is L * Z_t, where Z_t represents the chosen normal r.v. at day i simulation j then sum each row k of L * Z_t
    # to get full matrix multiplication and correlation adjusted volatility of k with the rest of the securities
    # rather than use a nested loop of i*j, we can utilize NumPy's einstein summation
    # The string notation tells us the axes of the inputs and the output
    # b being in both inputs tells us to multiply those axes and sum (matrix multiplication)
    adjusted_epsilons = np.einsum('ab,bij->aij', L, epsilons)
    
    for i in range(1, num_days+1):
        price_matrix[:, i, :] = price_matrix[:, i-1, :]*np.exp(mean_log_returns[:, None] + adjusted_epsilons[:, i-1, :])
    
    return price_matrix

def plot_trial(data, title):
    for i in range(0, len(data)):
        plt.plot(data[i])
    #for i in range(int(data.shape[1]/10)):
    #    plt.plot(data[:,i], linewidth=0.5, alpha=0.5)
    plt.title(title)
    plt.xlabel("Day")
    plt.ylabel("Price")
    plt.grid("True")
    plt.show()

def plot_visualization_1(initial_prices, sorted_final_returns, ci):
    fig = go.Figure()

    initial_portfolio_price = initial_prices.sum()

    sorted_portfolio_values = sorted_final_returns * initial_portfolio_price

    cdf = np.arange(1, len(sorted_final_returns)+1) / len(sorted_final_returns)

    min_ret = sorted_final_returns[0]
    max_ret = sorted_final_returns[-1]

    quantile_ci = np.quantile(sorted_final_returns, 1-ci)
    quantile_50 = np.quantile(sorted_final_returns, 0.5)
    
    # In plotly, nbinsx=X will give you bins <= X by rounding the bin width to a "nice" number. However, there is no way to extract the resulting bin width or number of bins, which we could use to calculate bin width.
    # This is used later on for scaling the KDE estimation of the histogram. 
    # I go along with plotly's idea of nicely sized bins, and mimic that autobin function by rounding the the scientific notation decimal of bin widths to 1, 2, or 5,d
    # defined as nice numbers by d3.js https://d3js.org/d3-array/ticks. plotly.js is built on d3.js, so I assume some similarity between JS and Python versions and use d3's implementation.
    
    max_bins = 20

    naive_width = (max_ret - min_ret)/max_bins

    # get the exponent in the form naive_width = X.X * 10^k
    exp = np.floor(np.log10(naive_width))

    sn_dec = naive_width / 10**exp

    if sn_dec <= 1: nice_dec = 1
    elif sn_dec <= 2: nice_dec = 2
    elif sn_dec <= 5: nice_dec = 5
    else: nice_dec = 10 #rounds up to 1, magnitude of 10 above

    nice_width = nice_dec * 10**exp

    # given new weight, give number of bins that at least covers the span
    nice_bins = int(np.ceil((max_ret - min_ret)/ nice_width))

    # histogram
    fig.add_trace(go.Histogram(
        x=sorted_final_returns,
        autobinx = False,
        xbins=dict(
            start=min_ret,
            end=min_ret + nice_width * nice_bins,
            size=nice_width
        ),
        histnorm="probability", #normalizes the counts so histogram stays within range of density 1 to not interfere with desired range for CDF
        name="Returns Probability Histogram",
        marker_color=secondary_colors[0],
        opacity=0.5,
        meta=[nice_width],
        #hovertemplate="Center: %{x:.2%}<br>Density: %{y:.4f}<extra></extra>"
    ))

    # kernel density estimation (KDE) estimates probability distribution from a set of data points, I use this to show continuous estiamte of the histogram
    # scipy defaults to Gaussian KDE with `gaussian_kde()`, used commonly.
    kde = gaussian_kde(sorted_final_returns)
    xs = np.linspace(sorted_final_returns[0], sorted_final_returns[-1], 500)

    #explain how kde is scaled from density to probability.

    # need to scale the kde estimate to match the histogram, which was normalizated with `histnorm="probability'`
    # histogram heights sum to 1 rather than the area
    # in kde, area sums to 1. Instead, we multiply by the bin width to get it on the probability scale. Note that the definition of  
    
    ys = kde(xs) * nice_width

    # kde line
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="lines",
        line=dict(color=secondary_colors[1], width=3),
        name="Scaled KDE",
        hovertemplate="Scaled KDE<extra></extra>"
    ))

    # alternatively, plotly has distplot, which is histogram, kde, rugplot
    abs_sorted_ret_values = np.abs(sorted_portfolio_values)

    formatted_sorted_ret_values = [f"-${abs_sorted_ret_values[i]:,.2f}" if sorted_portfolio_values[i] < 0 else f"+${abs_sorted_ret_values[i]:,.2f}" for i in range(len(sorted_portfolio_values))]

    # CDF
    fig.add_trace(go.Scatter(
        x=sorted_final_returns,
        y=cdf,
        mode="markers",
        name="CDF",
        marker=dict(color=main_colors[0]),
        hovertext=formatted_sorted_ret_values,
        hovertemplate="%{y:.2f} Quantile:<br>%{x:.2%} Return<br>%{hovertext}<extra></extra>"
    ))

    formatted_quantile_ci_value = f"-${abs(quantile_ci)*initial_portfolio_price:,.2f}" if quantile_ci < 0 else f"+${abs(quantile_ci)*initial_portfolio_price:,.2f}"

    # VaR vertical line
    fig.add_trace(go.Scatter(
        x=np.full(200, quantile_ci, dtype=np.float32),
        y=np.linspace(0, 1, 200),
        mode="lines",
        name=f'{1-ci:.2f} Quantile',
        line=dict(color=main_colors[1], width=2),
        meta=[1-ci, ci, quantile_ci, formatted_quantile_ci_value, formatted_quantile_ci_value[1:]],
        hovertemplate="%{meta[0]:.2f} Quantile:<br>%{meta[2]:.2%} Return<br>%{meta[3]}<br>%{meta[1]} VaR:<br>%{meta[4]}<extra></extra>",
    ))

    # VaR point
    fig.add_trace(go.Scatter(
        x=[quantile_ci],
        y=[1-ci],
        mode="markers",
        marker=dict(color=main_colors[1], size=10),
        meta=[1-ci, ci, quantile_ci, formatted_quantile_ci_value, formatted_quantile_ci_value[1:]],
        hovertemplate="%{meta[0]:.2f} Quantile:<br>%{meta[2]:.2%} Return<br>%{meta[3]}<br>%{meta[1]} VaR:<br>%{meta[4]}<extra></extra>",
        showlegend=False
    ))

    # VaR annotation
    fig.add_annotation(
        x=quantile_ci,
        y=0.85,
        text=f"{formatted_quantile_ci_value} ({quantile_ci:.2%})",
        textangle=270,
        showarrow=False,
        xshift=-10,
        yref="y domain"
    )

    formatted_quantile_50_value = f"-${abs(quantile_50)*initial_portfolio_price:,.2f}" if quantile_50 < 0 else f"+${abs(quantile_50)*initial_portfolio_price:,.2f}"

    # 50% quantile point
    fig.add_trace(go.Scatter(
        x=[quantile_50],
        y=[0.50],
        mode="markers",
        marker=dict(color=main_colors[2], size=10),
        meta=[0.50, quantile_50, formatted_quantile_50_value],
        hovertemplate="%{meta[0]:.2f} Quantile (Median):<br>%{meta[1]:.2%} Return<br>%{meta[2]}<extra></extra>",
        showlegend=False
    ))

    # 50% quantile vertical line
    fig.add_trace(go.Scatter(
        x=np.full(200, quantile_50, dtype=np.float32),
        y=np.linspace(0, 1, 200),
        mode="lines",
        name=f'0.50 Quantile',
        line=dict(color=main_colors[2], width=2),
        meta=[0.50, quantile_50, formatted_quantile_50_value],
        hovertemplate="%{meta[0]:.2f} Quantile (Median):<br>%{meta[1]:.2%} Return<br>%{meta[2]}<extra></extra>",
    ))

    # 50% quantile annotation
    fig.add_annotation(
        x=quantile_50,
        y=0.85,
        text=f"{formatted_quantile_50_value} ({quantile_50:.2%})",
        textangle=270,
        showarrow=False,
        xshift=-10,
        yref="y domain"
    )

    # returns are right skewed and by default x axis will be—extend negative return value to match up to -100% for a close to centered histogram
    limit = max(-min_ret, max_ret)
    if limit > 1: limit = 1
    fig.update_xaxes(
        range=[-limit, max_ret],
        tickmode="linear",
        tick0=-limit,
        dtick=max_ret/5,
        tickformat=".1%",    
        showgrid=True,
        gridwidth=1,
    )

    fig.update_layout(
        barmode="overlay",
        title="Returns Cumulative Distribution Function (CDF) and Returns Probability Histogram",
        xaxis=dict(
            title="Return (%)",
        ),
        yaxis=dict(
            title="Probability",
            hoverformat=".3f",),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        template="plotly_dark",
    )

    return fig

def plot_visualization_2(initial_prices, weighted_prices, final_portfolio_values, tickers, ci):
    fig = go.Figure()

    initial_portfolio_price = initial_prices.sum()

    final_wprices = weighted_prices[:, -1, :]

    # stack of securities w/ portfolio value in last row
    final_wprices_returns = np.vstack([final_wprices, final_portfolio_values])
    
    # sort by portfolio value
    sorted_order = np.argsort(final_portfolio_values)
    final_wprices_returns_sorted = final_wprices_returns[:, sorted_order]

    # look at the VaR of the lower tail at 1-ci, CVaR
    lower_ci = int(np.ceil((1-ci)*(final_wprices.shape[1])))

    cut_final_wprices_returns_sorted = final_wprices_returns_sorted[:, :lower_ci]
    
    # security's contribution to VaR = average(change in security/change in portfolio)
    # all relevant rows of arrays are already ordered by tickers, can simply loop
    #var_contributions = np.array([np.mean((cut_final_wprices_returns_sorted[i]-initial_prices[i])/(cut_final_wprices_returns_sorted[-1]-initial_portfolio_price)) for i in range(len(tickers))])
    var_contributions = (-(cut_final_wprices_returns_sorted[:-1] - initial_prices[:, None]).sum(axis=1)) / np.maximum(1e-12, (-(cut_final_wprices_returns_sorted[-1] - initial_portfolio_price)).sum())
    initial_portfolio_contributions = np.array([initial_prices[i]/initial_portfolio_price for i in range(len(tickers))])
    contribution_change = var_contributions/initial_portfolio_contributions

    df = pd.DataFrame({"var": var_contributions, "init": initial_portfolio_contributions, "tickers": tickers, "change": contribution_change})

    # want in increasing order of change from initial weight to CVaR contribution
    df = df.sort_values(by="change", ascending=True)

    # makes more sense for CVaR to be red rather than initial
    fig.add_trace(go.Bar(
        x=df["tickers"],
        y=df["var"],
        name="Mean Contribution to CVaR",
        marker_color=main_colors[1],
        text=df["var"],
        texttemplate="%{text:.1%}",
        textposition="inside",
        hovertext=df["change"],
        meta=[100*ci],
        hovertemplate="%{y:.1%} of %{meta[0]}% CVaR<br>%{hovertext:.3f}x Weight of Initial<extra></extra>",
    ))

    fig.add_trace(go.Bar(
        x=df["tickers"],
        y=df["init"],
        name="Weight of Initial Portfolio Value",
        marker_color=main_colors[0],
        text=df["init"],
        texttemplate="%{text:.1%}",
        textposition="inside",
        hovertemplate="%{y:.1%} of Initial Portfolio<extra></extra>",
    ))

    fig.update_layout(
        barmode="group",
        xaxis=dict(
            title="Security",
        ),
        yaxis=dict(
            title="Weight",
            tickformat=".1%",
        ),
        legend=dict(
            yanchor="bottom",
            orientation="h",
            y=1.02,
        ),
        title=f"Mean Contribution to Conditional {int(ci*100)}% VaR (CVaR) vs Initial Portfolio Weight",
        template="plotly_dark"
    )

    return fig

def plot_visualization_3(initial_prices, weighted_prices, final_portfolio_values, final_wprices_returns_sorted, tickers, ci, mean_log_returns, covariance):
    fig = go.Figure()

    initial_portfolio_price = initial_prices.sum()

    final_wprices = weighted_prices[:, -1, :]

    # stack of securities w/ portfolio value in last row
    final_wprices_returns = np.vstack([final_wprices, final_portfolio_values])
    
    # sort by portfolio value
    sorted_order = np.argsort(final_portfolio_values)
    final_wprices_returns_sorted = final_wprices_returns[:, sorted_order]

    # look at the VaR of the lower tail at 1-ci, CVaR
    lower_ci = int(np.ceil((1-ci)*(final_wprices.shape[1])))

    cut_final_wprices_returns_sorted = final_wprices_returns_sorted[:, :lower_ci]
    
    # security's contribution to VaR = average(change in security/change in portfolio)
    # all relevant rows of arrays are already ordered by tickers, can simply loop
    var_contributions = np.array([np.mean((cut_final_wprices_returns_sorted[i]-initial_prices[i])/(cut_final_wprices_returns_sorted[-1]-initial_portfolio_price)) for i in range(len(tickers))])
    initial_portfolio_contributions = np.array([initial_prices[i]/initial_portfolio_price for i in range(len(tickers))])
    contribution_change = np.abs(var_contributions/initial_portfolio_contributions)

    df = pd.DataFrame({"var": var_contributions, "init": initial_portfolio_contributions, "tickers": tickers, "change": contribution_change})

    std_devs = [covariance[i,i] for i in range(0, len(covariance))]
    means = np.exp(mean_log_returns)-1
    hover_text = [f'''{float(df["change"][i]):.3f}x Weight of CVaR vs Initial''' for i in range(len(tickers))]

    fig.add_trace(go.Scatter(
        x=means,
        y=std_devs,
        mode='markers+text',
        text=tickers,
        textposition='middle center',
        marker=dict(
            size=df["change"]*100,
            sizemode="diameter",
            sizeref=1,
            color=main_colors,
            line=dict(
                width=2,
                color="white",
            )
        ),
        hovertext=hover_text,
        hovertemplate="Mean: %{x:.3%}<br>SD: %{y:.2%}<br>%{hovertext}<extra></extra>",
    ))

    fig.update_layout(
        title="Change in Weight of Conditional Value at Risk (CVaR) vs Initial Portfolio Value",
        xaxis=dict(
            tickformat=".3%",
            title="Mean Daily Return (%)",
        ),
        yaxis=dict(
            tickformat=".2%",
            title="Standard Deviation of Daily Return (%)",
        ),
        template="plotly_dark"
    )
    
    return fig

def plot_visualization_4(initial_prices, weighted_prices, ci):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)

    initial_portfolio_price = initial_prices.sum()
    
    # weighted prices
    wprices = weighted_prices

    # summed weighted prices over days x iterations
    portfolio_values = np.sum(wprices, axis=0)

    #portfolio_values = np.sort(portfolio_values, axis=1)

    return_values = portfolio_values[-1] - initial_portfolio_price

    # when using default np.array, large simulations were slow, use contiguous array and fp32
    contiguous_return_values = np.ascontiguousarray(return_values, dtype=np.float32)

    contiguous_return_values_buffer = contiguous_return_values

    start = int(np.ceil(20/(1-ci)))

    #we want 20 tail points, so 

    step_size = (len(portfolio_values[0])-start)/100

    var_values = []

    var_upper = []

    var_lower = []
    
    z = ndtri((1-ci)/2) # lower z score

    iterations = np.ceil(start + np.arange(1,101) * step_size).astype(int)
    
    for i in range(0,100):
        step = iterations[i]
        var_index = int(np.ceil(step*(1-ci))-1)
        lower = int(np.floor(step*(1-ci) - z * np.sqrt(step*(1-ci)*(ci))))
        upper = int(np.ceil(step*(1-ci) + z * np.sqrt(step*(1-ci)*(ci))))
        contiguous_return_values_buffer[:step].partition([lower, var_index, upper])
        var_values.append(-(contiguous_return_values_buffer[var_index]))
        var_lower.append(-(contiguous_return_values_buffer[lower]))
        var_upper.append(-(contiguous_return_values_buffer[upper]))

    final_var = -(contiguous_return_values_buffer[int(np.ceil(iterations[99]*(1-ci))-1)])

    r, g, b = pc.hex_to_rgb(secondary_colors[0])

    fill_color = f"rgba({r}, {g}, {b}, 0.1)"

    fig.add_trace(go.Scatter(
        x=iterations,
        y=var_lower,
        mode="lines",
        name="Lower Bound of 95% CI",
        line=dict(color=main_colors[1], width=1),
        hovertemplate="%{x:,} Simulations<br>Lower Bound: %{y}<extra></extra>"
    ), row=1, col=1,)

    fig.add_trace(go.Scatter(
        x=iterations,
        y=var_upper,
        mode="lines",
        name="Upper Bound of 95% CI",
        line=dict(color=main_colors[1], width=1),
        fill="tonexty",
        fillcolor=fill_color,
        hovertemplate="%{x:,} Simulations<br>Upper Bound: %{y}<extra></extra>"
    ), row=1, col=1,)

    fig.add_trace(go.Scatter(
        x=iterations,
        y=var_values,
        mode="lines",
        name=f"{int(ci*100)}% VaR",
        line=dict(color=main_colors[0], width=3),
        meta=[int(ci*100)],
        hovertemplate="%{x:,} Simulations<br>%{meta[0]}% VaR: %{y}<extra></extra>",
    ), row=1, col=1,)

    fig.add_trace(go.Scatter(
        x=[iterations[0], iterations[-1]],
        y=[final_var, final_var],
        mode="lines",
        name=f"{int(ci*100)}% VaR at {iterations[-1]:,} Simulations",
        line=dict(color="white", width=1),
        hoverinfo="skip"
    ), row=1, col=1,)

    var_values_np = np.array(var_values)
    var_upper_np = np.array(var_upper)
    var_lower_np = np.array(var_lower)
    ci_half_width = (var_upper_np - var_lower_np) / (2*var_values_np)
    
    fig.add_trace(go.Scatter(
        x=iterations,
        y=ci_half_width,
        mode="lines",
        line=dict(color=main_colors[3], width=1),
        showlegend=False,
        meta=[int(ci*100)],
        hovertemplate="%{x:,} Simulations<br>%{y:.2%} of %{meta[0]}% VaR<extra></extra>",
    ), row=2, col=1,)

    fig.add_annotation(
        x=iterations[int(len(iterations)/2)],
        y=final_var,
        text=f"${final_var:,.0f}",
        ax=0,
        ay=10
    )

    fig.update_layout(
        title=f"{int(ci*100)}% VaR vs Simulations",
        xaxis2=dict(
            title="Number of Simulations",
        ),
        yaxis=dict(
            title=f"{int(ci*100)}% VaR ($)",
            tickprefix="$", 
            tickformat=",.0f",
        ),
        yaxis2=dict(
            title="Relative CI Half-Width (%)",
            tickformat=".2%",
        ),
        legend=dict(
            yanchor="bottom",
            orientation="h",
            y=1.02,
        ),
        template="plotly_dark"
    )

    return fig

def plot_visualization_5(simulated_prices, num_days, num_simulations, tickers):
    fig = go.Figure()

    np_data = np.empty([len(tickers), num_days+1, num_simulations], dtype=np.float64)

    for i, ticker in enumerate(tickers):
        for j in range(num_days+1):
            np_data[i, j, :] = np.asarray(simulated_prices[i, j, :])

    np_returns = (np_data / np_data[:, :1, :]) - 1

    median = int(np.ceil(0.5*num_simulations)-1)

    np_returns.partition(median, axis=2)

    for i, ticker in enumerate(tickers):
        fig.add_trace(go.Scatter(
            x=np.arange(0, num_days),
            y=np_returns[i,:,median],
            mode="lines",
            name=ticker,
            line=dict(color=main_colors[i], width=2),
            meta=[ticker],
            hovertemplate="%{meta[0]} Day %{x}<br>Cumulative return: %{y:.3%}<extra></extra>",
        ))
    
    fig.update_layout(
        title="Median Cumulative Return",
        legend=dict(
            yanchor="bottom",
            orientation="h",
            y=1.02,
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),
        xaxis=dict(
            title="Day",
        ),
        yaxis=dict(
            title="Cumulative Return (%)",
            tickformat=".3%",    
        ),
        template="plotly_dark"
    )

    return fig