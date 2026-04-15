"""
Exploratory Data Analysis: plots and summary statistics.
Source: Lectures 2-3.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, probplot, skew, kurtosis, shapiro, kstest, anderson
from statsmodels.stats.stattools import jarque_bera
from statsmodels.graphics.tsaplots import plot_acf

from config import ROLLING_WINDOW, ACF_LAGS


def plot_price_and_returns(price, log_ret, ticker):
    """
    2x1 subplot: Close prices on top, log-returns below.
    From Lecture 2.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(price.index, price.values, color="steelblue", linewidth=0.9)
    axes[0].set_title(f"{ticker} — Daily Close Price")
    axes[0].set_ylabel("Price")

    axes[1].plot(log_ret.index, log_ret.values, color="black", linewidth=0.5)
    axes[1].axhline(0, color="red", linestyle="--", linewidth=0.8)
    axes[1].set_title(f"{ticker} — Daily Log-Returns")
    axes[1].set_ylabel("Log-Return")
    axes[1].set_xlabel("Date")

    plt.tight_layout()
    return fig


def plot_distribution_diagnostics(log_ret, ticker, acf_lags=ACF_LAGS):
    """
    2x2 subplot: time-series, ACF, histogram + Normal fit, QQ-plot.
    From Lecture 2.
    """
    mu_hat, sig_hat = norm.fit(log_ret.values)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    # Time series
    axes[0, 0].plot(log_ret.index, log_ret.values, linewidth=0.5, color="black")
    axes[0, 0].axhline(0, color="red", linestyle="--", linewidth=0.8)
    axes[0, 0].set_title(f"{ticker} — Log-Returns")

    # ACF
    plot_acf(log_ret.values, ax=axes[0, 1], lags=acf_lags, alpha=0.05)
    axes[0, 1].set_title(f"{ticker} — ACF of Log-Returns")

    # Histogram + Normal
    axes[1, 0].hist(log_ret.values, bins=50, density=True, alpha=0.6,
                    color="steelblue", edgecolor="black")
    x_grid = np.linspace(log_ret.min(), log_ret.max(), 300)
    axes[1, 0].plot(x_grid, norm.pdf(x_grid, mu_hat, sig_hat),
                    color="red", linewidth=1.5, label="Normal fit")
    axes[1, 0].set_title(f"{ticker} — Histogram + Normal")
    axes[1, 0].legend()

    # QQ-plot
    probplot(log_ret.values, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title(f"{ticker} — QQ Plot vs Normal")

    plt.suptitle(f"{ticker} — Distribution Diagnostics", fontsize=14, y=1.01)
    plt.tight_layout()
    return fig


def plot_volatility_clustering(log_ret, ticker, window=ROLLING_WINDOW):
    """
    |log-return| and rolling standard deviation overlay.
    From Lecture 2.
    """
    abs_ret = log_ret.abs()
    roll_sd = log_ret.rolling(window).std()

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(abs_ret.index, abs_ret.values, alpha=0.3, color="grey",
            linewidth=0.7, label="|Log-Return|")
    ax.plot(roll_sd.index, roll_sd.values, color="red",
            linewidth=1.2, label=f"Rolling SD ({window}d)")
    ax.set_title(f"{ticker} — Volatility Clustering")
    ax.set_xlabel("Date")
    ax.set_ylabel("Absolute Return / Rolling SD")
    ax.legend()

    plt.tight_layout()
    return fig


def compute_summary_statistics(log_ret):
    """
    Returns dict with mean, variance, std, skewness, excess kurtosis.
    From Lecture 3.
    """
    vals = log_ret.dropna().values
    return {
        "mean": float(np.mean(vals)),
        "variance": float(np.var(vals, ddof=1)),
        "std": float(np.std(vals, ddof=1)),
        "skewness": float(skew(vals)),
        "excess_kurtosis": float(kurtosis(vals, fisher=True)),
    }


def run_normality_tests(log_ret):
    """
    Run Shapiro-Wilk, Jarque-Bera, KS, Anderson-Darling tests.
    Returns DataFrame with test results.
    From Lecture 3.
    """
    vals = log_ret.dropna().values
    mu, sigma = norm.fit(vals)

    rows = []

    # Shapiro-Wilk (limit to 5000 obs)
    sw_stat, sw_p = shapiro(vals[:5000])
    rows.append(("Shapiro-Wilk", sw_stat, sw_p, "Reject" if sw_p < 0.05 else "Fail to reject"))

    # Jarque-Bera
    jb_stat, jb_p, jb_skew, jb_kurt = jarque_bera(vals)
    rows.append(("Jarque-Bera", jb_stat, jb_p, "Reject" if jb_p < 0.05 else "Fail to reject"))

    # KS test
    ks_stat, ks_p = kstest(vals, "norm", args=(mu, sigma))
    rows.append(("Kolmogorov-Smirnov", ks_stat, ks_p, "Reject" if ks_p < 0.05 else "Fail to reject"))

    # Anderson-Darling
    ad = anderson(vals, dist="norm")
    # Use 5% significance level
    ad_crit_5 = ad.critical_values[2]  # index 2 = 5% level
    ad_decision = "Reject" if ad.statistic > ad_crit_5 else "Fail to reject"
    rows.append(("Anderson-Darling", ad.statistic, None, ad_decision))

    df = pd.DataFrame(rows, columns=["Test", "Statistic", "p-value", "Decision (5%)"])
    return df


def plot_acf_returns_and_squared(log_ret, ticker, lags=30):
    """
    1x2 subplot: ACF of log-returns (left), ACF of squared log-returns (right).
    From Lecture 7.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    plot_acf(log_ret.values, ax=axes[0], lags=lags, alpha=0.05)
    axes[0].set_title(f"{ticker} — ACF of Log-Returns")

    plot_acf(log_ret.values ** 2, ax=axes[1], lags=lags, alpha=0.05)
    axes[1].set_title(f"{ticker} — ACF of Squared Log-Returns")

    plt.tight_layout()
    return fig
