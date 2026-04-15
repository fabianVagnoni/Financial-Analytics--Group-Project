"""
AR model selection, fitting, and diagnostics.
Source: Lecture 5.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t as sp_t
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
from arch import arch_model

from config import AR_MAX_LAGS


def test_stationarity(log_ret):
    """
    ADF test on log-returns.
    Returns dict with test results.
    From Lecture 5.
    """
    result = adfuller(log_ret.dropna().values)
    adf_stat, p_value, used_lag, n_obs, crit_vals, ic_best = result

    return {
        "adf_stat": adf_stat,
        "p_value": p_value,
        "used_lag": used_lag,
        "critical_values": crit_vals,
        "is_stationary": p_value < 0.05,
    }


def test_autocorrelation(log_ret, lags=10):
    """
    Ljung-Box test for autocorrelation.
    Returns dict with test results.
    From Lecture 5.
    """
    lb = acorr_ljungbox(log_ret.dropna().values, lags=[lags], return_df=True)
    lb_stat = float(lb["lb_stat"].iloc[0])
    lb_p = float(lb["lb_pvalue"].iloc[0])

    return {
        "lb_stat": lb_stat,
        "p_value": lb_p,
        "has_autocorrelation": lb_p < 0.05,
    }


def select_ar_order(log_ret, max_p=AR_MAX_LAGS, dist="t", criterion="BIC"):
    """
    Grid search AR(1)..AR(max_p) with constant variance, Student-t innovations.
    Returns (best_p, ic_table DataFrame).
    From Lecture 5.
    """
    ret = 100 * log_ret.dropna()
    rows = []

    for p in range(1, max_p + 1):
        try:
            m = arch_model(ret, mean="AR", lags=p, vol="Constant", dist=dist)
            r = m.fit(disp="off")
            rows.append({"p": p, "AIC": r.aic, "BIC": r.bic})
        except Exception:
            rows.append({"p": p, "AIC": np.nan, "BIC": np.nan})

    ic_table = pd.DataFrame(rows)

    col = criterion
    best_idx = ic_table[col].idxmin()
    best_p = int(ic_table.loc[best_idx, "p"])

    return best_p, ic_table


def fit_ar_model(log_ret, p, dist="t"):
    """
    Fit AR(p) with constant variance and return the arch result object.
    From Lecture 5.
    """
    ret = 100 * log_ret.dropna()
    m = arch_model(ret, mean="AR", lags=p, vol="Constant", dist=dist)
    return m.fit(disp="off")


def plot_ar_diagnostics(result, ticker):
    """
    2x2 panel: standardized residuals diagnostics.
    From Lecture 5.
    """
    std_resid = result.std_resid.dropna()
    nu = result.params.get("nu", 5.0)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    # Residuals time series
    axes[0, 0].plot(std_resid.index, std_resid.values, linewidth=0.5, color="black")
    axes[0, 0].axhline(0, color="red", linestyle="--", linewidth=0.8)
    axes[0, 0].set_title(f"{ticker} — Standardized Residuals")

    # ACF of residuals
    plot_acf(std_resid.values, ax=axes[0, 1], lags=30, alpha=0.05)
    axes[0, 1].set_title(f"{ticker} — ACF of Residuals")

    # ACF of squared residuals
    plot_acf(std_resid.values ** 2, ax=axes[1, 0], lags=30, alpha=0.05)
    axes[1, 0].set_title(f"{ticker} — ACF of Squared Residuals")

    # QQ plot vs Student-t
    sorted_resid = np.sort(std_resid.values)
    n = len(sorted_resid)
    theoretical = sp_t.ppf(np.linspace(1 / (n + 1), n / (n + 1), n), df=nu)
    axes[1, 1].scatter(theoretical, sorted_resid, s=5, alpha=0.5)
    lims = [min(theoretical.min(), sorted_resid.min()),
            max(theoretical.max(), sorted_resid.max())]
    axes[1, 1].plot(lims, lims, "r--", linewidth=1)
    axes[1, 1].set_xlabel("Theoretical Quantiles (Student-t)")
    axes[1, 1].set_ylabel("Sample Quantiles")
    axes[1, 1].set_title(f"{ticker} — QQ Plot vs Student-t (df={nu:.2f})")

    plt.suptitle(f"{ticker} — AR Model Diagnostics", fontsize=14, y=1.01)
    plt.tight_layout()
    return fig


def plot_ar_ic_selection(ic_table, ticker):
    """
    Plot AIC and BIC vs AR order with minima marked.
    From Lecture 5.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(ic_table["p"], ic_table["AIC"], marker="o", label="AIC")
    ax.plot(ic_table["p"], ic_table["BIC"], marker="s", label="BIC")

    best_aic_p = int(ic_table.loc[ic_table["AIC"].idxmin(), "p"])
    best_bic_p = int(ic_table.loc[ic_table["BIC"].idxmin(), "p"])

    ax.axvline(best_aic_p, color="blue", linestyle=":", alpha=0.5, label=f"Best AIC: AR({best_aic_p})")
    ax.axvline(best_bic_p, color="orange", linestyle=":", alpha=0.5, label=f"Best BIC: AR({best_bic_p})")

    ax.set_xlabel("AR Order (p)")
    ax.set_ylabel("Information Criterion")
    ax.set_title(f"{ticker} — AR Order Selection")
    ax.legend()

    plt.tight_layout()
    return fig
