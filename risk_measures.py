"""
Static and dynamic VaR/ES, and EVT (Peaks Over Threshold / GPD).
Source: Lectures 8-9.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import norm, genpareto

from config import VAR_LEVELS, EVT_THRESHOLD_QUANTILE, EVT_ALPHA


# =====================================================================
# STATIC VaR & ES (Lecture 8)
# =====================================================================

def compute_var_from_losses(losses_series, var_levels=VAR_LEVELS):
    """
    Historical, Normal, Student-t VaR at given confidence levels.
    Returns DataFrame with columns: [VaR Level, Historical, Normal, T-dist].
    LIFTED from Lecture 8.
    """
    x = pd.Series(losses_series).dropna().astype(float)

    try:
        df_t, loc_t, scale_t = stats.t.fit(x)
    except Exception:
        loc_t, scale_t = x.mean(), x.std(ddof=1)
        df_t = 5.0

    mu = x.mean()
    sigma = x.std(ddof=1)

    rows = {"VaR Level": [], "Historical": [], "Normal": [], "T-dist": []}

    for alpha in var_levels:
        var_hist = float(np.percentile(x, 100 * alpha))
        var_norm = mu + sigma * norm.ppf(alpha)
        var_t = float(stats.t.ppf(alpha, df_t, loc=loc_t, scale=scale_t))

        rows["VaR Level"].append(alpha)
        rows["Historical"].append(var_hist)
        rows["Normal"].append(var_norm)
        rows["T-dist"].append(var_t)

    return pd.DataFrame(rows)


def compute_es_from_losses(losses_series, es_levels=VAR_LEVELS):
    """
    Historical, Normal, Student-t Expected Shortfall.
    Returns DataFrame with columns: [ES Level, Historical, Normal, T-dist].
    LIFTED from Lecture 8.
    """
    x = pd.Series(losses_series).dropna().astype(float)

    try:
        df_t, loc_t, scale_t = stats.t.fit(x)
    except Exception:
        loc_t, scale_t = x.mean(), x.std(ddof=1)
        df_t = 5.0

    mu = x.mean()
    sigma = x.std(ddof=1)

    rows = {"ES Level": [], "Historical": [], "Normal": [], "T-dist": []}

    for alpha in es_levels:
        # Historical ES
        var_hist = float(np.percentile(x, 100 * alpha))
        es_hist = float(x[x >= var_hist].mean())

        # Normal ES
        z = norm.ppf(alpha)
        es_norm = mu + sigma * (norm.pdf(z) / (1 - alpha))

        # Student-t ES
        q = stats.t.ppf(alpha, df_t)
        pdf_q = stats.t.pdf(q, df_t)
        if df_t > 1:
            es_t = loc_t + scale_t * ((df_t + q ** 2) / ((df_t - 1) * (1 - alpha))) * pdf_q
        else:
            es_t = np.nan

        rows["ES Level"].append(alpha)
        rows["Historical"].append(es_hist)
        rows["Normal"].append(es_norm)
        rows["T-dist"].append(es_t)

    return pd.DataFrame(rows)


def plot_var_comparison(var_table, ticker, losses=None):
    """
    1x2 subplot: histogram with VaR lines (left), VaR vs level (right).
    From Lecture 8.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: histogram with VaR lines
    if losses is not None:
        axes[0].hist(losses.dropna(), bins=50, density=True, alpha=0.6,
                     color="darkgreen", edgecolor="black")
        for _, row in var_table.iterrows():
            alpha = row["VaR Level"]
            v = row["Historical"]
            if alpha in [0.95, 0.99, 0.999]:
                axes[0].axvline(v, color="red", linestyle="--", linewidth=1.2,
                                label=f"Hist VaR {alpha:.1%} = {v:.2f}%")
        axes[0].set_title(f"{ticker} — Loss Distribution with VaR")
        axes[0].set_xlabel("Loss (%)")
        axes[0].set_ylabel("Density")
        handles, labels = axes[0].get_legend_handles_labels()
        if labels:
            axes[0].legend(fontsize="small")

    # Right: VaR vs confidence level
    axes[1].plot(var_table["VaR Level"], var_table["Historical"],
                 label="Historical", linestyle="dashed", marker="o")
    axes[1].plot(var_table["VaR Level"], var_table["Normal"],
                 label="Normal", linestyle="dotted", marker="s")
    axes[1].plot(var_table["VaR Level"], var_table["T-dist"],
                 label="T-dist", linestyle="solid", marker="^")
    axes[1].set_title(f"{ticker} — VaR vs Confidence Level")
    axes[1].set_xlabel("Confidence Level")
    axes[1].set_ylabel("VaR (Loss %)")
    axes[1].set_ylim(bottom=0)
    axes[1].legend()

    plt.tight_layout()
    return fig


def plot_es_comparison(es_table, ticker):
    """
    ES vs confidence level plot.
    From Lecture 8.
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(es_table["ES Level"], es_table["Historical"],
            label="Historical", linestyle="dashed", marker="o")
    ax.plot(es_table["ES Level"], es_table["Normal"],
            label="Normal", linestyle="dotted", marker="s")
    ax.plot(es_table["ES Level"], es_table["T-dist"],
            label="T-dist", linestyle="solid", marker="^")

    ax.set_title(f"{ticker} — Expected Shortfall vs Confidence Level")
    ax.set_xlabel("Confidence Level")
    ax.set_ylabel("ES (Loss %)")
    ax.set_ylim(bottom=0)
    ax.legend()

    plt.tight_layout()
    return fig


# =====================================================================
# DYNAMIC VaR & ES (Lecture 8, GARCH-based)
# =====================================================================

def compute_dynamic_var(mu_r, scale_lambda, nu, alpha_levels=None):
    """
    Dynamic GARCH-based VaR for losses.
    Returns dict mapping alpha -> pd.Series of VaR values.
    From Lecture 8.

    VaR_alpha(L) = -q_{1-alpha}(R) where R is the return distribution.
    """
    if alpha_levels is None:
        alpha_levels = [0.95, 0.99]

    result = {}
    for alpha in alpha_levels:
        q_ret = stats.t.ppf(1 - alpha, df=nu, loc=mu_r.values, scale=scale_lambda.values)
        var_l = pd.Series(-q_ret, index=mu_r.index, name=f"VaR{int(alpha * 100)}")
        result[alpha] = var_l

    return result


def compute_dynamic_es(mu_r, scale_lambda, nu, alpha_levels=None):
    """
    Dynamic GARCH-based ES for losses.
    Returns dict mapping alpha -> pd.Series of ES values.
    From Lecture 8.
    """
    if alpha_levels is None:
        alpha_levels = [0.95, 0.99]

    result = {}
    for alpha in alpha_levels:
        q = stats.t.ppf(1 - alpha, df=nu)
        pdf_q = stats.t.pdf(q, df=nu)

        if nu <= 1:
            raise ValueError("ES undefined for t distribution with df <= 1.")

        es_std_left = -(pdf_q / (1 - alpha)) * (nu + q ** 2) / (nu - 1)
        es_returns = mu_r.values + scale_lambda.values * es_std_left
        es_losses = -es_returns

        result[alpha] = pd.Series(es_losses, index=mu_r.index, name=f"ES{int(alpha * 100)}")

    return result


def plot_dynamic_risk(loss_proxy, var_dict, es_dict=None, ticker=""):
    """
    Time series of losses with dynamic VaR/ES overlays.
    From Lecture 8.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(loss_proxy.index, loss_proxy.values, alpha=0.6,
            linewidth=0.7, label="Loss $L_t = -R_t$")

    colors_var = ["tab:orange", "tab:red"]
    for i, (alpha, var_s) in enumerate(var_dict.items()):
        c = colors_var[i % len(colors_var)]
        ax.plot(var_s.index, var_s.values, linestyle="dashed", color=c,
                label=f"Dynamic VaR {alpha:.0%}")

    if es_dict is not None:
        colors_es = ["tab:purple", "tab:brown"]
        for i, (alpha, es_s) in enumerate(es_dict.items()):
            c = colors_es[i % len(colors_es)]
            ax.plot(es_s.index, es_s.values, linestyle="dotted", color=c,
                    linewidth=1.5, label=f"Dynamic ES {alpha:.0%}")

    ax.set_xlabel("Date")
    ax.set_ylabel("Loss (%)")
    ax.set_title(f"{ticker} — Dynamic VaR & ES (GARCH, t innovations)")
    ax.legend(fontsize="small")

    plt.tight_layout()
    return fig


def count_var_exceedances(loss_proxy, var_dict):
    """
    Count VaR exceedances vs expected rate.
    Returns DataFrame with results.
    From Lecture 8.
    """
    rows = []
    for alpha, var_s in var_dict.items():
        aligned = loss_proxy.reindex(var_s.index).dropna()
        var_aligned = var_s.reindex(aligned.index)
        exceedances = (aligned > var_aligned).sum()
        n = len(aligned)
        expected = n * (1 - alpha)
        rows.append({
            "Level": alpha,
            "Exceedances": int(exceedances),
            "Expected": expected,
            "Rate": exceedances / n if n > 0 else np.nan,
            "Expected Rate": 1 - alpha,
        })
    return pd.DataFrame(rows)


# =====================================================================
# EVT: Peaks Over Threshold / GPD (Lecture 9)
# =====================================================================

def fit_gpd_pot(std_losses, threshold_quantile=EVT_THRESHOLD_QUANTILE, threshold_value=None):
    """
    Fit GPD to exceedances over threshold.
    Returns dict with u, q_u, p_u, xi, sigma, nu_exceed, n.
    From Lecture 9.
    """
    x = pd.Series(std_losses).dropna().astype(float)

    if threshold_value is not None:
        u = float(threshold_value)
        q_u = float((x <= u).mean())
    else:
        q_u = float(threshold_quantile)
        u = float(np.quantile(x, q_u))

    exceed = x[x > u]
    excesses = (exceed - u).values
    n = len(x)
    nu_exceed = len(excesses)
    p_u = nu_exceed / n

    if nu_exceed < 10:
        raise RuntimeError(
            f"Too few exceedances ({nu_exceed}) for stable GPD fit. Consider lowering threshold."
        )

    xi_hat, _, sigma_hat = genpareto.fit(excesses, floc=0)

    return {
        "u": u,
        "q_u": q_u,
        "p_u": p_u,
        "xi": xi_hat,
        "sigma": sigma_hat,
        "nu_exceed": nu_exceed,
        "n": n,
    }


def compute_evt_var_es(gpd_params, alpha=EVT_ALPHA):
    """
    Compute EVT VaR and ES on the STANDARDIZED loss scale.
    Returns dict with VaR_Z and ES_Z.
    From Lecture 9.
    """
    u = gpd_params["u"]
    p_u = gpd_params["p_u"]
    xi = gpd_params["xi"]
    sigma = gpd_params["sigma"]

    if alpha <= 1 - p_u:
        raise ValueError(
            f"alpha={alpha} not in tail region. Need alpha > {1 - p_u:.4f}."
        )

    # VaR formula
    if abs(xi) < 1e-8:
        var_z = u + sigma * np.log(p_u / (1 - alpha))
    else:
        var_z = u + (sigma / xi) * (((1 - alpha) / p_u) ** (-xi) - 1)

    # ES (exists only if xi < 1)
    if xi >= 1:
        es_z = np.inf
    else:
        es_z = (var_z + sigma - xi * u) / (1 - xi)

    return {"VaR_Z": var_z, "ES_Z": es_z}


def compute_dynamic_evt_risk(garch_analyzer, log_ret, evt_var_z, evt_es_z):
    """
    Scale EVT standardized VaR/ES by GARCH conditional volatility.
    Returns (VaR_L series, ES_L series) for actual losses.
    From Lecture 9.
    """
    cond_vol = garch_analyzer.conditional_volatility.dropna()
    mu = garch_analyzer.mu_const

    loss = -(100 * log_ret).dropna()
    idx = loss.index.intersection(cond_vol.index)
    cond_vol = cond_vol.loc[idx]

    var_l = -mu + cond_vol * evt_var_z
    es_l = -mu + cond_vol * evt_es_z

    var_l.name = "EVT_VaR"
    es_l.name = "EVT_ES"

    return var_l, es_l


def plot_gpd_fit(std_losses, gpd_params, ticker):
    """
    2x1: full distribution with GPD tail overlay, exceedances histogram + GPD fit.
    From Lecture 9.
    """
    x = pd.Series(std_losses).dropna().astype(float)
    u = gpd_params["u"]
    xi = gpd_params["xi"]
    sigma = gpd_params["sigma"]
    p_u = gpd_params["p_u"]

    excesses = (x[x > u] - u).values

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Full distribution with GPD tail
    axes[0].hist(x.values, bins=40, density=True, alpha=0.6, edgecolor="black")
    axes[0].axvline(u, linestyle="--", linewidth=2, color="red", label=f"u={u:.3f}")
    x_tail = np.linspace(u, x.max() * 1.1, 200)
    y_tail = x_tail - u
    gpd_density = genpareto.pdf(y_tail, xi, loc=0, scale=sigma)
    axes[0].plot(x_tail, p_u * gpd_density, linewidth=2, label="GPD tail")
    axes[0].set_title(f"{ticker} — Std Losses with GPD Tail")
    axes[0].set_xlabel("Standardized Loss")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    # Exceedances only
    y_grid = np.linspace(0, excesses.max() * 1.1, 300)
    pdf = genpareto.pdf(y_grid, xi, loc=0, scale=sigma)
    axes[1].hist(excesses, bins=30, density=True, alpha=0.7, edgecolor="black")
    axes[1].plot(y_grid, pdf, linewidth=2, label=f"GPD (xi={xi:.3f})")
    axes[1].set_title(f"{ticker} — GPD Fit to Exceedances (POT)")
    axes[1].set_xlabel("Excess Y = X - u")
    axes[1].set_ylabel("Density")
    axes[1].legend()

    plt.tight_layout()
    return fig


def plot_evt_dynamic_risk(loss_proxy, var_l, es_l, ticker):
    """
    Time series of losses with EVT-GARCH VaR and ES overlays.
    From Lecture 9.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    idx = loss_proxy.index.intersection(var_l.index)
    loss_aligned = loss_proxy.loc[idx]

    ax.plot(loss_aligned.index, loss_aligned.values, linewidth=0.7, label="Losses $L_t = -R_t$")
    ax.plot(var_l.index, var_l.values, linewidth=1.5, label="EVT-GARCH VaR")
    ax.plot(es_l.index, es_l.values, linewidth=1.5, label="EVT-GARCH ES")

    ax.set_title(f"{ticker} — Losses with Dynamic EVT-GARCH VaR and ES")
    ax.set_xlabel("Date")
    ax.set_ylabel("Loss (%)")
    ax.legend()

    plt.tight_layout()
    return fig


def plot_threshold_diagnostics(std_losses, gpd_params, ticker):
    """
    2x1 subplot: histogram with threshold, time series with exceedances highlighted.
    From Lecture 9.
    """
    x = pd.Series(std_losses).dropna().astype(float)
    u = gpd_params["u"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    # Histogram
    axes[0].hist(x.values, bins=40, density=True, edgecolor="black")
    axes[0].axvline(u, linestyle="--", linewidth=2, color="red")
    axes[0].set_title(f"{ticker} — Histogram of Standardized Losses")
    axes[0].set_xlabel("Standardized Loss")
    axes[0].set_ylabel("Density")

    # Time series with exceedances
    axes[1].plot(x.index, x.values, linewidth=0.9)
    exceed_mask = x > u
    axes[1].scatter(x.index[exceed_mask], x[exceed_mask], s=18, color="red", zorder=5)
    axes[1].axhline(u, linestyle="--", linewidth=2, color="red")
    axes[1].set_title(f"{ticker} — Exceedances Above Threshold u={u:.3f}")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Standardized Loss")

    plt.tight_layout()
    return fig
