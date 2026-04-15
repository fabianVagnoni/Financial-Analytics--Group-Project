"""
Distribution fitting (Normal vs Student-t), KS tests, and bootstrap CIs.
Source: Lectures 3-4.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import norm, probplot

from config import BOOTSTRAP_B, BOOTSTRAP_SEED, BOOTSTRAP_ALPHA


def fit_normal_and_t(log_ret):
    """
    Fit Normal and Student-t distributions to log-returns.
    Returns dict with fitted parameters.
    From Lecture 4.
    """
    vals = log_ret.dropna().values.astype(float)
    mu_hat, sig_hat = norm.fit(vals)
    df_hat, loc_hat, scale_hat = stats.t.fit(vals)

    return {
        "normal": {"mu": mu_hat, "sigma": sig_hat},
        "t": {"df": df_hat, "loc": loc_hat, "scale": scale_hat},
    }


def plot_fitted_distributions(log_ret, ticker):
    """
    2x2 subplot: histogram + Normal, histogram + Student-t, QQ Normal, QQ Student-t.
    From Lecture 4.
    """
    vals = log_ret.dropna().values.astype(float)
    params = fit_normal_and_t(log_ret)
    mu, sig = params["normal"]["mu"], params["normal"]["sigma"]
    df_t, loc_t, scale_t = params["t"]["df"], params["t"]["loc"], params["t"]["scale"]

    x_grid = np.linspace(vals.min(), vals.max(), 300)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Histogram + Normal
    axes[0, 0].hist(vals, bins=50, density=True, alpha=0.6, color="steelblue", edgecolor="black")
    axes[0, 0].plot(x_grid, norm.pdf(x_grid, mu, sig), "r-", lw=1.5, label="Normal fit")
    axes[0, 0].set_title(f"{ticker} — Histogram + Normal Fit")
    axes[0, 0].legend()

    # Histogram + Student-t
    axes[0, 1].hist(vals, bins=50, density=True, alpha=0.6, color="steelblue", edgecolor="black")
    axes[0, 1].plot(x_grid, stats.t.pdf(x_grid, df_t, loc_t, scale_t), "r-", lw=1.5,
                    label=f"Student-t (df={df_t:.2f})")
    axes[0, 1].set_title(f"{ticker} — Histogram + Student-t Fit")
    axes[0, 1].legend()

    # QQ Normal
    probplot(vals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title(f"{ticker} — QQ Plot vs Normal")

    # QQ Student-t
    theoretical_q = stats.t.ppf(
        np.linspace(0.001, 0.999, len(vals)), df_t, loc_t, scale_t
    )
    sample_q = np.sort(vals)
    axes[1, 1].scatter(theoretical_q, sample_q, s=5, alpha=0.5)
    lims = [min(theoretical_q.min(), sample_q.min()),
            max(theoretical_q.max(), sample_q.max())]
    axes[1, 1].plot(lims, lims, "r--", linewidth=1)
    axes[1, 1].set_xlabel("Theoretical Quantiles (Student-t)")
    axes[1, 1].set_ylabel("Sample Quantiles")
    axes[1, 1].set_title(f"{ticker} — QQ Plot vs Student-t (df={df_t:.2f})")

    plt.suptitle(f"{ticker} — Normal vs Student-t Comparison", fontsize=14, y=1.01)
    plt.tight_layout()
    return fig


def ks_test_comparison(log_ret):
    """
    KS test of log-returns against fitted Normal and fitted Student-t.
    Returns DataFrame with results.
    From Lecture 4.
    """
    vals = log_ret.dropna().values.astype(float)
    params = fit_normal_and_t(log_ret)

    ks_norm_stat, ks_norm_p = stats.kstest(
        vals, "norm", args=(params["normal"]["mu"], params["normal"]["sigma"])
    )
    ks_t_stat, ks_t_p = stats.kstest(
        vals, "t", args=(params["t"]["df"], params["t"]["loc"], params["t"]["scale"])
    )

    rows = [
        ("Normal", ks_norm_stat, ks_norm_p, "Reject" if ks_norm_p < 0.05 else "Fail to reject"),
        ("Student-t", ks_t_stat, ks_t_p, "Reject" if ks_t_p < 0.05 else "Fail to reject"),
    ]
    return pd.DataFrame(rows, columns=["Distribution", "KS Statistic", "p-value", "Decision (5%)"])


def compute_tail_quantiles_and_probabilities(log_ret):
    """
    Empirical tail quantiles q(0.01) and q(0.001), plus model-implied
    tail probabilities at those thresholds under Normal and Student-t.
    From Lecture 4.
    """
    vals = log_ret.dropna().values.astype(float)
    params = fit_normal_and_t(log_ret)
    mu, sig = params["normal"]["mu"], params["normal"]["sigma"]
    df_t, loc_t, scale_t = params["t"]["df"], params["t"]["loc"], params["t"]["scale"]

    q_01 = float(np.percentile(vals, 1))
    q_001 = float(np.percentile(vals, 0.1))

    return {
        "q_0.01": q_01,
        "q_0.001": q_001,
        "P_normal_below_q01": float(norm.cdf(q_01, mu, sig)),
        "P_t_below_q01": float(stats.t.cdf(q_01, df_t, loc_t, scale_t)),
        "P_normal_below_q001": float(norm.cdf(q_001, mu, sig)),
        "P_t_below_q001": float(stats.t.cdf(q_001, df_t, loc_t, scale_t)),
    }


def bootstrap_df_and_quantiles(
    log_ret,
    B=BOOTSTRAP_B,
    alpha=BOOTSTRAP_ALPHA,
    seed=BOOTSTRAP_SEED,
    make_plots=True,
):
    """
    Non-parametric bootstrap for Student-t df and tail quantiles.
    Returns dict with df_hat, q_hat, bootstrap arrays, and CIs.
    From Lecture 4.
    """
    vals = log_ret.dropna().values.astype(float)
    vals = vals[np.isfinite(vals)]
    n = len(vals)

    # Point estimates
    df_hat, loc_hat, scale_hat = stats.t.fit(vals)
    q_01 = np.percentile(vals, 1)
    q_001 = np.percentile(vals, 0.1)

    rng = np.random.default_rng(seed)

    boot_df = np.empty(B)
    boot_q01 = np.empty(B)
    boot_q001 = np.empty(B)

    for b in range(B):
        sample = rng.choice(vals, size=n, replace=True)
        try:
            df_b, _, _ = stats.t.fit(sample)
        except Exception:
            df_b = np.nan
        boot_df[b] = df_b
        boot_q01[b] = np.percentile(sample, 1)
        boot_q001[b] = np.percentile(sample, 0.1)

    def _pct_ci(arr, alpha):
        lo = 100 * (alpha / 2)
        hi = 100 * (1 - alpha / 2)
        return np.nanpercentile(arr, [lo, hi])

    ci_df = _pct_ci(boot_df, alpha)
    ci_q01 = _pct_ci(boot_q01, alpha)
    ci_q001 = _pct_ci(boot_q001, alpha)

    result = {
        "df_hat": df_hat,
        "q_01": q_01,
        "q_001": q_001,
        "ci_df": ci_df,
        "ci_q01": ci_q01,
        "ci_q001": ci_q001,
        "boot_df": boot_df,
        "boot_q01": boot_q01,
        "boot_q001": boot_q001,
    }

    if make_plots:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].hist(boot_df[np.isfinite(boot_df)], bins=40, density=True, alpha=0.7)
        axes[0].axvline(df_hat, color="red", linestyle="--", label=f"df={df_hat:.2f}")
        axes[0].set_title("Bootstrap: Student-t df")
        axes[0].legend()

        axes[1].hist(boot_q01, bins=40, density=True, alpha=0.7)
        axes[1].axvline(q_01, color="red", linestyle="--", label=f"q(0.01)={q_01:.4f}")
        axes[1].set_title("Bootstrap: q(0.01)")
        axes[1].legend()

        axes[2].hist(boot_q001, bins=40, density=True, alpha=0.7)
        axes[2].axvline(q_001, color="red", linestyle="--", label=f"q(0.001)={q_001:.4f}")
        axes[2].set_title("Bootstrap: q(0.001)")
        axes[2].legend()

        plt.tight_layout()
        result["figure"] = fig

    return result
