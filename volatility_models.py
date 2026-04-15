"""
ARCH, GARCH, and GJR-GARCH model selection, fitting, and diagnostics.
Source: Lectures 6-7.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t as sp_t
from statsmodels.graphics.tsaplots import plot_acf
from arch import arch_model

from config import AR_MAX_FOR_VOL, ARCH_MAX_Q, GARCH_MAX_P, GARCH_MAX_Q


def search_arch(log_ret, ar_max=AR_MAX_FOR_VOL, arch_max_q=ARCH_MAX_Q, dist="t"):
    """
    Grid search over AR(p) + ARCH(q) combinations.
    Returns ((best_ar, best_q), results DataFrame).
    From Lecture 6.
    """
    ret = 100 * log_ret.dropna()
    best_bic = np.inf
    best_order = (0, 1)
    rows = []

    for p in range(ar_max + 1):
        for q in range(1, arch_max_q + 1):
            try:
                m = arch_model(ret, mean="AR", lags=p, vol="ARCH", p=q, dist=dist)
                r = m.fit(disp="off")
                rows.append({"ar": p, "arch_q": q, "AIC": r.aic, "BIC": r.bic})
                if r.bic < best_bic:
                    best_bic = r.bic
                    best_order = (p, q)
            except Exception:
                rows.append({"ar": p, "arch_q": q, "AIC": np.nan, "BIC": np.nan})

    return best_order, pd.DataFrame(rows)


def search_garch(
    log_ret,
    ar_max=AR_MAX_FOR_VOL,
    garch_max_p=GARCH_MAX_P,
    garch_max_q=GARCH_MAX_Q,
    include_gjr=True,
    dist="t",
):
    """
    Grid search over AR(ar) + GARCH(p,o,q) combinations.
    If include_gjr is True, also searches with o=1 (GJR-GARCH).
    Returns (best_order_tuple, best_fitted_result).
    best_order_tuple = (ar, garch_p, garch_o, garch_q).
    From Lectures 7 and 9.
    """
    ret = 100 * log_ret.dropna()
    best_bic = np.inf
    best_order = (0, 1, 0, 1)
    best_result = None

    o_range = [0, 1] if include_gjr else [0]

    for ar in range(ar_max + 1):
        for p in range(1, garch_max_p + 1):
            for q in range(0, garch_max_q + 1):
                for o in o_range:
                    try:
                        if ar == 0:
                            m = arch_model(ret, mean="Constant", vol="Garch",
                                           p=p, o=o, q=q, dist=dist)
                        else:
                            m = arch_model(ret, mean="AR", lags=ar, vol="Garch",
                                           p=p, o=o, q=q, dist=dist)
                        r = m.fit(disp="off")
                        if r.bic < best_bic:
                            best_bic = r.bic
                            best_order = (ar, p, o, q)
                            best_result = r
                    except Exception:
                        pass

    return best_order, best_result


class GARCHAnalyzer:
    """
    Wraps a fitted GARCH-type model result and provides convenience
    properties/methods used by downstream analysis (risk_measures, EVT).
    """

    def __init__(self, fitted_result):
        self._res = fitted_result

    @property
    def result(self):
        return self._res

    @property
    def conditional_volatility(self):
        """Conditional std dev series (in same units as input, i.e. percent)."""
        return self._res.conditional_volatility

    @property
    def std_resid(self):
        """Standardized residuals series."""
        return self._res.std_resid

    @property
    def nu(self):
        """Degrees of freedom of fitted Student-t distribution."""
        return float(self._res.params["nu"])

    @property
    def mu_const(self):
        """Mean intercept from the fitted model."""
        for key in ["mu", "Const", "const"]:
            if key in self._res.params.index:
                return float(self._res.params[key])
        return 0.0

    def forecast_series(self):
        """
        One-step-ahead forecasts: (mu_r, sigma_r, scale_lambda).
        scale_lambda = sigma * sqrt((nu-2)/nu) for Student-t adjustment.
        From Lecture 8.
        """
        # start must be >= number of AR lags used in the mean model
        nobs = self._res.model.volatility.start
        fcst = self._res.forecast(start=nobs)
        mu_r = fcst.mean.iloc[:, -1]
        sigma_r = np.sqrt(fcst.variance.iloc[:, -1])
        nu = self.nu
        scale_lambda = sigma_r * np.sqrt((nu - 2) / nu)
        return mu_r, sigma_r, scale_lambda

    def plot_conditional_volatility(self, log_ret, ticker):
        """
        2x1 subplot: returns with +/-2 conditional SD bands,
        |returns| with conditional volatility overlay.
        From Lecture 7.
        """
        ret = 100 * log_ret
        cond_vol = self.conditional_volatility

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Returns with bands
        axes[0].plot(ret.index, ret.values, linewidth=0.5, color="blue", label="Log-Returns (%)")
        axes[0].fill_between(
            cond_vol.index, -2 * cond_vol.values, 2 * cond_vol.values,
            color="red", alpha=0.2, label="+/- 2 Cond. SD"
        )
        axes[0].set_title(f"{ticker} — Returns with Conditional Volatility Bands")
        axes[0].set_ylabel("Return (%)")
        axes[0].legend()

        # |Returns| vs conditional volatility
        axes[1].plot(ret.index, ret.abs().values, alpha=0.3, color="grey",
                     linewidth=0.7, label="|Log-Returns| (%)")
        axes[1].plot(cond_vol.index, cond_vol.values, color="red",
                     linewidth=1.2, label="Conditional Std Dev")
        axes[1].set_title(f"{ticker} — Conditional Volatility")
        axes[1].set_xlabel("Date")
        axes[1].set_ylabel("Volatility (%)")
        axes[1].legend()

        plt.tight_layout()
        return fig

    def plot_residual_diagnostics(self, ticker):
        """
        2x2 subplot: standardized residuals, ACF, ACF of squared, QQ vs t(nu).
        From Lecture 7.
        """
        std_resid = self.std_resid.dropna()
        nu = self.nu

        fig, axes = plt.subplots(2, 2, figsize=(13, 8))

        # Residuals time series
        axes[0, 0].plot(std_resid.index, std_resid.values, linewidth=0.5, color="black")
        axes[0, 0].axhline(0, color="red", linestyle="--", linewidth=0.8)
        axes[0, 0].set_title(f"{ticker} — Std Residuals")

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
        axes[1, 1].set_title(f"{ticker} — QQ vs Student-t (df={nu:.2f})")

        plt.suptitle(f"{ticker} — GARCH Residual Diagnostics", fontsize=14, y=1.01)
        plt.tight_layout()
        return fig

    def summary(self):
        """Return model summary string."""
        return str(self._res.summary())
