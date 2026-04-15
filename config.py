"""
Central configuration for the Financial Analytics project.
All tickers, date ranges, and tunable parameters in one place.
"""

# ── Tickers ──────────────────────────────────────────────────────────
TICKERS = ["SBUX", "PEP", "TSLA", "AMZN", "HESAY", "HOOD"]

# ── Date ranges ──────────────────────────────────────────────────────
START_DATE = "2023-01-01"
END_DATE = "2025-12-31"

# Optional Q1 2026 evaluation
START_DATE_Q1_2026 = "2026-01-01"
END_DATE_Q1_2026 = "2026-03-31"

# ── EDA parameters ───────────────────────────────────────────────────
ROLLING_WINDOW = 30
ACF_LAGS = 40

# ── AR model selection ───────────────────────────────────────────────
AR_MAX_LAGS = 15

# ── ARCH/GARCH grid search bounds ────────────────────────────────────
ARCH_MAX_Q = 6
GARCH_MAX_P = 3
GARCH_MAX_Q = 3
AR_MAX_FOR_VOL = 5

# ── VaR / ES confidence levels ───────────────────────────────────────
VAR_LEVELS = [0.90, 0.95, 0.975, 0.99, 0.995, 0.999]

# ── EVT parameters ───────────────────────────────────────────────────
EVT_THRESHOLD_QUANTILE = 0.95
EVT_ALPHA = 0.99

# ── Bootstrap parameters ─────────────────────────────────────────────
BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 123
BOOTSTRAP_ALPHA = 0.05

# ── Portfolio parameters ─────────────────────────────────────────────
HOLDING_PERIOD = 21          # trading days (approx 1 month)
RISK_FREE_RATE = 0.01        # 21-day risk-free log-return
FRONTIER_POINTS = 100

# ── Matplotlib style ─────────────────────────────────────────────────
PLOT_STYLE = {
    "figure.dpi": 110,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
}
