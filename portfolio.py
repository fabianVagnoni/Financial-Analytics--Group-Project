"""
Two-asset and N-asset portfolio optimization.
Source: Lectures 10-11.
"""

import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt

from config import HOLDING_PERIOD, RISK_FREE_RATE, FRONTIER_POINTS


# ── Portfolio Input Statistics ────────────────────────────────────────

def compute_portfolio_inputs(returns_df, holding_period=HOLDING_PERIOD):
    """
    Compute daily and scaled statistics for portfolio optimization.
    Returns dict with daily and scaled (holding_period) mu, cov, corr, std.
    From Lectures 10-11.
    """
    mu_daily = returns_df.mean()
    cov_daily = returns_df.cov()
    corr_daily = returns_df.corr()
    std_daily = returns_df.std()

    mu_scaled = mu_daily * holding_period
    cov_scaled = cov_daily * holding_period
    std_scaled = std_daily * np.sqrt(holding_period)

    return {
        "mu_daily": mu_daily,
        "cov_daily": cov_daily,
        "corr_daily": corr_daily,
        "std_daily": std_daily,
        "mu_scaled": mu_scaled,
        "cov_scaled": cov_scaled,
        "std_scaled": std_scaled,
    }


def plot_correlation_heatmap(corr_matrix, tickers):
    """
    Annotated correlation heatmap.
    From Lecture 11.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1)

    ax.set_xticks(range(len(tickers)))
    ax.set_yticks(range(len(tickers)))
    ax.set_xticklabels(tickers, rotation=45, ha="right")
    ax.set_yticklabels(tickers)

    # Annotate cells
    for i in range(len(tickers)):
        for j in range(len(tickers)):
            ax.text(j, i, f"{corr_matrix.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=10,
                    color="white" if abs(corr_matrix.values[i, j]) > 0.6 else "black")

    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlation Matrix")
    plt.tight_layout()
    return fig


# ── N-Asset Optimization (cvxpy) ─────────────────────────────────────

def compute_mvp(mu, cov, tickers):
    """
    Minimum Variance Portfolio using cvxpy (long-only).
    Returns dict with weights, expected_return, std_dev.
    From Lecture 11.
    """
    n = len(mu)
    w = cp.Variable(n)

    objective = cp.Minimize(cp.quad_form(w, cov))
    constraints = [cp.sum(w) == 1, w >= 0]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS)

    weights = w.value
    exp_ret = float(mu.values @ weights)
    std_dev = float(np.sqrt(weights @ cov.values @ weights))

    return {
        "weights": pd.Series(weights, index=tickers),
        "expected_return": exp_ret,
        "std_dev": std_dev,
    }


def compute_tangency(mu, cov, rf=RISK_FREE_RATE, tickers=None):
    """
    Tangency Portfolio (max Sharpe) using cvxpy.
    Returns dict with weights, expected_return, std_dev, sharpe_ratio.
    From Lecture 11.
    """
    n = len(mu)
    mu_exc = mu.values - rf

    w = cp.Variable(n)
    objective = cp.Maximize(mu_exc @ w)
    constraints = [
        cp.quad_form(w, cov) <= 1,
        cp.sum(w) == 1,
        w >= 0,
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS)

    weights = w.value
    # Normalize weights to sum to 1 (solver might have slight deviation)
    weights = weights / weights.sum()

    exp_ret = float(mu.values @ weights)
    std_dev = float(np.sqrt(weights @ cov.values @ weights))
    sharpe = (exp_ret - rf) / std_dev if std_dev > 0 else 0.0

    return {
        "weights": pd.Series(weights, index=tickers),
        "expected_return": exp_ret,
        "std_dev": std_dev,
        "sharpe_ratio": sharpe,
    }


def compute_efficient_frontier(mu, cov, n_points=FRONTIER_POINTS):
    """
    Sweep target returns and solve min-variance for each.
    Returns (risks_list, returns_list, weights_list).
    From Lecture 11.
    """
    target_returns = np.linspace(mu.min(), mu.max(), n_points)
    risks = []
    returns_out = []
    weights_list = []

    n = len(mu)
    for target in target_returns:
        w = cp.Variable(n)
        objective = cp.Minimize(cp.quad_form(w, cov))
        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            mu.values @ w == target,
        ]

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.SCS)

        if problem.status in ("optimal", "optimal_inaccurate") and w.value is not None:
            wv = w.value
            risk = float(np.sqrt(wv @ cov.values @ wv))
            risks.append(risk)
            returns_out.append(target)
            weights_list.append(wv)

    return risks, returns_out, weights_list


def compute_efficient_frontier_and_tangency(
    rf, mean_returns, cov_matrix, n_points=50
):
    """
    Combined: efficient frontier + MVP + tangency in one call.
    Returns dict with frontier data and key portfolios.
    LIFTED from Lecture 11 cell 11.
    """
    target_returns = np.linspace(mean_returns.min(), mean_returns.max(), n_points)
    risks = []
    sharpe_ratios = []
    efficient_portfolios = []

    min_var_risk, min_var_return, min_var_weights = None, None, None

    for target_return in target_returns:
        n = len(mean_returns)
        w = cp.Variable(n)
        risk = cp.quad_form(w, cov_matrix)
        objective = cp.Minimize(risk)
        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            mean_returns.values @ w == target_return,
        ]

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.SCS)

        if problem.status not in ("optimal", "optimal_inaccurate") or w.value is None:
            continue

        optimal_weights = w.value
        portfolio_risk = float(np.sqrt(optimal_weights @ cov_matrix.values @ optimal_weights))
        risks.append(portfolio_risk)
        efficient_portfolios.append(optimal_weights)

        sharpe_ratio = (target_return - rf) / portfolio_risk if portfolio_risk > 0 else 0
        sharpe_ratios.append(sharpe_ratio)

        if min_var_risk is None or portfolio_risk < min_var_risk:
            min_var_risk = portfolio_risk
            min_var_return = target_return
            min_var_weights = optimal_weights

    max_sharpe_idx = int(np.argmax(sharpe_ratios))
    tangency_weights = efficient_portfolios[max_sharpe_idx]
    tangency_risk = risks[max_sharpe_idx]
    tangency_return = target_returns[max_sharpe_idx] if len(risks) == len(target_returns) else float(mean_returns.values @ tangency_weights)
    tangency_sharpe = sharpe_ratios[max_sharpe_idx]

    return {
        "risks": risks,
        "returns": [float(r) for r in target_returns[:len(risks)]],
        "min_var_portfolio": (min_var_risk, min_var_return, min_var_weights),
        "tangency_portfolio": (tangency_risk, tangency_return, tangency_weights, tangency_sharpe),
    }


# ── Plotting ─────────────────────────────────────────────────────────

def plot_efficient_frontier(
    frontier_std, frontier_ret, mvp, tangency, individual_assets, rf,
    title="Efficient Frontier",
):
    """
    Full frontier plot with assets, MVP, tangency, and CAL.
    From Lecture 11.
    """
    frontier_std = np.array(frontier_std)
    frontier_ret = np.array(frontier_ret)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Split into efficient / inefficient
    mvp_ret = mvp["expected_return"]
    below = frontier_ret < mvp_ret
    above = frontier_ret >= mvp_ret

    ax.plot(frontier_std[below], frontier_ret[below], color="gold",
            linewidth=2, label="Inefficient Portfolios")
    ax.plot(frontier_std[above], frontier_ret[above], color="blue",
            linewidth=2, label="Efficient Frontier")

    # CAL from rf through tangency
    tang_std = tangency["std_dev"]
    tang_ret = tangency["expected_return"]
    cal_x = np.linspace(0, tang_std * 1.2, 100)
    cal_y = rf + ((tang_ret - rf) / tang_std) * cal_x
    ax.plot(cal_x, cal_y, color="black", linestyle="dashed", linewidth=1,
            label="Capital Allocation Line")

    # Individual assets
    colors = plt.cm.Set1(np.linspace(0, 1, len(individual_assets)))
    for (ticker, (std_i, ret_i)), c in zip(individual_assets.items(), colors):
        ax.scatter(std_i, ret_i, color=c, marker="o", s=80, label=ticker, zorder=5)

    # MVP
    ax.scatter(mvp["std_dev"], mvp["expected_return"], color="purple",
               marker="D", s=120, label="Min Variance Portfolio", zorder=6)

    # Tangency
    ax.scatter(tang_std, tang_ret, color="orange", marker="*", s=220,
               label="Tangency Portfolio", zorder=6)

    # Risk-free
    ax.scatter(0, rf, color="black", marker="x", s=100, label="Risk-Free Rate", zorder=6)

    ax.set_xlabel("Risk (Standard Deviation)")
    ax.set_ylabel("Expected Return")
    ax.set_title(title)
    ax.legend(fontsize="small", loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_weight_comparison(mvp_weights, tangency_weights, tickers):
    """
    Side-by-side bar charts: MVP weights vs Tangency weights.
    From Lecture 11.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(tickers))
    width = 0.6

    axes[0].bar(x, mvp_weights, width, color="steelblue")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tickers, rotation=45, ha="right")
    axes[0].set_title("Minimum Variance Portfolio — Weights")
    axes[0].set_ylabel("Weight")
    axes[0].set_ylim(0, 1)

    axes[1].bar(x, tangency_weights, width, color="darkorange")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tickers, rotation=45, ha="right")
    axes[1].set_title("Tangency Portfolio — Weights")
    axes[1].set_ylabel("Weight")
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    return fig


def compare_portfolio_vs_individual_risk(individual_stds, mvp_std, tangency_std, tickers):
    """
    Table comparing individual asset std devs vs portfolio std devs.
    From Lecture 11.
    """
    data = {"Asset": list(tickers) + ["MVP", "Tangency"],
            "Std Dev": list(individual_stds) + [mvp_std, tangency_std]}
    return pd.DataFrame(data)
