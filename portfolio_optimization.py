"""
================================================================================
 MARKOWITZ MEAN-VARIANCE PORTFOLIO OPTIMIZATION  --  a teaching implementation
================================================================================

WHAT THIS PROGRAM DOES
----------------------
Modern Portfolio Theory (Markowitz, 1952) says something deceptively simple:

    An investor should not care about an asset's risk in isolation.
    They should care about how that asset moves *together with everything
    else they own*.

Because of that, the ENTIRE input to the model is just two objects:

    mu     -- a vector of expected returns, one per asset          (N x 1)
    Sigma  -- a covariance matrix of returns                       (N x N)

and the entire output is a set of portfolio weights `w` that are "efficient":
no other portfolio offers more expected return for the same variance.

This script walks that idea end to end on a real portfolio, printing a
plain-English explanation of the math at every stage.

READ THE CAVEATS SECTION AT THE BOTTOM. The most important lesson in this file
is not "here are the optimal weights" -- it is "here is why you should not
trust the optimal weights very much."

Run with:   python portfolio_optimization.py
Outputs to: ./output/
"""

from __future__ import annotations

import os
import sys
import textwrap
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless: render straight to PNG, never open a window
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import scipy.optimize as sco
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
# ==  CONFIG BLOCK  --  EDIT EVERYTHING IN HERE, NOTHING BELOW  ===============
# =============================================================================

# -----------------------------------------------------------------------------
# >>> PASTE YOUR EXACT ROBINHOOD SHARE COUNTS HERE <<<
#
# Open Robinhood -> your positions -> copy the "Shares" number for each holding.
# Fractional shares are fine (Robinhood reports them to 6 decimals).
# The dollar value of each position is computed automatically as
# shares x latest adjusted close, and your CURRENT portfolio weights come from
# that -- you never have to type a weight or a dollar amount yourself.
#
# To remove a holding, delete its line. To add one, add a line -- everything
# downstream (charts, tables, optimizer) adapts to whatever is in this dict.
# -----------------------------------------------------------------------------
SHARES: dict[str, float] = {
    "SPY":   0.95,    # S&P 500 index ETF
    "VOOG":  4.4,     # S&P 500 Growth
    "SCHD":  1.0,     # US Dividend / Value
    "VXUS":  3.0,     # Total International ex-US
    "GLD":   0.57,    # Gold
    "GOOGL": 0.58,    # Alphabet -- single stock
    "NVDA":  0.74,    # Nvidia -- single stock
    "SMH":   1.13,    # Semiconductor sector ETF
    "WQTM":  2.0,     # Quantum computing ETF (launched late 2025 -- SHORT HISTORY)
    "TSLY":  8.66,    # YieldMax TSLA option-income ETF (launched ~2022)
}

# Human-readable names, used in chart labels and the written summary.
ASSET_NAMES: dict[str, str] = {
    "SPY":   "S&P 500 (SPY)",
    "VOOG":  "S&P 500 Growth (VOOG)",
    "SCHD":  "US Dividend/Value (SCHD)",
    "VXUS":  "Intl ex-US (VXUS)",
    "GLD":   "Gold (GLD)",
    "GOOGL": "Alphabet (GOOGL)",
    "NVDA":  "Nvidia (NVDA)",
    "SMH":   "Semiconductors (SMH)",
    "WQTM":  "Quantum ETF (WQTM)",
    "TSLY":  "TSLA Option-Income (TSLY)",
}

# --- Estimation window --------------------------------------------------------
LOOKBACK_YEARS = 3.0          # main analysis window, in years

# --- What to do about assets with short price histories -----------------------
# WQTM launched in late 2025; TSLY in ~2022. If you naively align all ten series
# to a common date range, the window collapses to the YOUNGEST asset's history,
# and every covariance in the model is then estimated from that short window.
#
#   "truncate"   -- keep all assets, shorten the window to the common overlap.
#                   Pro: nothing is thrown away. Con: if the youngest asset has
#                   6 months of data, ALL 45 covariance pairs are estimated from
#                   6 months, which is statistically terrible.
#
#   "drop_young" -- drop any asset with less than MIN_HISTORY_YEARS of data,
#                   then use the common window of what remains.
#                   Pro: covariance estimates stay trustworthy. Con: the dropped
#                   asset is invisible to the model even though you own it.
#
# There is no free lunch here. The script prints the consequences of your choice.
# Default is "drop_young" for this portfolio: WQTM launched in Oct 2025, so
# "truncate" collapses the whole analysis to ~0.8 years of data. Flip this one
# line to "truncate" to see that happen -- it is worth doing once.
HISTORY_POLICY = "drop_young"  # "truncate" | "drop_young"
MIN_HISTORY_YEARS = 2.0       # only used when HISTORY_POLICY == "drop_young"

# --- Risk-free rate -----------------------------------------------------------
# Used in the Sharpe ratio and to anchor the Capital Market Line. Set this to
# the current 3-month T-bill yield (annualized, as a decimal).
RISK_FREE_RATE = 0.043        # 4.3%

# --- Monte Carlo simulation ---------------------------------------------------
N_PORTFOLIOS = 50_000         # random long-only portfolios to simulate
# Dirichlet concentration parameters. alpha=1 is uniform on the simplex, but for
# N=10 assets that concentrates almost everything near equal weights -- the cloud
# comes out as a tiny blob. Mixing in small alphas (concentrated portfolios) and
# large alphas (very even portfolios) fills out the whole feasible region, which
# is what makes the efficient frontier visible as its upper-left boundary.
DIRICHLET_ALPHAS = (0.15, 0.4, 1.0, 4.0)

# --- Optimizer ----------------------------------------------------------------
N_FRONTIER_POINTS = 80        # target returns to sweep when tracing the frontier
MAX_WEIGHT = 0.25             # per-asset cap for the CONSTRAINED run (stretch goal)
                              # set to 1.0 to disable the cap entirely

# --- Stretch goals (set False to skip; each adds runtime) ---------------------
RUN_LOOKBACK_SENSITIVITY = True
LOOKBACK_SENSITIVITY_YEARS = (1.0, 2.0, 3.0)
RUN_SHRINKAGE = True          # Ledoit-Wolf shrinkage covariance
RUN_RESAMPLING = True         # bootstrap / resampled efficient frontier
N_RESAMPLES = 300             # bootstrap draws for the resampled frontier
RUN_WEIGHT_CAP = True         # compare capped vs uncapped optimum

# --- Output -------------------------------------------------------------------
OUTPUT_DIR = "output"
THEME = "light"               # "light" | "dark"  -- figure color scheme
FIG_DPI = 160
RANDOM_SEED = 42              # makes the Monte Carlo + bootstrap reproducible

TRADING_DAYS = 252            # trading days per year -- the annualization factor


# =============================================================================
# ==  SECTION 0  --  PLUMBING: color system, printing, output folder  =========
# =============================================================================
#
# Nothing conceptual here. Skip to SECTION 1 for the actual finance.

# A deliberately chosen, colorblind-validated palette. Categorical hues are
# assigned in a fixed order and never cycled; sequential encodings use a single
# hue light->dark; the correlation heatmap uses a warm/cool diverging pair with a
# neutral (not colored) midpoint so that "zero correlation" reads as "nothing".
PALETTE = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e",
        "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
        "series": ["#2a78d6", "#008300", "#e87ba4", "#eda100",
                   "#1baf7a", "#eb6834", "#4a3aa7", "#e34948"],
        "seq": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                "#2a78d6", "#256abf", "#184f95", "#0d366b"],
        "div_lo": "#2a78d6", "div_mid": "#f0efec", "div_hi": "#e34948",
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7",
        "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
        "series": ["#3987e5", "#008300", "#d55181", "#c98500",
                   "#199e70", "#d95926", "#9085e9", "#e66767"],
        "seq": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                "#2a78d6", "#256abf", "#184f95", "#0d366b"],
        "div_lo": "#3987e5", "div_mid": "#383835", "div_hi": "#e66767",
    },
}
C = PALETTE[THEME]

SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", C["seq"])
DIV_CMAP = LinearSegmentedColormap.from_list(
    "div_blue_red", [C["div_lo"], C["div_mid"], C["div_hi"]]
)

# Everything printed to the console is also captured here and written to
# output/analysis_log.txt so you can re-read the walkthrough later.
_TRANSCRIPT: list[str] = []


def say(text: str = "") -> None:
    """Print to console and capture for the log file."""
    print(text)
    _TRANSCRIPT.append(text)


def header(title: str) -> None:
    """A visually obvious section break."""
    say("")
    say("=" * 78)
    say(f"  {title}")
    say("=" * 78)


def explain(title: str, body: str) -> None:
    """
    Print a plain-English explanation block.

    This is the 'teaching' half of the program: after each computation we state
    what was computed, the formula, and how to read the result.
    """
    say("")
    say(f"--- {title} " + "-" * max(0, 74 - len(title)))
    for para in textwrap.dedent(body).strip().split("\n\n"):
        lines = para.split("\n")
        # A paragraph is PREFORMATTED if its first line is indented, or if any
        # continuation line is indented -- that covers formulas, numbered lists,
        # and hand-aligned tables, all of which carry meaning in their line
        # structure and must never be reflowed. Everything else is prose and
        # gets wrapped to a uniform width.
        preformatted = (
            lines[0][:1].isspace()
            or any(ln[:1].isspace() for ln in lines[1:] if ln.strip())
        )
        if preformatted:
            for ln in lines:
                say(ln.rstrip())
        else:
            say(textwrap.fill(para, width=78))
        say("")


def style_axes(ax, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    """
    Apply the chart chrome rules consistently:
      - recessive hairline grid (solid, never dashed -- dashing reads as
        'projection' or 'threshold' when it is just a grid)
      - no top/right spines
      - muted axis ink so the data is the loudest thing on the page
    """
    ax.set_facecolor(C["surface"])
    ax.grid(True, color=C["grid"], linewidth=0.6, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=C["muted"], labelsize=9, length=0)
    if title:
        ax.set_title(title, color=C["ink"], fontsize=13, fontweight="bold",
                     loc="left", pad=14)
    if xlabel:
        ax.set_xlabel(xlabel, color=C["ink2"], fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=C["ink2"], fontsize=10)


def new_figure(figsize=(11, 7)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(C["surface"])
    return fig, ax


def save_figure(fig, filename: str, note: str = "") -> None:
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight", facecolor=C["surface"])
    plt.close(fig)
    say(f"  [saved] {path}" + (f"  -- {note}" if note else ""))


def pct(x: float, decimals: int = 2) -> str:
    return f"{x * 100:.{decimals}f}%"


# =============================================================================
# ==  SECTION 1  --  DATA ACQUISITION  ========================================
# =============================================================================

def download_prices(tickers: list[str], years: float) -> pd.DataFrame:
    """
    Download daily AUTO-ADJUSTED closing prices.

    WHY auto_adjust=True IS NOT OPTIONAL HERE
    -----------------------------------------
    yfinance can give you two different "close" numbers:

      Close (raw)      -- the literal last traded price that day.
      Close (adjusted) -- the price rebuilt so that dividends and splits are
                          reinvested back into the series.

    The difference is the difference between PRICE return and TOTAL return.
    Markowitz needs total return, because a dollar of dividend is exactly as
    valuable to you as a dollar of price appreciation.

    For this specific portfolio the gap is enormous:
      - TSLY is an option-income fund that pays out most of its economics as
        distributions. Its raw price GRINDS DOWNWARD by construction. Using raw
        price would tell the optimizer TSLY has a large negative expected
        return, which is simply false.
      - SCHD is a dividend-focused fund (~3.5% yield).
      - GLD has no dividend but does bleed an expense ratio into the series.

    Feed raw prices into this model and the entire optimization is corrupted --
    not slightly, but qualitatively: it will short-change every income asset and
    over-favor every growth asset. auto_adjust=True fixes this at the source.
    """
    end = datetime.today()
    start = end - timedelta(days=int(365.25 * years) + 10)  # +10d padding

    say(f"  Downloading {len(tickers)} tickers from "
        f"{start:%Y-%m-%d} to {end:%Y-%m-%d} (auto-adjusted / total return)...")

    raw = yf.download(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,      # <-- the critical flag; see docstring above
        progress=False,
        actions=False,
    )
    if raw is None or len(raw) == 0:
        raise SystemExit("No data returned by yfinance. Check your connection.")

    # yfinance returns MultiIndex columns for >1 ticker, flat for exactly 1.
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].copy()
        prices.columns = [tickers[0]]

    # Preserve the config ordering, and surface tickers that returned nothing.
    prices = prices.reindex(columns=tickers)
    dead = [t for t in tickers if prices[t].notna().sum() == 0]
    if dead:
        say("")
        say(f"  !! WARNING: no data at all for {', '.join(dead)}.")
        say("     Possible causes: ticker delisted, renamed, or not on Yahoo.")
        say("     These are dropped from the entire analysis.")
        prices = prices.drop(columns=dead)

    if prices.shape[1] < 2:
        raise SystemExit("Need at least 2 assets with data to optimize.")

    return prices


def report_history_and_align(
    prices: pd.DataFrame, policy: str, min_years: float, years: float
) -> tuple[pd.DataFrame, list[str]]:
    """
    Show each asset's true available history, then align to a common window
    according to the configured policy.

    THE PROBLEM THIS SOLVES
    -----------------------
    A covariance matrix needs every asset observed on the SAME days. If one
    asset only started trading 8 months ago, then the naive "drop all rows with
    any NaN" alignment silently throws away years of data on the other nine, and
    every number in the model is now estimated from 8 months of history.

    That is not a rounding problem. Sample covariance error scales roughly with
    1/sqrt(T). Going from 756 trading days to 170 roughly DOUBLES the standard
    error on every one of the 45 pairwise covariances -- and the optimizer,
    which is a machine for finding and exploiting extreme values, will happily
    build a portfolio on top of that noise.
    """
    first_valid = {t: prices[t].first_valid_index() for t in prices.columns}
    n_obs = {t: int(prices[t].notna().sum()) for t in prices.columns}

    say("")
    say("  Available history per asset (within the requested window):")
    say("")
    say(f"    {'Ticker':<8}{'First date':<14}{'Obs':>7}{'Years':>9}   Assessment")
    say(f"    {'-' * 8}{'-' * 14}{'-' * 7}{'-' * 9}   {'-' * 26}")

    years_avail = {}
    for t in prices.columns:
        yrs = n_obs[t] / TRADING_DAYS
        years_avail[t] = yrs
        if yrs >= min_years:
            verdict = "ok"
        elif yrs >= 1.0:
            verdict = "SHORT -- treat with caution"
        else:
            verdict = "VERY SHORT -- unreliable"
        say(f"    {t:<8}{first_valid[t]:%Y-%m-%d}    {n_obs[t]:>7}{yrs:>9.2f}   {verdict}")

    # The common window is dictated by the LATEST start date across all assets.
    limiter = max(prices.columns, key=lambda t: first_valid[t])
    common_start = first_valid[limiter]
    full_start = min(first_valid.values())
    common_years = (prices.index[-1] - common_start).days / 365.25
    full_years = (prices.index[-1] - full_start).days / 365.25

    say("")
    if common_years < full_years - 0.25:
        say(f"  !! UNEQUAL HISTORY DETECTED")
        say(f"     Requested window ......... {years:.1f} years")
        say(f"     Oldest asset starts ...... {full_start:%Y-%m-%d} "
            f"({full_years:.2f} years available)")
        say(f"     Common window starts ..... {common_start:%Y-%m-%d} "
            f"({common_years:.2f} years) -- limited by {limiter}")
        say(f"     Aligning all assets would DISCARD "
            f"{full_years - common_years:.2f} years of data on the others.")
    else:
        say(f"  History is roughly balanced across assets "
            f"(common window {common_years:.2f} years).")

    # --- Apply the configured policy ----------------------------------------
    universe = list(prices.columns)

    if policy == "drop_young":
        young = [t for t in universe if years_avail[t] < min_years]
        if young:
            say("")
            say(f"  POLICY = 'drop_young': dropping {', '.join(young)} "
                f"(< {min_years:.1f} years of history).")
            say("     Tradeoff: the covariance matrix for the remaining assets is now")
            say("     estimated over a much longer window and is far more stable, but")
            say("     these holdings are invisible to the optimizer even though you")
            say("     own them. Your 'current portfolio' stats below are computed on")
            say("     the surviving assets, renormalized to sum to 1.")
            universe = [t for t in universe if t not in young]
        else:
            say("")
            say(f"  POLICY = 'drop_young': no asset falls below "
                f"{min_years:.1f} years -- nothing dropped.")
    else:
        say("")
        say("  POLICY = 'truncate': keeping all assets, shortening the window to")
        say("     the common overlap.")
        if common_years < min_years:
            say(f"     !! The resulting window is only {common_years:.2f} years.")
            say(f"        Every covariance below is estimated from ~"
                f"{int(common_years * TRADING_DAYS)} daily observations.")
            say("        With that little data the optimizer's output is closer to a")
            say("        description of recent noise than an estimate of the future.")
            say("        Consider re-running with HISTORY_POLICY = 'drop_young'.")

    # Align: restrict to the chosen universe, then drop any day where any
    # surviving asset is missing (holidays, halts, pre-inception).
    aligned = prices[universe].dropna(how="any")

    say("")
    say(f"  Final estimation window: {aligned.index[0]:%Y-%m-%d} -> "
        f"{aligned.index[-1]:%Y-%m-%d}")
    say(f"  {len(aligned)} trading days x {len(universe)} assets "
        f"({len(aligned) / TRADING_DAYS:.2f} years)")

    return aligned, universe


# =============================================================================
# ==  SECTION 2  --  RETURNS, EXPECTED RETURN VECTOR, COVARIANCE MATRIX  ======
# =============================================================================

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Daily SIMPLE returns:

        R_t = (P_t / P_{t-1}) - 1

    WHY SIMPLE AND NOT LOG RETURNS
    ------------------------------
    Log returns are r_t = ln(P_t / P_{t-1}). They are lovely for time
    aggregation, because they add up across time:  r_{0->T} = sum of r_t.

    But they do NOT add up across ASSETS. The return of a portfolio is

        R_p = sum_i ( w_i * R_i )        <- true for SIMPLE returns
        r_p != sum_i ( w_i * r_i )       <- FALSE for log returns

    Markowitz is built entirely on that first identity -- the whole model is
    "portfolio return is a weighted sum of asset returns, so portfolio variance
    is w' Sigma w". So simple returns are the theoretically correct input.

    In practice the two are nearly identical for daily data (ln(1+x) ~ x for
    small x), and log returns are often preferred for VOLATILITY estimation
    because they are better behaved in the tails. The difference is second-order
    here; the conceptual reason above is why this code uses simple returns.
    """
    return prices.pct_change().dropna(how="any")


def annualize_moments(returns: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Turn daily returns into the two objects Markowitz actually consumes.

        mu    = mean(daily returns) * 252          (annualized expected return)
        Sigma = cov(daily returns)  * 252          (annualized covariance)

    WHY 252, AND WHY LINEAR FOR BOTH
    --------------------------------
    252 is the approximate number of trading days in a year.

    Means scale LINEARLY with time: E[sum of 252 daily returns] = 252 * E[daily].

    Variances ALSO scale linearly with time -- but only under the assumption
    that daily returns are independent across days. Var(sum) = sum(Var) requires
    zero autocorrelation. That is roughly true for liquid equities and roughly
    false for anything with momentum or mean reversion. It is one of the model's
    quiet assumptions.

    Note that VOLATILITY (the square root of variance) therefore scales with
    sqrt(252), not 252 -- this is the famous "square root of time" rule.
    """
    mu = returns.mean() * TRADING_DAYS
    sigma = returns.cov() * TRADING_DAYS
    corr = returns.corr()
    return mu, sigma, corr


# =============================================================================
# ==  SECTION 3  --  PORTFOLIO MATH  ==========================================
# =============================================================================
#
# These three functions are the mathematical core of the entire file. Every
# chart and every optimization below is built out of just these.

def port_return(w: np.ndarray, mu: np.ndarray) -> float:
    """
    Expected portfolio return:      R_p = w' mu = sum_i w_i * mu_i

    Just a weighted average. Nothing subtle happens here -- all the interesting
    behavior of MPT lives in the variance function below.
    """
    return float(w @ mu)


def port_vol(w: np.ndarray, sigma: np.ndarray) -> float:
    """
    Portfolio volatility:           sigma_p = sqrt( w' Sigma w )

    THIS IS THE WHOLE IDEA. Expanded, the variance is:

        w' Sigma w  =  sum_i  w_i^2 * sigma_i^2                (own variances)
                    +  sum_{i != j}  w_i w_j * sigma_ij        (co-movements)

    The second sum is why diversification works. Each covariance term is
    sigma_ij = rho_ij * sigma_i * sigma_j. If two assets are uncorrelated
    (rho = 0) those cross terms vanish and the portfolio's variance is much
    less than the weighted-average variance of its holdings. If they are
    perfectly correlated (rho = 1) the cross terms are maximal and the whole
    expression collapses to (sum_i w_i sigma_i)^2 -- i.e. portfolio volatility
    is just the weighted average volatility, and you got NO diversification
    benefit at all despite holding many names.

    So: owning ten things is not diversification. Owning ten things with low
    pairwise correlation is diversification.
    """
    return float(np.sqrt(w @ sigma @ w))


def sharpe_ratio(w: np.ndarray, mu: np.ndarray, sigma: np.ndarray, rf: float) -> float:
    """
    Sharpe ratio:                   S = (R_p - r_f) / sigma_p

    "Units of excess return earned per unit of risk taken."

    The risk-free rate matters more than people expect. It sets the origin of
    the reward axis: an asset returning 4% when cash pays 4.3% has *negative*
    excess return, so more of it is strictly bad regardless of its volatility.
    Raising r_f pushes the optimizer toward higher-return / higher-risk
    portfolios (it takes more return to beat the hurdle); lowering it pushes
    toward safer ones.

    Geometrically, S is the SLOPE of the line from (0, r_f) to the portfolio's
    point in risk/return space. Maximizing Sharpe therefore means: find the
    point on the efficient frontier where a line drawn from the risk-free rate
    just barely touches it. That tangency point is the "tangency portfolio",
    and the line itself is the Capital Market Line.
    """
    vol = port_vol(w, sigma)
    if vol == 0:
        return 0.0
    return (port_return(w, mu) - rf) / vol


# =============================================================================
# ==  SECTION 4  --  MONTE CARLO: WHAT DOES THE FEASIBLE SET LOOK LIKE?  ======
# =============================================================================

def simulate_random_portfolios(
    mu: np.ndarray, sigma: np.ndarray, rf: float, n: int, rng: np.random.Generator
) -> pd.DataFrame:
    """
    Generate n random LONG-ONLY, FULLY-INVESTED portfolios and evaluate each.

    "Long-only, fully invested" means:
        w_i >= 0  for all i     (no short selling)
        sum(w) = 1              (all cash deployed, no leverage)

    Those two constraints define a geometric object called the unit simplex.
    The natural way to sample uniformly from it is the Dirichlet distribution,
    which by construction returns non-negative numbers summing to 1 -- no
    rejection sampling or renormalization needed.

    A subtlety worth knowing: Dirichlet(alpha=1,...,1) is uniform over the
    simplex, but in 10 dimensions "uniform" is heavily concentrated near the
    center (equal weights) -- the corners are a vanishingly small share of the
    volume. Sampling only that way produces a small blob that never reveals the
    frontier. Mixing several alpha values (see DIRICHLET_ALPHAS) gives both
    concentrated and evenly-spread portfolios, filling out the feasible region.

    This step is pure intuition-building: the optimizer in Section 5 finds the
    upper-left boundary of this cloud analytically. Seeing the cloud first makes
    the frontier feel like a discovered fact rather than a formula.
    """
    n_assets = len(mu)
    alphas = np.array(DIRICHLET_ALPHAS)
    # Split the sample budget evenly across the concentration parameters.
    chunks = []
    per = n // len(alphas)
    for i, a in enumerate(alphas):
        size = per if i < len(alphas) - 1 else n - per * (len(alphas) - 1)
        chunks.append(rng.dirichlet(np.full(n_assets, a), size=size))
    weights = np.vstack(chunks)

    # Vectorized evaluation of all n portfolios at once.
    rets = weights @ mu                                     # (n,)
    # einsum computes w' Sigma w row-by-row without building an n x n matrix.
    vars_ = np.einsum("ij,jk,ik->i", weights, sigma, weights)
    vols = np.sqrt(vars_)
    sharpes = (rets - rf) / vols

    return pd.DataFrame({"ret": rets, "vol": vols, "sharpe": sharpes}), weights


# =============================================================================
# ==  SECTION 5  --  FORMAL OPTIMIZATION (scipy)  =============================
# =============================================================================
#
# The Monte Carlo cloud shows us roughly where the good portfolios are. Now we
# solve for them exactly, using Sequential Least Squares Programming (SLSQP) --
# a constrained nonlinear optimizer that handles our equality constraint
# (weights sum to 1) and bounds (no shorting, optional max weight) directly.

def _bounds(n: int, max_w: float) -> tuple:
    """Long-only bounds, with an optional per-asset cap."""
    return tuple((0.0, max_w) for _ in range(n))


def _sum_to_one() -> dict:
    """Equality constraint: sum(w) - 1 = 0."""
    return {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}


def _solve(objective, n: int, max_w: float, extra_constraints=(), rng=None,
           n_restarts: int = 6) -> np.ndarray:
    """
    Minimize `objective` subject to long-only + sum-to-one (+ extras).

    WHY MULTIPLE RESTARTS
    ---------------------
    SLSQP is a LOCAL optimizer. The minimum-variance problem is convex, so any
    starting point converges to the same answer. But the negative-Sharpe
    objective is NOT convex, so a bad starting point can leave the solver stuck
    on a local optimum. Restarting from several random points and keeping the
    best result is cheap insurance.
    """
    cons = [_sum_to_one(), *extra_constraints]
    bnds = _bounds(n, max_w)

    # Start from equal weights, then from random simplex points.
    starts = [np.full(n, 1.0 / n)]
    if rng is not None:
        starts += [rng.dirichlet(np.full(n, 1.0)) for _ in range(n_restarts - 1)]

    best_w, best_f = None, np.inf
    for w0 in starts:
        # A feasible start matters when max_w < 1: equal weights may violate it.
        w0 = np.clip(w0, 0.0, max_w)
        w0 = w0 / w0.sum() if w0.sum() > 0 else np.full(n, 1.0 / n)
        res = sco.minimize(objective, w0, method="SLSQP",
                           bounds=bnds, constraints=cons,
                           options={"maxiter": 1000, "ftol": 1e-12})
        if res.success and res.fun < best_f:
            best_f, best_w = res.fun, res.x

    if best_w is None:
        raise RuntimeError("Optimizer failed to find any feasible solution.")

    # Clean up floating-point dust so weights print nicely and sum to exactly 1.
    w = np.clip(best_w, 0.0, max_w)
    return w / w.sum()


def min_variance_portfolio(sigma: np.ndarray, max_w: float = 1.0, rng=None) -> np.ndarray:
    """
    GLOBAL MINIMUM VARIANCE portfolio.

        minimize   w' Sigma w
        subject to sum(w) = 1,  0 <= w_i <= max_w

    Note what is ABSENT from that statement: mu. The GMV portfolio does not use
    expected returns at all.

    That is a genuinely important practical fact. Expected returns are by far
    the noisiest thing we estimate -- far noisier than covariances. A portfolio
    that ignores them entirely is therefore much more stable out-of-sample than
    the max-Sharpe portfolio, which leans on them heavily. Many practitioners
    use minimum-variance or risk-parity portfolios for exactly this reason:
    not because they believe variance is all that matters, but because they
    distrust their own return forecasts.
    """
    n = sigma.shape[0]
    return _solve(lambda w: w @ sigma @ w, n, max_w, rng=rng)


def max_sharpe_portfolio(mu: np.ndarray, sigma: np.ndarray, rf: float,
                         max_w: float = 1.0, rng=None) -> np.ndarray:
    """
    MAXIMUM SHARPE (tangency) portfolio.

        maximize   (w' mu - r_f) / sqrt(w' Sigma w)
        subject to sum(w) = 1,  0 <= w_i <= max_w

    Implemented as minimizing the NEGATIVE Sharpe ratio, since scipy minimizes.

    !! ESTIMATION-ERROR WARNING -- READ THIS ONE !!
    This is where mean-variance optimization earns its nickname,
    "error maximization". The objective rewards high mu and low sigma. Our mu
    is a noisy sample mean. So whichever asset happened to have a lucky run in
    the lookback window gets a high estimated mu, and the optimizer -- which has
    no idea that mu is an estimate -- treats that luck as a permanent feature
    and piles weight into it.

    The result is typically an extreme, concentrated portfolio that would have
    been perfect in hindsight and is unlikely to repeat. Section 8's lookback
    sensitivity test demonstrates this directly by re-solving on different
    windows and showing how violently the weights move.
    """
    n = len(mu)
    return _solve(lambda w: -sharpe_ratio(w, mu, sigma, rf), n, max_w, rng=rng)


def achievable_return_range(mu: np.ndarray, max_w: float) -> tuple[float, float]:
    """
    Lowest and highest expected return reachable under long-only + cap
    constraints.

    Without a cap this is simply [min(mu), max(mu)] -- put everything in the
    worst or best asset. With a cap of, say, 25%, you cannot hold more than 25%
    of the best asset, so the reachable maximum is a blend of the top few. We
    compute it greedily: fill the cap starting from the best asset until the
    weights sum to 1.
    """
    if max_w >= 1.0:
        return float(mu.min()), float(mu.max())

    def greedy(order: np.ndarray) -> float:
        remaining, total = 1.0, 0.0
        for i in order:
            take = min(max_w, remaining)
            total += take * mu[i]
            remaining -= take
            if remaining <= 1e-12:
                break
        return total

    hi = greedy(np.argsort(mu)[::-1])   # best assets first
    lo = greedy(np.argsort(mu))         # worst assets first
    return float(lo), float(hi)


def efficient_frontier(mu: np.ndarray, sigma: np.ndarray, n_points: int,
                       max_w: float = 1.0, rng=None) -> pd.DataFrame:
    """
    Trace the efficient frontier by solving a sequence of problems:

        for each target return R*:
            minimize   w' Sigma w
            subject to w' mu = R*,  sum(w) = 1,  0 <= w_i <= max_w

    Each solve answers "what is the least risky way to earn exactly R*?".
    Sweeping R* from low to high and plotting (sigma_p, R_p) traces out the
    curve. The upper half of that curve -- from the minimum-variance point
    upward -- is the EFFICIENT frontier. The lower half is technically the
    frontier too, but no rational investor would sit on it: for every point
    below the GMV portfolio there is a point directly above it with the same
    risk and more return.

    The frontier is a hyperbola in (sigma, R) space. Its curvature IS the
    diversification benefit -- a perfectly correlated universe would produce a
    straight line instead.
    """
    lo, hi = achievable_return_range(mu, max_w)
    targets = np.linspace(lo, hi, n_points)

    rows = []
    for target in targets:
        cons = [{"type": "eq", "fun": lambda w, t=target: w @ mu - t}]
        try:
            w = _solve(lambda w: w @ sigma @ w, len(mu), max_w,
                       extra_constraints=cons, rng=rng, n_restarts=3)
        except RuntimeError:
            continue  # target unreachable under the constraints; skip it
        # Verify the solver actually hit the target (SLSQP can quietly miss).
        if abs(w @ mu - target) > 1e-4:
            continue
        rows.append({"target": target, "ret": float(w @ mu),
                     "vol": float(np.sqrt(w @ sigma @ w)), "weights": w})

    return pd.DataFrame(rows)


# =============================================================================
# ==  SECTION 6  --  DIVERSIFICATION & RISK CONTRIBUTION DIAGNOSTICS  =========
# =============================================================================

def diversification_ratio(w: np.ndarray, sigma: np.ndarray) -> float:
    """
    Diversification Ratio:

        DR = ( sum_i w_i * sigma_i ) / sigma_p

    Numerator: the volatility your portfolio WOULD have if every holding moved
               in lockstep (correlation 1 everywhere).
    Denominator: the volatility it ACTUALLY has.

    So DR is literally "how much volatility did imperfect correlation cancel
    out for me?"

        DR = 1.0  -> no benefit at all; everything you own is the same bet.
        DR = 1.5  -> your portfolio is 33% less volatile than its parts.
        DR = 2.0+ -> strong diversification.

    A portfolio of ten highly-correlated tech names can easily score ~1.1
    despite holding ten tickers. A portfolio of four genuinely different assets
    can score 1.6. This is the number that catches "fake" diversification.
    """
    individual_vols = np.sqrt(np.diag(sigma))
    return float((w @ individual_vols) / port_vol(w, sigma))


def effective_n_assets(w: np.ndarray) -> float:
    """
    Effective number of assets (inverse Herfindahl index):

        N_eff = 1 / sum_i ( w_i^2 )

    If you hold N assets at equal weight, N_eff = N exactly.
    If you hold one asset, N_eff = 1.
    Anything in between measures how concentrated your DOLLARS are.

    Caveat worth internalizing: this only looks at weights, not at correlation.
    Ten equally-weighted S&P 500 index funds would score N_eff = 10 while being
    one single bet. Pair this number with the diversification ratio above --
    N_eff measures dollar spread, DR measures whether that spread bought you
    anything.
    """
    return float(1.0 / np.sum(w ** 2))


def risk_contributions(w: np.ndarray, sigma: np.ndarray) -> dict[str, np.ndarray]:
    """
    Decompose total portfolio risk into per-asset contributions.

    THE MATH
    --------
    Portfolio volatility is sigma_p = sqrt(w' Sigma w). Its partial derivative
    with respect to asset i is the MARGINAL risk contribution:

        MRC_i = d(sigma_p) / d(w_i) = (Sigma w)_i / sigma_p

    "If I add one more dollar of asset i, how much does total portfolio
    volatility rise?"

    Multiply by the weight to get the COMPONENT risk contribution:

        CRC_i = w_i * MRC_i

    Euler's theorem for homogeneous functions guarantees these sum exactly to
    the total:  sum_i CRC_i = sigma_p. So the percentages below are a genuine,
    exhaustive decomposition of your risk -- not an approximation.

    WHY THIS IS THE MOST USEFUL TABLE IN THE FILE
    ---------------------------------------------
    Your DOLLAR weights say how your money is split. Your RISK weights say how
    your outcomes are split. For a portfolio containing both a volatile,
    highly-correlated cluster (say NVDA + SMH + VOOG, which all move on the same
    semiconductor cycle) and some genuinely independent assets (gold, ex-US
    equity), these two columns can look completely different: a 20% dollar
    allocation to the tech cluster can easily account for 45% of the risk.

    Note the asymmetry: an asset's risk contribution depends on its covariance
    with THE PORTFOLIO, not just its own volatility. A volatile asset that is
    negatively correlated with everything else can even have a NEGATIVE risk
    contribution -- it is a hedge, and adding it reduces total risk.
    """
    vol = port_vol(w, sigma)
    mrc = (sigma @ w) / vol          # marginal: d sigma_p / d w_i
    crc = w * mrc                    # component: sums to sigma_p
    return {"mrc": mrc, "crc": crc, "pct": crc / vol, "total_vol": vol}


def portfolio_stats(w: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
                    rf: float) -> dict[str, float]:
    """Bundle every headline metric for one weight vector."""
    r = port_return(w, mu)
    v = port_vol(w, sigma)
    return {
        "Annual return": r,
        "Annual volatility": v,
        "Sharpe ratio": (r - rf) / v,
        "Diversification ratio": diversification_ratio(w, sigma),
        "Effective N assets": effective_n_assets(w),
        "Max weight": float(w.max()),
    }


# =============================================================================
# ==  SECTION 7  --  STRETCH GOAL: LEDOIT-WOLF SHRINKAGE COVARIANCE  ==========
# =============================================================================

def ledoit_wolf_shrinkage(returns: pd.DataFrame) -> tuple[np.ndarray, float]:
    """
    Ledoit-Wolf (2004) shrinkage toward a constant-correlation target.

    THE PROBLEM WITH THE SAMPLE COVARIANCE MATRIX
    ---------------------------------------------
    With N assets you must estimate N(N+1)/2 parameters -- for 10 assets that's
    55 numbers -- from T observations. When T is not enormous relative to N, the
    sample covariance matrix is badly conditioned: its extreme eigenvalues are
    biased outward (largest too large, smallest too small).

    That matters because the optimizer INVERTS this matrix (implicitly). The
    smallest eigenvalues, which are the most biased, dominate the inverse. The
    result: the optimizer finds "arbitrage-like" combinations of assets that
    look near-riskless in-sample and are nothing of the sort out-of-sample.

    THE FIX
    -------
    Blend the noisy-but-unbiased sample estimate S with a heavily-structured,
    biased-but-stable target F:

        Sigma_shrunk = delta * F + (1 - delta) * S

    Here F is the "constant correlation" matrix: keep each asset's own variance,
    but replace every pairwise correlation with the average correlation across
    all pairs. That is a strong assumption, deliberately -- it has almost no
    estimation error.

    delta is the shrinkage intensity, in [0, 1]. Ledoit and Wolf derive the
    delta that minimizes expected squared error analytically:

        delta* = (pi - rho) / gamma / T

        pi    = total estimation variance of the sample covariance entries
        rho   = covariance between the estimation errors of S and F
        gamma = how wrong the target F is (its bias)

    Read that formula as a bias-variance tradeoff made quantitative: shrink more
    when the sample estimate is noisy (large pi) and when the target is not too
    wrong (small gamma). A short lookback window drives pi up, which is exactly
    when you most want shrinkage.

    (sklearn.covariance.LedoitWolf implements the simpler scaled-identity
    target. The constant-correlation target below is the one Ledoit & Wolf
    recommend for stock returns, and is implemented here from the paper so you
    can see the mechanics rather than a black box.)

    Returns the ANNUALIZED shrunk covariance matrix and the chosen delta.
    """
    X = returns.values
    T, N = X.shape
    Xc = X - X.mean(axis=0)

    # Sample covariance, MLE convention (divide by T, as in the paper).
    S = (Xc.T @ Xc) / T
    var = np.diag(S).copy()
    sd = np.sqrt(var)

    # --- Target F: constant correlation ------------------------------------
    corr = S / np.outer(sd, sd)
    r_bar = (corr.sum() - N) / (N * (N - 1))     # mean off-diagonal correlation
    F = r_bar * np.outer(sd, sd)
    np.fill_diagonal(F, var)

    # --- pi: asymptotic variance of each sample covariance entry -----------
    Xc2 = Xc ** 2
    pi_mat = (Xc2.T @ Xc2) / T - S ** 2
    pi_hat = pi_mat.sum()

    # --- rho: covariance between errors in S and errors in F ---------------
    rho_diag = np.trace(pi_mat)
    # A[i, j] = E[x_i^3 x_j] - S_ii * S_ij  = theta_{ii,ij}
    A = ((Xc ** 3).T @ Xc) / T - var[:, None] * S
    np.fill_diagonal(A, 0.0)                      # diagonal handled by rho_diag
    theta_ii_ij = A
    theta_jj_ij = A.T
    rho_off = (r_bar / 2.0) * (
        (np.outer(1.0 / sd, sd) * theta_ii_ij).sum()   # sqrt(S_jj / S_ii)
        + (np.outer(sd, 1.0 / sd) * theta_jj_ij).sum()  # sqrt(S_ii / S_jj)
    )
    rho_hat = rho_diag + rho_off

    # --- gamma: squared misspecification of the target ---------------------
    gamma_hat = ((F - S) ** 2).sum()

    delta = 0.0 if gamma_hat <= 0 else (pi_hat - rho_hat) / gamma_hat / T
    delta = float(np.clip(delta, 0.0, 1.0))

    shrunk = delta * F + (1.0 - delta) * S
    return shrunk * TRADING_DAYS, delta


# =============================================================================
# ==  SECTION 8  --  STRETCH GOAL: RESAMPLED (BOOTSTRAPPED) OPTIMIZATION  =====
# =============================================================================

def resampled_max_sharpe(returns: pd.DataFrame, rf: float, n_resamples: int,
                         max_w: float, rng: np.random.Generator) -> pd.DataFrame:
    """
    Michaud-style resampled optimization.

    THE IDEA
    --------
    We only ever see ONE sample of history. The optimal weights we computed are
    a function of that one sample -- so they are themselves a random variable
    with a sampling distribution.

    To see that distribution, bootstrap: draw T days WITH REPLACEMENT from the
    actual return history, recompute mu and Sigma from the resampled data, and
    re-solve for max Sharpe. Repeat many times.

    Two things come out of this:

      1. The SPREAD of weights across resamples is a direct, honest measure of
         how much you should trust any single optimal weight. If NVDA's weight
         ranges from 0% to 60% depending on which days happened to get drawn,
         then "the optimizer says 34% NVDA" is not a meaningful statement.

      2. The AVERAGE of the resampled weights is itself a portfolio -- the
         "resampled efficient portfolio". It is typically far more diversified
         than the naive optimum, because assets that only win under some
         resamples get partial rather than all-or-nothing allocations. It tends
         to hold up better out of sample.

    This is arguably the single most useful diagnostic in the whole file: it
    turns "MPT is sensitive to inputs" from a warning into a measurement.
    """
    T = len(returns)
    R = returns.values
    n_assets = R.shape[1]
    all_w = np.zeros((n_resamples, n_assets))
    n_ok = 0

    for b in range(n_resamples):
        idx = rng.integers(0, T, size=T)          # bootstrap sample of days
        sample = R[idx]
        mu_b = sample.mean(axis=0) * TRADING_DAYS
        sig_b = np.cov(sample, rowvar=False) * TRADING_DAYS
        try:
            # Fewer restarts here -- we do this hundreds of times.
            w_b = _solve(lambda w: -(w @ mu_b - rf) / np.sqrt(w @ sig_b @ w),
                         n_assets, max_w, rng=rng, n_restarts=2)
            all_w[n_ok] = w_b
            n_ok += 1
        except RuntimeError:
            continue

    all_w = all_w[:n_ok]
    return pd.DataFrame({
        "mean": all_w.mean(axis=0),
        "std": all_w.std(axis=0),
        "p05": np.percentile(all_w, 5, axis=0),
        "p95": np.percentile(all_w, 95, axis=0),
    }, index=returns.columns)


# =============================================================================
# ==  SECTION 9  --  FIGURES  =================================================
# =============================================================================

def fig_price_history(prices: pd.DataFrame, outfile: str) -> None:
    """
    Growth of $1, total-return basis. Context, not analysis.

    Every series starts at 1.0 so the SHAPES are comparable regardless of share
    price. With 10 series we are past the point where distinct hues stay
    distinguishable, so the chart uses one recessive color for the pack and
    direct-labels each line at its endpoint -- identity comes from the label,
    not from color.
    """
    growth = prices / prices.iloc[0]
    fig, ax = new_figure((12, 7))

    finals = growth.iloc[-1].sort_values(ascending=False)
    for t in growth.columns:
        ax.plot(growth.index, growth[t], color=C["muted"], linewidth=1.2,
                alpha=0.55, zorder=2)

    # Highlight the best and worst performers -- the two the eye should catch.
    for t, color in ((finals.index[0], C["series"][1]),
                     (finals.index[-1], C["series"][7])):
        ax.plot(growth.index, growth[t], color=color, linewidth=2.0, zorder=3)

    # Direct labels at the right edge, nudged apart to avoid collisions.
    # Walk from the LOWEST endpoint upward, pushing each label up only as far as
    # needed to clear the one below it. Iterating in the other direction would
    # invert the stack and label every line with its neighbour's value.
    span = growth.values.max() - growth.values.min()
    ys, last_y = [], -np.inf
    for t in finals.sort_values().index:
        y = max(finals[t], last_y + span * 0.045)
        ys.append((t, y))
        last_y = y
    x_end = growth.index[-1]
    for t, y in ys:
        is_extreme = t in (finals.index[0], finals.index[-1])
        col = (C["series"][1] if t == finals.index[0]
               else C["series"][7] if t == finals.index[-1] else C["ink2"])
        # Draw the label at the collision-adjusted y, with a leader line back to
        # the series' true endpoint so the nudge never misleads.
        if abs(y - finals[t]) > 1e-9:
            ax.plot([x_end, x_end], [finals[t], y], color=C["grid"],
                    linewidth=0.8, zorder=1, clip_on=False)
        ax.annotate(f"{t}  {finals[t]:.2f}x", xy=(x_end, y),
                    xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, annotation_clip=False,
                    fontweight="bold" if is_extreme else "normal", color=col)

    ax.axhline(1.0, color=C["axis"], linewidth=0.8, zorder=1)
    style_axes(ax, "Growth of $1 (total return, dividends reinvested)",
               "", "Multiple of initial investment")
    ax.margins(x=0.02)
    fig.subplots_adjust(right=0.86)
    save_figure(fig, outfile, "context: how each holding actually performed")


def fig_heatmap(matrix: pd.DataFrame, title: str, outfile: str,
                fmt: str, center_zero: bool, cbar_label: str) -> None:
    """
    Annotated heatmap with a diverging colormap centered at zero.

    Diverging (warm/cool poles, NEUTRAL GRAY midpoint) is the right encoding
    here because the data has a meaningful zero with opposite signs either side:
    positive correlation and negative correlation are qualitatively different
    things, and zero means "nothing". A rainbow or single-hue ramp would hide
    the sign change, which is the most important feature of the matrix.
    """
    n = len(matrix)
    fig, ax = plt.subplots(figsize=(1.0 * n + 3.2, 0.85 * n + 2.4))
    fig.patch.set_facecolor(C["surface"])

    vmax = np.abs(matrix.values).max()
    sns.heatmap(
        matrix, annot=True, fmt=fmt, cmap=DIV_CMAP,
        center=0 if center_zero else None,
        vmin=-vmax if center_zero else None, vmax=vmax if center_zero else None,
        square=True, linewidths=2, linecolor=C["surface"],   # 2px surface gap
        cbar_kws={"label": cbar_label, "shrink": 0.72},
        annot_kws={"fontsize": 8.5}, ax=ax,
    )
    ax.set_title(title, color=C["ink"], fontsize=13, fontweight="bold",
                 loc="left", pad=14)
    ax.tick_params(colors=C["ink2"], labelsize=9.5, length=0)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors=C["muted"], labelsize=8, length=0)
    cbar.ax.yaxis.label.set_color(C["ink2"])
    cbar.outline.set_visible(False)
    save_figure(fig, outfile)


def fig_monte_carlo(cloud: pd.DataFrame, outfile: str) -> None:
    """
    The random-portfolio cloud, colored by Sharpe ratio.

    Sharpe is a continuous magnitude, so it gets a SEQUENTIAL single-hue ramp
    (light = low, dark = high) with a scale legend. The point of this chart is
    the SHAPE of the cloud: notice that it has a hard upper-left boundary. No
    matter how you mix these ten assets, you cannot get above and to the left of
    that edge. That edge is the efficient frontier, and the next figure computes
    it exactly.
    """
    fig, ax = new_figure((11, 7))
    sc = ax.scatter(cloud["vol"], cloud["ret"], c=cloud["sharpe"],
                    cmap=SEQ_CMAP, s=4, alpha=0.45, linewidths=0, zorder=2)
    cbar = fig.colorbar(sc, ax=ax, pad=0.015)
    cbar.set_label("Sharpe ratio", color=C["ink2"], fontsize=10)
    cbar.ax.tick_params(colors=C["muted"], labelsize=8, length=0)
    cbar.outline.set_visible(False)

    style_axes(ax, f"{len(cloud):,} random long-only portfolios",
               "Annualized volatility (risk)", "Annualized expected return")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    save_figure(fig, outfile, "the feasible set; note its upper-left edge")


def fig_efficient_frontier(cloud, frontier, assets, portfolios, rf, outfile,
                           frontier_capped=None):
    """
    THE MAIN CHART. Layers, from back to front:

      1. Monte Carlo cloud (recessive, sets the context)
      2. Individual assets, direct-labeled
      3. The efficient frontier curve
      4. The Capital Market Line from (0, r_f) through the tangency portfolio
      5. The four named portfolios, each a distinct hue AND a distinct marker
         AND a direct label -- identity never rests on color alone

    HOW TO READ IT
    --------------
    Up and to the LEFT is better (more return, less risk). The vertical distance
    from your current portfolio's marker to the frontier directly above it is
    the return you are leaving on the table at your current risk level. The
    horizontal distance to the frontier on your left is the risk you could shed
    without giving up any return.
    """
    fig, ax = new_figure((12.5, 8))

    # --- Layer 1: the cloud -------------------------------------------------
    ax.scatter(cloud["vol"], cloud["ret"], c=cloud["sharpe"], cmap=SEQ_CMAP,
               s=3, alpha=0.22, linewidths=0, zorder=1)

    # --- Layer 2: individual assets ----------------------------------------
    ax.scatter(assets["vol"], assets["ret"], s=42, c=C["muted"],
               edgecolors=C["surface"], linewidths=2, zorder=4, marker="o")
    for t, row in assets.iterrows():
        ax.annotate(t, (row["vol"], row["ret"]), xytext=(7, -3),
                    textcoords="offset points", fontsize=8.5,
                    color=C["ink2"], zorder=5)

    # --- Layer 3: the frontier ---------------------------------------------
    eff = frontier[frontier["ret"] >= frontier.loc[frontier["vol"].idxmin(), "ret"]]
    ax.plot(eff["vol"], eff["ret"], color=C["ink"], linewidth=2.0,
            zorder=6, label="Efficient frontier")
    if frontier_capped is not None and len(frontier_capped):
        effc = frontier_capped[
            frontier_capped["ret"]
            >= frontier_capped.loc[frontier_capped["vol"].idxmin(), "ret"]]
        ax.plot(effc["vol"], effc["ret"], color=C["series"][6], linewidth=2.0,
                zorder=6, label=f"Frontier, max {MAX_WEIGHT:.0%} per asset")

    # --- Layer 4: the Capital Market Line ----------------------------------
    tan = portfolios["Max Sharpe"]
    slope = (tan["ret"] - rf) / tan["vol"]           # = the Sharpe ratio
    x_max = max(cloud["vol"].max(), assets["vol"].max()) * 1.04
    xs = np.linspace(0, x_max, 50)
    ax.plot(xs, rf + slope * xs, color=C["muted"], linewidth=1.6, zorder=5,
            label="Capital Market Line")
    ax.scatter([0], [rf], s=40, c=C["muted"], edgecolors=C["surface"],
               linewidths=2, zorder=6)
    ax.annotate(f"risk-free  {rf:.1%}", (0, rf), xytext=(8, -14),
                textcoords="offset points", fontsize=8.5, color=C["muted"])

    # --- Layer 5: the named portfolios -------------------------------------
    # These four points sit close together, so each label gets a hand-picked
    # offset and alignment that fans them apart rather than a shared offset
    # that would stack them on top of one another.
    styles = {
        # name: (color, marker, size, label offset, ha)
        "Current (yours)": (C["series"][7], "*", 460, (16, -20), "left"),
        "Max Sharpe":      (C["series"][5], "D", 150, (-12, 14), "right"),
        "Min variance":    (C["series"][1], "s", 150, (16, -4), "left"),
        "Equal weight":    (C["series"][6], "^", 160, (-12, -20), "right"),
    }
    for name, (color, marker, size, offset, ha) in styles.items():
        if name not in portfolios:
            continue
        p = portfolios[name]
        ax.scatter(p["vol"], p["ret"], s=size, c=color, marker=marker,
                   edgecolors=C["surface"], linewidths=2, zorder=8, label=name)
        ax.annotate(name, (p["vol"], p["ret"]), xytext=offset, ha=ha,
                    textcoords="offset points", fontsize=9.5,
                    fontweight="bold", color=color, zorder=9)

    style_axes(ax, "Efficient frontier, your portfolio, and the feasible set",
               "Annualized volatility (risk)", "Annualized expected return")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    ax.set_xlim(left=0)
    # Bound the y-axis to the DATA, not to the Capital Market Line -- the CML is
    # unbounded above and, left alone, squashes the entire cloud and frontier
    # into the bottom third of the plot. Let it run off the top instead.
    ax.set_ylim(min(0.0, float(assets["ret"].min())) - 0.02,
                max(float(assets["ret"].max()), float(cloud["ret"].max())) * 1.10)
    # Legend goes upper-left: that region is empty (it lies above the frontier,
    # which is by definition unreachable), whereas lower-right holds real marks.
    leg = ax.legend(loc="upper left", frameon=True, fontsize=9.5,
                    facecolor=C["surface"], edgecolor=C["grid"])
    for text in leg.get_texts():
        text.set_color(C["ink2"])
    save_figure(fig, outfile, "THE MAIN CHART -- read this one first")


def fig_weights_comparison(weights: pd.DataFrame, outfile: str) -> None:
    """
    Grouped bars: your weights next to each optimized weight vector.

    Four series -> the first four categorical slots, in fixed order. A 2px
    surface gap separates adjacent bars rather than a drawn border.
    """
    fig, ax = new_figure((13, 6.5))
    n_groups, n_series = len(weights.index), len(weights.columns)
    x = np.arange(n_groups)
    width = 0.8 / n_series

    for i, col in enumerate(weights.columns):
        ax.bar(x + i * width - 0.4 + width / 2, weights[col], width * 0.92,
               label=col, color=C["series"][i], linewidth=0, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(weights.index, rotation=0, fontsize=9.5)
    ax.axhline(0, color=C["axis"], linewidth=0.8, zorder=2)
    style_axes(ax, "Portfolio weights: yours vs. what the optimizer wants",
               "", "Weight")
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    leg = ax.legend(frameon=True, fontsize=9.5, ncol=n_series,
                    facecolor=C["surface"], edgecolor=C["grid"])
    for text in leg.get_texts():
        text.set_color(C["ink2"])
    save_figure(fig, outfile, "how the optimizer would reshape your portfolio")


def fig_risk_contributions(tickers, weights, risk_pct, outfile) -> None:
    """
    Dollar weight vs. risk contribution, side by side, for the current portfolio.

    Two series -> categorical slots 1 and 2. The gap between the paired bars for
    a given ticker IS the finding: where the risk bar towers over the weight
    bar, that holding is punching above its dollar allocation in determining
    your outcomes.
    """
    order = np.argsort(risk_pct)[::-1]
    labels = [tickers[i] for i in order]
    w_sorted = np.array(weights)[order]
    r_sorted = np.array(risk_pct)[order]

    fig, ax = new_figure((12, 6.5))
    x = np.arange(len(labels))
    ax.bar(x - 0.20, w_sorted, 0.37, label="Dollar weight",
           color=C["series"][0], linewidth=0, zorder=3)
    ax.bar(x + 0.20, r_sorted, 0.37, label="Share of portfolio risk",
           color=C["series"][1], linewidth=0, zorder=3)

    # Direct-label only where the gap is large -- never a number on every bar.
    for i, (w_, r_) in enumerate(zip(w_sorted, r_sorted)):
        if abs(r_ - w_) > 0.04:
            ax.annotate(f"{r_ - w_:+.0%}", (i + 0.20, r_), xytext=(0, 5),
                        textcoords="offset points", ha="center", fontsize=8.5,
                        fontweight="bold", color=C["ink2"])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.axhline(0, color=C["axis"], linewidth=0.8, zorder=2)
    style_axes(ax, "Where your money is vs. where your risk is",
               "", "Share of portfolio")
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    leg = ax.legend(frameon=True, fontsize=9.5, facecolor=C["surface"],
                    edgecolor=C["grid"])
    for text in leg.get_texts():
        text.set_color(C["ink2"])
    save_figure(fig, outfile, "labels show risk share minus dollar share")


def fig_lookback_sensitivity(table: pd.DataFrame, outfile: str) -> None:
    """
    Max-Sharpe weights recomputed on different lookback windows.

    If mean-variance optimization were extracting a stable signal, these bars
    would be nearly identical. The degree to which they are NOT is a direct
    readout of how much of the "optimal" portfolio is an artifact of the
    particular window you happened to choose.
    """
    fig, ax = new_figure((13, 6.5))
    n_groups, n_series = len(table.index), len(table.columns)
    x = np.arange(n_groups)
    width = 0.8 / n_series
    for i, col in enumerate(table.columns):
        ax.bar(x + i * width - 0.4 + width / 2, table[col], width * 0.9,
               label=col, color=C["series"][i], linewidth=0, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(table.index, fontsize=9.5)
    ax.axhline(0, color=C["axis"], linewidth=0.8, zorder=2)
    style_axes(ax,
               "Same optimizer, same assets, different lookback window",
               "", "Max-Sharpe weight")
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    leg = ax.legend(frameon=True, fontsize=9.5, facecolor=C["surface"],
                    edgecolor=C["grid"], ncol=n_series)
    for text in leg.get_texts():
        text.set_color(C["ink2"])
    save_figure(fig, outfile, "instability here = estimation error, not signal")


def fig_shrinkage(frontier_naive, frontier_lw, w_naive, w_lw, tickers,
                  delta, outfile) -> None:
    """Two panels: frontier comparison, and the weights it implies."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2))
    fig.patch.set_facecolor(C["surface"])

    e_n = frontier_naive[frontier_naive["ret"]
                         >= frontier_naive.loc[frontier_naive["vol"].idxmin(), "ret"]]
    e_l = frontier_lw[frontier_lw["ret"]
                      >= frontier_lw.loc[frontier_lw["vol"].idxmin(), "ret"]]
    ax1.plot(e_n["vol"], e_n["ret"], color=C["series"][0], linewidth=2.0,
             label="Sample covariance")
    ax1.plot(e_l["vol"], e_l["ret"], color=C["series"][1], linewidth=2.0,
             label="Ledoit-Wolf shrunk")
    style_axes(ax1, "Efficient frontier: sample vs. shrunk covariance",
               "Annualized volatility", "Annualized expected return")
    ax1.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax1.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    leg1 = ax1.legend(frameon=True, fontsize=9.5, facecolor=C["surface"],
                      edgecolor=C["grid"])
    for t in leg1.get_texts():
        t.set_color(C["ink2"])

    x = np.arange(len(tickers))
    ax2.bar(x - 0.19, w_naive, 0.36, label="Sample covariance",
            color=C["series"][0], linewidth=0, zorder=3)
    ax2.bar(x + 0.19, w_lw, 0.36, label="Ledoit-Wolf shrunk",
            color=C["series"][1], linewidth=0, zorder=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(tickers, fontsize=9, rotation=45, ha="right")
    style_axes(ax2, f"Max-Sharpe weights  (shrinkage intensity = {delta:.3f})",
               "", "Weight")
    ax2.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    leg2 = ax2.legend(frameon=True, fontsize=9.5, facecolor=C["surface"],
                      edgecolor=C["grid"])
    for t in leg2.get_texts():
        t.set_color(C["ink2"])

    fig.tight_layout()
    save_figure(fig, outfile, "shrinkage pulls the frontier toward reality")


def fig_resampled(resampled: pd.DataFrame, w_naive: np.ndarray, outfile) -> None:
    """
    Naive optimum vs. bootstrap-averaged optimum, with 5th-95th percentile bars.

    The error bars are the headline. A weight whose 5-95 band spans 0% to 50%
    is not an allocation -- it is a coin flip that the optimizer reported to
    two decimal places.
    """
    fig, ax = new_figure((12.5, 6.5))
    x = np.arange(len(resampled))
    # Clip at zero: when an asset gets weight 0 in >95% of bootstraps, the mean
    # can sit just outside its own [p05, p95] band, which would make one of
    # these error-bar arms negative. matplotlib rejects that.
    lo = (resampled["mean"] - resampled["p05"]).clip(lower=0)
    hi = (resampled["p95"] - resampled["mean"]).clip(lower=0)

    ax.bar(x - 0.19, w_naive, 0.36, label="Naive max-Sharpe (single sample)",
           color=C["series"][0], linewidth=0, zorder=3)
    ax.bar(x + 0.19, resampled["mean"], 0.36,
           label=f"Resampled mean ({N_RESAMPLES} bootstraps)",
           color=C["series"][1], linewidth=0, zorder=3)
    ax.errorbar(x + 0.19, resampled["mean"], yerr=[lo, hi], fmt="none",
                ecolor=C["ink2"], elinewidth=1.4, capsize=4, zorder=4,
                label="5th-95th percentile across bootstraps")

    ax.set_xticks(x)
    ax.set_xticklabels(resampled.index, fontsize=9.5)
    ax.axhline(0, color=C["axis"], linewidth=0.8, zorder=2)
    style_axes(ax, "How stable is the 'optimal' portfolio, really?",
               "", "Max-Sharpe weight")
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    leg = ax.legend(frameon=True, fontsize=9.5, facecolor=C["surface"],
                    edgecolor=C["grid"])
    for t in leg.get_texts():
        t.set_color(C["ink2"])
    save_figure(fig, outfile, "wide bars = the optimizer is guessing")


def fig_weight_cap(w_uncapped, w_capped, tickers, cap, outfile) -> None:
    """Uncapped vs. capped max-Sharpe weights."""
    fig, ax = new_figure((12.5, 6.5))
    x = np.arange(len(tickers))
    ax.bar(x - 0.19, w_uncapped, 0.36, label="Unconstrained",
           color=C["series"][0], linewidth=0, zorder=3)
    ax.bar(x + 0.19, w_capped, 0.36, label=f"Capped at {cap:.0%} per asset",
           color=C["series"][1], linewidth=0, zorder=3)
    ax.axhline(cap, color=C["muted"], linewidth=1.2, zorder=4)
    ax.annotate(f"{cap:.0%} cap", (len(tickers) - 0.5, cap), xytext=(6, 3),
                textcoords="offset points", fontsize=9, color=C["muted"])
    ax.set_xticks(x)
    ax.set_xticklabels(tickers, fontsize=9.5)
    style_axes(ax, "Effect of a per-asset weight cap on the optimum", "", "Weight")
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    leg = ax.legend(frameon=True, fontsize=9.5, facecolor=C["surface"],
                    edgecolor=C["grid"])
    for t in leg.get_texts():
        t.set_color(C["ink2"])
    save_figure(fig, outfile, "a cap trades a little in-sample Sharpe for realism")


# =============================================================================
# ==  SECTION 10  --  WRITTEN SUMMARY  ========================================
# =============================================================================

def write_summary(ctx: dict) -> None:
    """Generate SUMMARY.md: the numbers, interpreted in plain English."""
    t = ctx["tickers"]
    stats = ctx["stats_table"]
    rc = ctx["risk_contrib"]
    cur_w = ctx["current_weights"]
    ms_w = ctx["max_sharpe_weights"]
    mv_w = ctx["min_var_weights"]
    mu, sigma, corr = ctx["mu"], ctx["sigma"], ctx["corr"]
    rf = ctx["rf"]

    cur = stats["Current (yours)"]
    ms = stats["Max Sharpe"]
    mv = stats["Min variance"]

    # --- Derived talking points ---------------------------------------------
    rc_series = pd.Series(rc["pct"], index=t).sort_values(ascending=False)
    w_series = pd.Series(cur_w, index=t)
    gap = (rc_series - w_series).sort_values(ascending=False)
    top_risk = rc_series.head(3)
    biggest_gap = gap.head(3)

    # What would the frontier give you at your current risk level?
    front = ctx["frontier"]
    eff = front[front["ret"] >= front.loc[front["vol"].idxmin(), "ret"]]
    reachable = eff[eff["vol"] <= cur["Annual volatility"]]
    if len(reachable):
        frontier_ret_at_your_risk = reachable["ret"].max()
        return_gap = frontier_ret_at_your_risk - cur["Annual return"]
    else:
        frontier_ret_at_your_risk = np.nan
        return_gap = np.nan
    # And how little risk could you take for your current return?
    same_ret = eff[eff["ret"] >= cur["Annual return"]]
    vol_at_your_return = same_ret["vol"].min() if len(same_ret) else np.nan
    vol_saving = (cur["Annual volatility"] - vol_at_your_return
                  if not np.isnan(vol_at_your_return) else np.nan)

    overweight = pd.Series(ms_w - cur_w, index=t).sort_values(ascending=False)
    wants_more = overweight[overweight > 0.02].head(4)
    wants_less = overweight[overweight < -0.02].tail(4).sort_values()

    # Average correlation of each asset with the rest -- explains the optimizer.
    avg_corr = ((corr.sum() - 1) / (len(t) - 1)).sort_values()

    lines: list[str] = []
    A = lines.append

    A("# What This Analysis Says About My Portfolio")
    A("")
    A(f"*Generated {datetime.today():%Y-%m-%d} from "
      f"{ctx['window_start']:%Y-%m-%d} to {ctx['window_end']:%Y-%m-%d} "
      f"({ctx['n_obs']} trading days, {ctx['n_obs'] / TRADING_DAYS:.2f} years), "
      f"risk-free rate {rf:.2%}.*")
    A("")
    A("> This is a coursework exercise in mean-variance optimization. Every")
    A("> number below is an estimate from a specific slice of history, not a")
    A("> forecast and not advice. The final section explains exactly how much")
    A("> to distrust it, which is the real point of the exercise.")
    A("")

    # --- 1. Headline ---------------------------------------------------------
    A("## 1. Where my portfolio actually sits")
    A("")
    A("| | Annual return | Annual vol | Sharpe | Diversification ratio | Effective N |")
    A("|---|---|---|---|---|---|")
    for name in ("Current (yours)", "Equal weight", "Min variance", "Max Sharpe"):
        if name not in stats:
            continue
        s = stats[name]
        A(f"| **{name}** | {s['Annual return']:.2%} | {s['Annual volatility']:.2%} "
          f"| {s['Sharpe ratio']:.3f} | {s['Diversification ratio']:.2f} "
          f"| {s['Effective N assets']:.1f} |")
    A("")
    A(f"My portfolio earned an estimated **{cur['Annual return']:.2%}** annualized "
      f"with **{cur['Annual volatility']:.2%}** volatility, a Sharpe ratio of "
      f"**{cur['Sharpe ratio']:.3f}**.")
    A("")
    if not np.isnan(return_gap) and return_gap > 0.0005:
        A(f"**The distance to the frontier.** At my current risk level "
          f"({cur['Annual volatility']:.2%} volatility), the efficient frontier "
          f"reached **{frontier_ret_at_your_risk:.2%}** — about "
          f"**{return_gap * 100:.1f} percentage points** more return for the "
          f"*same* risk. Read the other way: to earn my current "
          f"{cur['Annual return']:.2%} return, an efficient portfolio would have "
          f"needed only **{vol_at_your_return:.2%}** volatility instead of "
          f"{cur['Annual volatility']:.2%}"
          + (f" — about **{vol_saving * 100:.1f} points less risk**."
             if not np.isnan(vol_saving) else "."))
        A("")
        A("That gap is the cost of my particular mix — mostly the cost of "
          "holding several things that are really the same bet. It is *not* "
          "money I could have captured in real life, because the frontier was "
          "drawn with hindsight (see section 5).")
    else:
        A("**The distance to the frontier.** My portfolio sits at or very near "
          "the frontier over this window. That is a pleasant result but a "
          "suspicious one: the frontier is fitted to the same history my "
          "portfolio lived through, so landing on it partly reflects luck in "
          "this specific window.")
    A("")

    # --- 2. Concentration ----------------------------------------------------
    A("## 2. The concentration problem: where my risk really is")
    A("")
    excl = ctx.get("excluded") or []
    A(f"The model sees **{len(t)} tickers**"
      + (f" (of the {len(t) + len(excl)} I hold — {', '.join(excl)} "
         f"{'was' if len(excl) == 1 else 'were'} excluded for lack of history)"
         if excl else "")
      + f", but the effective number of independent positions is only "
        f"**{cur['Effective N assets']:.1f}** by dollar weight — and the risk "
        f"decomposition tells a sharper story still.")
    A("")
    A("| Asset | Dollar weight | Share of risk | Difference |")
    A("|---|---|---|---|")
    for tk in rc_series.index:
        d = rc_series[tk] - w_series[tk]
        flag = " ⚠️" if d > 0.05 else ""
        A(f"| {tk} | {w_series[tk]:.1%} | {rc_series[tk]:.1%} | {d:+.1%}{flag} |")
    A("")
    A(f"**Top three risk sources:** "
      + ", ".join(f"{k} ({v:.1%} of total risk)" for k, v in top_risk.items())
      + f" — together **{top_risk.sum():.1%}** of all portfolio risk from "
      f"**{w_series[top_risk.index].sum():.1%}** of the money.")
    A("")
    if len(biggest_gap) and biggest_gap.iloc[0] > 0.02:
        A("**Punching above their weight:** "
          + ", ".join(f"{k} (+{v:.1%})" for k, v in biggest_gap.items()
                      if v > 0.02)
          + ". These holdings determine more of my outcome than their dollar "
            "allocation suggests, because they are both volatile *and* highly "
            "correlated with the rest of what I own — so their moves reinforce "
            "rather than offset everything else.")
        A("")
    A(f"**Diversification ratio: {cur['Diversification ratio']:.2f}.** "
      + ("Imperfect correlation is cancelling out a meaningful share of my "
         "holdings' individual volatility."
         if cur["Diversification ratio"] > 1.35 else
         f"This is low. It means my {len(t)} holdings behave close to one big "
         "position — the correlations between them are high enough that "
         "spreading money across them bought relatively little risk reduction.")
      )
    A("")

    # --- 3. What the optimizer wants ----------------------------------------
    A("## 3. What the optimizer wants to change, and why")
    A("")
    A("| Asset | My weight | Min-variance | Max-Sharpe | Avg. correlation with rest |")
    A("|---|---|---|---|---|")
    for tk in t:
        A(f"| {tk} | {w_series[tk]:.1%} | {mv_w[t.index(tk)]:.1%} "
          f"| {ms_w[t.index(tk)]:.1%} | {avg_corr[tk]:.2f} |")
    A("")
    if len(wants_more):
        A("**Wants more of:** "
          + ", ".join(f"{k} (+{v:.1%})" for k, v in wants_more.items()) + ".")
        A("")
        # Attribute the optimizer's preference by RANK within this universe, not
        # by an absolute threshold -- "0.47 correlation" is only low or high
        # relative to the other things you own. Where the driver is a high
        # realized return, say so plainly: that is the fragile kind of reason.
        n_a = len(t)
        cutoff = max(2, round(n_a / 3))
        corr_rank = avg_corr.rank()                  # 1 = least correlated
        ret_rank = mu.rank(ascending=False)          # 1 = highest return
        for k in wants_more.index:
            why = []
            if corr_rank[k] <= cutoff:
                why.append(f"it is among the least correlated things I own "
                           f"(avg correlation {avg_corr[k]:.2f}), so it genuinely "
                           f"damps portfolio variance")
            if ret_rank[k] <= cutoff:
                why.append(f"it posted one of the highest returns in this window "
                           f"({mu[k]:.1%} annualized) — which is precisely the "
                           f"kind of input that is mostly estimation noise")
            if not why:
                why.append(f"it is a useful middle case (return {mu[k]:.1%}, "
                           f"avg correlation {avg_corr[k]:.2f})")
            A(f"- **{k}**: " + "; and ".join(why) + ".")
        A("")
    if len(wants_less):
        A("**Wants less of:** "
          + ", ".join(f"{k} ({v:.1%})" for k, v in wants_less.items()) + ".")
        A("")
        A("Usually this is not because the optimizer dislikes the asset in "
          "isolation, but because it is *redundant*: another holding already "
          "supplies the same exposure with a better risk/return profile over "
          "this window. When two assets are ~0.9 correlated, mean-variance "
          "optimization will nearly always pick one and discard the other, "
          "even when their expected returns are nearly identical. That "
          "all-or-nothing behavior is a known pathology, not wisdom.")
        A("")
    A(f"Note the min-variance column ignores expected returns entirely — it is "
      f"built from the covariance matrix alone. It is the more trustworthy of "
      f"the two optimized portfolios for exactly that reason "
      f"(Sharpe {mv['Sharpe ratio']:.3f} vs {ms['Sharpe ratio']:.3f} in-sample, "
      f"but far less dependent on return estimates that are mostly noise).")
    A("")

    # --- 4. Correlation structure -------------------------------------------
    A("## 4. What the correlation matrix shows")
    A("")
    pairs = []
    for i in range(len(t)):
        for j in range(i + 1, len(t)):
            pairs.append((corr.iloc[i, j], t[i], t[j]))
    pairs.sort(reverse=True)
    A("**Most correlated pairs** (these are the redundant ones):")
    A("")
    for c_, a_, b_ in pairs[:5]:
        A(f"- {a_} / {b_}: **{c_:.2f}**")
    A("")
    A("**Least correlated pairs** (these are what actually diversifies):")
    A("")
    for c_, a_, b_ in pairs[-5:]:
        A(f"- {a_} / {b_}: **{c_:.2f}**")
    A("")
    A(f"The single most independent holding by average correlation is "
      f"**{avg_corr.index[0]}** ({avg_corr.iloc[0]:.2f}); the most redundant is "
      f"**{avg_corr.index[-1]}** ({avg_corr.iloc[-1]:.2f}).")
    A("")

    # --- 5. Caveats ----------------------------------------------------------
    A("## 5. How much of this should I believe?")
    A("")
    A("Honestly: the *structure* a lot, the *numbers* very little. In order of "
      "how much they should worry me —")
    A("")
    A("**Estimation error is the dominant problem.** Mean-variance optimization "
      "has been called an \"error maximizer\" for good reason. The optimizer "
      "treats my estimated expected returns as facts, when a 3-year sample mean "
      "of a volatile asset has a standard error of several percentage points a "
      "year. Whatever happened to do best in this window gets loaded up on. "
      "See `08_lookback_sensitivity.png`: the same optimizer on the same assets "
      "produces materially different portfolios depending only on whether I "
      "look back 1, 2, or 3 years. None of those windows is more \"correct\" "
      "than the others — which means the weight differences between them are "
      "pure noise being reported as precision.")
    A("")
    if ctx.get("resampled") is not None:
        rs = ctx["resampled"]
        widest = (rs["p95"] - rs["p05"]).sort_values(ascending=False)
        A(f"The bootstrap makes this concrete (`10_resampled_weights.png`). "
          f"Across {N_RESAMPLES} resamples of my own return history, "
          f"**{widest.index[0]}**'s max-Sharpe weight ranged from "
          f"{rs.loc[widest.index[0], 'p05']:.0%} to "
          f"{rs.loc[widest.index[0], 'p95']:.0%}. A weight that unstable is not "
          f"an allocation decision; it is a coin flip reported to two decimals.")
        A("")
    sh = [s for s in ctx["short_history"] if s != "(none)"]
    if sh:
        plural = len(sh) > 1
        A(f"**Short history distorts specific holdings.** "
          f"{', '.join(sh)} {'have' if plural else 'has'} far less history than "
          f"the rest. {'Their' if plural else 'Its'} estimated "
          f"mean{'s' if plural else ''}, variance{'s' if plural else ''}, and — "
          f"worst of all — correlation{'s' if plural else ''} with everything "
          f"else {'are' if plural else 'is'} built on very few observations. The "
          f"optimizer has no way to know this and would state an opinion about "
          f"{'them' if plural else 'it'} with exactly the same confidence it "
          f"states one about SPY. Discount that view heavily — and note that "
          f"excluding an asset does not make the problem go away, it just moves "
          f"it: I still own it, the model simply cannot see it.")
        A("")
    A("**TSLY is structurally unlike the others.** It is a covered-call / "
      "option-income fund: it sells away TSLA's upside in exchange for premium, "
      "and returns most of its economics as distributions. Even on a "
      "total-return basis, its historical distribution is *engineered* — capped "
      "on the upside, largely uncapped on the downside, with a return stream "
      "whose shape depends on option-implied volatility levels that will not "
      "repeat. A historical mean and variance describe what that structure did "
      "under one volatility regime; they do not extrapolate.")
    A("")
    A("**Variance is not risk.** Mean-variance assumes returns are adequately "
      "described by a mean and a variance — effectively, that they are roughly "
      "normal. Real returns have fat tails (crashes far more often than a "
      "normal distribution allows) and negative skew (down moves bigger than up "
      "moves). Variance also penalizes upside deviation identically to "
      "downside, which no actual investor does. For a portfolio containing an "
      "option-income fund with deliberately asymmetric payoffs, this assumption "
      "is not a technicality — it is materially wrong for that holding.")
    A("")
    A("**The frontier is drawn with hindsight.** It shows what *would have been* "
      "optimal, knowing the outcome. Nobody could have selected that portfolio "
      "in advance. The honest use of the frontier is as a diagnostic of "
      "redundancy and concentration in what I already own — not as a target.")
    A("")

    # --- 6. Takeaways --------------------------------------------------------
    A("## 6. What I'd actually take away")
    A("")
    A("1. **The concentration finding is the robust one.** Risk contributions "
      "depend on covariances, which are estimated far more reliably than means. "
      f"That {top_risk.index[0]}, {top_risk.index[1]} and {top_risk.index[2]} "
      f"account for {top_risk.sum():.0%} of my risk on "
      f"{w_series[top_risk.index].sum():.0%} of my money is a real structural "
      "fact about my portfolio, not an artifact.")
    A("2. **Overlapping holdings are doing less than the ticker count suggests.** "
      f"An effective N of {cur['Effective N assets']:.1f} and a diversification "
      f"ratio of {cur['Diversification ratio']:.2f} both say the same thing: "
      "several of these positions are the same bet wearing different names.")
    A("3. **Ignore the exact optimal weights.** They are the least reliable "
      "output here. The min-variance portfolio, which never touches expected "
      "returns, is the one worth studying.")
    A("4. **The diversifiers are the ones with low average correlation**, not "
      f"the ones with the best returns — {avg_corr.index[0]} and "
      f"{avg_corr.index[1]} do more for portfolio risk per dollar than their "
      "standalone performance suggests.")
    A("")
    A("---")
    A("")
    A("*Figures in `output/`. Full console walkthrough in "
      "`output/analysis_log.txt`. Regenerate with `python "
      "portfolio_optimization.py`.*")

    with open("SUMMARY.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    say("  [saved] SUMMARY.md -- written interpretation of these results")


# =============================================================================
# ==  MAIN  ===================================================================
# =============================================================================

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    tickers_cfg = list(SHARES.keys())

    say("")
    say("#" * 78)
    say("#  MARKOWITZ MEAN-VARIANCE PORTFOLIO OPTIMIZATION")
    say(f"#  {len(tickers_cfg)} assets | {LOOKBACK_YEARS:.0f}-year lookback | "
        f"risk-free {RISK_FREE_RATE:.2%}")
    say("#" * 78)

    # ------------------------------------------------------------------ [1]
    header("STEP 1  --  DATA ACQUISITION")
    # Download the longest window any part of the analysis needs, once.
    max_years = max([LOOKBACK_YEARS] + list(LOOKBACK_SENSITIVITY_YEARS))
    prices_all = download_prices(tickers_cfg, max_years)

    # Latest price per ticker, for valuing the current portfolio. Taken from the
    # FULL download (before any window truncation) so it is genuinely current.
    latest_prices = prices_all.ffill().iloc[-1]

    # Restrict to the requested lookback, then handle unequal history.
    cutoff = prices_all.index[-1] - pd.Timedelta(days=int(365.25 * LOOKBACK_YEARS))
    prices_window = prices_all[prices_all.index >= cutoff]
    prices, universe = report_history_and_align(
        prices_window, HISTORY_POLICY, MIN_HISTORY_YEARS, LOOKBACK_YEARS
    )
    tickers = list(prices.columns)
    n = len(tickers)

    explain(
        "Why auto-adjusted prices, and why it matters here",
        """
        Every price above is an ADJUSTED close: dividends and distributions are
        reinvested back into the series, so what we are measuring is TOTAL
        return, not price return.

        This is not a technicality for this particular portfolio. TSLY is an
        option-income fund that pays out most of its economics as distributions
        -- its raw share price drifts DOWN by construction. SCHD is a dividend
        fund. Using raw prices would have told the optimizer that these assets
        lose money, which is false, and the resulting 'optimal' portfolio would
        have been garbage in a way that is completely invisible from the output.
        """
    )

    # ------------------------------------------------------------------ [2]
    header("STEP 2  --  RETURNS, EXPECTED RETURNS, AND THE COVARIANCE MATRIX")
    returns = compute_returns(prices)
    mu_s, sigma_s, corr = annualize_moments(returns)
    mu, sigma = mu_s.values, sigma_s.values
    vols = np.sqrt(np.diag(sigma))

    explain(
        "The two objects that ARE the Markowitz model",
        f"""
        We just reduced {len(returns) * n:,} daily return observations down to
        exactly two things:

            mu     -- expected annual return per asset      ({n}-element vector)
            Sigma  -- annualized covariance matrix          ({n}x{n} matrix)

        Formulas:

            mu_i      = mean(daily return_i) x 252
            Sigma_ij  = cov(daily_i, daily_j) x 252
            sigma_i   = sqrt(Sigma_ii)                    <- annualized volatility

        That is the ENTIRE input. Markowitz needs nothing else -- not price
        levels, not fundamentals, not your holding period. Everything the model
        can ever tell you is already contained in these two objects, which is
        both the source of its elegance and the source of its fragility: if mu
        is wrong, the answer is wrong, and there is no other information in the
        system to catch the error.

        Note the asymmetry in how well we know these. Sigma is estimated from
        the SPREAD of returns, and volatility is quite stable over time. mu is
        estimated from the AVERAGE of returns, and averages of noisy series
        converge painfully slowly -- with 3 years of data the standard error on
        an annual mean return is roughly sigma/sqrt(3), which for a 30%-vol
        asset is about +/- 17 percentage points a year. That is not a typo.
        """
    )

    table = pd.DataFrame({
        "Ann. return": mu_s, "Ann. volatility": pd.Series(vols, index=tickers),
        "Sharpe (solo)": (mu_s - RISK_FREE_RATE) / vols,
    }).sort_values("Sharpe (solo)", ascending=False)
    say("")
    say("  Per-asset annualized moments:")
    say("")
    # Format per column -- Sharpe is a ratio, not a percentage.
    tdisp = pd.DataFrame({
        "Ann. return": table["Ann. return"].map("{:>8.2%}".format),
        "Ann. volatility": table["Ann. volatility"].map("{:>8.2%}".format),
        "Sharpe (solo)": table["Sharpe (solo)"].map("{:>8.3f}".format),
    })
    for line in tdisp.to_string().split("\n"):
        say("    " + line)

    # ------------------------------------------------------------------ [3]
    header("STEP 3  --  VISUALIZING THE COVARIANCE STRUCTURE")
    fig_price_history(prices, "01_price_history.png")
    fig_heatmap(corr, "Correlation matrix (annualized daily returns)",
                "02_correlation_heatmap.png", ".2f", True, "Correlation")
    fig_heatmap(sigma_s, "Covariance matrix (annualized)",
                "03_covariance_heatmap.png", ".3f", True, "Covariance")

    # Data-driven observations about the correlation structure.
    avg_corr = ((corr.sum() - 1) / (n - 1)).sort_values()
    pairs = [(corr.iloc[i, j], tickers[i], tickers[j])
             for i in range(n) for j in range(i + 1, n)]
    pairs.sort(reverse=True)

    # Build the dynamic tables separately, at a fixed indent, and interpolate
    # them at column 0 below -- explain() dedents the whole body by its common
    # leading whitespace, so an interpolated block must not inherit the
    # template's own indentation on its first line only.
    most_corr_block = "\n".join(
        f"            {a:<6} / {b:<6}  {c:>6.2f}" for c, a, b in pairs[:4])
    least_corr_block = "\n".join(
        f"            {a:<6} / {b:<6}  {c:>6.2f}" for c, a, b in pairs[-4:])

    explain(
        "How to read the correlation heatmap",
        f"""
        The diagonal is 1.0 by definition (everything is perfectly correlated
        with itself). Everything interesting is off-diagonal.

        Your most correlated pairs -- the REDUNDANT ones:

{most_corr_block}

        Your least correlated pairs -- the ones that actually DIVERSIFY:

{least_corr_block}

        Most independent holding overall: {avg_corr.index[0]} (average
        correlation {avg_corr.iloc[0]:.2f} with everything else).
        Most redundant holding overall: {avg_corr.index[-1]} (average
        correlation {avg_corr.iloc[-1]:.2f}).

        NOTICE THE EMPTY HALF OF THE COLOR SCALE. The heatmap is scaled from
        -1 (blue) through 0 (neutral) to +1 (red), but the lowest correlation
        anywhere in your portfolio is {min(c for c, _, _ in pairs):.2f}. Not one pair is
        negatively correlated. That means you own no true HEDGE -- nothing here
        reliably rises when the rest falls. Your diversification comes entirely
        from assets that move WEAKLY together ({avg_corr.index[0]} at {avg_corr.iloc[0]:.2f} being the
        best of them), never from assets that move OPPOSITELY. That is normal
        for a long-only equity-plus-gold portfolio, but it is worth seeing
        explicitly: 'diversified' here means 'damped', not 'offset'.

        WHY THIS, AND NOT VOLATILITY, IS THE CENTER OF THE THEORY:

        A common intuition is that a portfolio's risk comes from how volatile
        its holdings are. That is only half true. Recall:

            Var(portfolio) = sum_i w_i^2 sigma_i^2
                           + sum_{{i != j}} w_i w_j rho_ij sigma_i sigma_j

        With {n} assets there are {n} variance terms and {n * (n - 1)} covariance
        terms. The covariance terms dominate. So a portfolio of many
        volatile-but-unrelated assets can be far calmer than a portfolio of a
        few mild-but-identical ones. Volatility is a property of an asset; RISK
        is a property of an asset's relationship to your other assets.

        This is also why adding a 'safe' asset that is highly correlated with
        what you already own does almost nothing for you, while adding a
        volatile asset that moves independently can genuinely reduce total
        portfolio risk.
        """
    )

    # --- Current portfolio weights from share counts -------------------------
    header("STEP 3b  --  YOUR CURRENT PORTFOLIO")
    values_all = pd.Series(
        {t: SHARES[t] * latest_prices[t]
         for t in latest_prices.index if t in SHARES}
    ).dropna()
    total_value_all = values_all.sum()
    values = values_all[tickers]
    current_w = (values / values.sum()).values

    say("")
    say(f"    {'Ticker':<8}{'Shares':>10}{'Price':>11}{'Value':>13}{'Weight':>10}")
    say(f"    {'-' * 8}{'-' * 10}{'-' * 11}{'-' * 13}{'-' * 10}")
    for tk in sorted(tickers, key=lambda x: -values[x]):
        say(f"    {tk:<8}{SHARES[tk]:>10.4f}{latest_prices[tk]:>11.2f}"
            f"{values[tk]:>13,.2f}{values[tk] / values.sum():>10.2%}")
    say(f"    {'-' * 52}")
    say(f"    {'TOTAL':<8}{'':<10}{'':<11}{values.sum():>13,.2f}{1.0:>10.2%}")

    excluded = [t for t in tickers_cfg if t not in tickers]
    if excluded:
        excl_val = values_all[[t for t in excluded if t in values_all.index]].sum()
        say("")
        say(f"  !! NOTE: {', '.join(excluded)} ({excl_val / total_value_all:.1%} of "
            f"your actual portfolio value) is excluded from the model because of")
        say(f"     the HISTORY_POLICY setting. The weights above are renormalized")
        say(f"     over the remaining assets, so every risk number below describes")
        say(f"     {1 - excl_val / total_value_all:.0%} of your real portfolio.")

    # ------------------------------------------------------------------ [4]
    header("STEP 4  --  MONTE CARLO: 50,000 RANDOM PORTFOLIOS")
    cloud, cloud_w = simulate_random_portfolios(
        mu, sigma, RISK_FREE_RATE, N_PORTFOLIOS, rng)
    fig_monte_carlo(cloud, "04_monte_carlo.png")

    best = cloud.loc[cloud["sharpe"].idxmax()]
    explain(
        "What the random cloud is telling you",
        f"""
        We generated {N_PORTFOLIOS:,} random long-only, fully-invested
        portfolios by sampling weights from a Dirichlet distribution (which
        guarantees w_i >= 0 and sum(w) = 1 by construction), then evaluated each
        one:

            R_p     = w' mu
            sigma_p = sqrt(w' Sigma w)
            Sharpe  = (R_p - r_f) / sigma_p        with r_f = {RISK_FREE_RATE:.2%}

        Best random portfolio found: Sharpe {best['sharpe']:.3f}
        ({best['ret']:.2%} return, {best['vol']:.2%} volatility).
        Worst: Sharpe {cloud['sharpe'].min():.3f}.

        Two things to notice in 04_monte_carlo.png:

        1. The cloud has a hard UPPER-LEFT BOUNDARY. Random mixing cannot get
           you past it. That boundary is the efficient frontier -- it exists as
           a real feature of the data, and Step 5 just computes it precisely
           instead of stumbling onto it.

        2. The cloud is a solid region, not a line. Every point inside it is a
           portfolio you could actually hold, and every one of them is
           INEFFICIENT: there is a point directly above it (more return, same
           risk) and a point directly to its left (same return, less risk).

        On the risk-free rate: r_f sets the origin of the reward axis. Sharpe
        measures excess return per unit of risk, so it is the SLOPE of the line
        from (0, r_f) to a portfolio's point. Maximizing Sharpe = finding the
        steepest such line = finding where a line from r_f is TANGENT to the
        frontier. That is why the max-Sharpe portfolio is also called the
        tangency portfolio, and why changing r_f moves it along the frontier.
        """
    )

    # ------------------------------------------------------------------ [5]
    header("STEP 5  --  FORMAL OPTIMIZATION AND THE EFFICIENT FRONTIER")
    say("  Solving for the global minimum variance portfolio...")
    w_minvar = min_variance_portfolio(sigma, 1.0, rng)
    say("  Solving for the maximum Sharpe (tangency) portfolio...")
    w_maxsharpe = max_sharpe_portfolio(mu, sigma, RISK_FREE_RATE, 1.0, rng)
    say(f"  Tracing the efficient frontier ({N_FRONTIER_POINTS} target returns)...")
    frontier = efficient_frontier(mu, sigma, N_FRONTIER_POINTS, 1.0, rng)
    w_equal = np.full(n, 1.0 / n)

    frontier_capped = None
    w_capped = None
    if RUN_WEIGHT_CAP and MAX_WEIGHT < 1.0:
        say(f"  Tracing the constrained frontier (max {MAX_WEIGHT:.0%} per asset)...")
        frontier_capped = efficient_frontier(mu, sigma, N_FRONTIER_POINTS,
                                             MAX_WEIGHT, rng)
        w_capped = max_sharpe_portfolio(mu, sigma, RISK_FREE_RATE, MAX_WEIGHT, rng)

    named = {
        "Current (yours)": current_w,
        "Equal weight": w_equal,
        "Min variance": w_minvar,
        "Max Sharpe": w_maxsharpe,
    }
    stats_table = {k: portfolio_stats(v, mu, sigma, RISK_FREE_RATE)
                   for k, v in named.items()}
    portfolios_xy = {
        k: {"ret": port_return(v, mu), "vol": port_vol(v, sigma)}
        for k, v in named.items()
    }
    assets_xy = pd.DataFrame({"ret": mu_s, "vol": pd.Series(vols, index=tickers)})

    fig_efficient_frontier(cloud, frontier, assets_xy, portfolios_xy,
                           RISK_FREE_RATE, "05_efficient_frontier.png",
                           frontier_capped)

    explain(
        "The efficient frontier, and what each solve actually asked",
        f"""
        Three separate optimization problems were solved, all with the same two
        constraints (long-only: w_i >= 0; fully invested: sum(w) = 1):

            GLOBAL MINIMUM VARIANCE
                minimize   w' Sigma w
            -> the single least-volatile portfolio obtainable from these assets.
               Notice mu does not appear. This portfolio has no opinion about
               returns at all, which makes it the most statistically robust
               thing in this entire file.

            MAXIMUM SHARPE (tangency)
                maximize   (w' mu - r_f) / sqrt(w' Sigma w)
            -> the best risk-adjusted mix. Leans heavily on mu, and therefore
               inherits all of mu's estimation error.

            THE FRONTIER ITSELF
                for each target return R*:
                    minimize   w' Sigma w   subject to   w' mu = R*
            -> sweeping R* traces the curve. The portion above the minimum
               variance point is 'efficient'; the portion below it is dominated
               and no one should hold it.

        Results over this window:

            Min variance ..... return {stats_table['Min variance']['Annual return']:>7.2%}   vol {stats_table['Min variance']['Annual volatility']:>7.2%}   Sharpe {stats_table['Min variance']['Sharpe ratio']:>6.3f}
            Max Sharpe ....... return {stats_table['Max Sharpe']['Annual return']:>7.2%}   vol {stats_table['Max Sharpe']['Annual volatility']:>7.2%}   Sharpe {stats_table['Max Sharpe']['Sharpe ratio']:>6.3f}
            Equal weight ..... return {stats_table['Equal weight']['Annual return']:>7.2%}   vol {stats_table['Equal weight']['Annual volatility']:>7.2%}   Sharpe {stats_table['Equal weight']['Sharpe ratio']:>6.3f}
            YOUR PORTFOLIO ... return {stats_table['Current (yours)']['Annual return']:>7.2%}   vol {stats_table['Current (yours)']['Annual volatility']:>7.2%}   Sharpe {stats_table['Current (yours)']['Sharpe ratio']:>6.3f}

        THE CAPITAL MARKET LINE is the straight line from (0, r_f) through the
        tangency portfolio. It represents portfolios built by mixing the
        tangency portfolio with cash: points to its left are partly in T-bills,
        points beyond it are leveraged. Its slope is exactly the tangency
        Sharpe ratio ({stats_table['Max Sharpe']['Sharpe ratio']:.3f}). Because it is a straight line
        lying above the curved frontier everywhere except the tangency point,
        Tobin's separation theorem follows: if you can borrow and lend at r_f,
        every investor should hold the SAME risky portfolio (the tangency one)
        and adjust risk purely by how much cash they mix in.

        Now look at where YOUR portfolio's star sits relative to the black
        curve. The vertical gap is return forgone at your current risk. The
        horizontal gap is risk taken without compensation.
        """
    )

    say("")
    say("  Optimal weight vectors:")
    say("")
    wdf = pd.DataFrame(named, index=tickers)
    for line in wdf.to_string(float_format=lambda v: f"{v:>8.2%}").split("\n"):
        say("    " + line)

    fig_weights_comparison(wdf, "06_weights_comparison.png")

    # ------------------------------------------------------------------ [6]
    header("STEP 6  --  DIVERSIFICATION AND RISK CONTRIBUTION")

    say("")
    say("  Comparison of all four portfolios:")
    say("")
    comp = pd.DataFrame(stats_table).T
    fmt = {"Annual return": "{:.2%}", "Annual volatility": "{:.2%}",
           "Sharpe ratio": "{:.3f}", "Diversification ratio": "{:.2f}",
           "Effective N assets": "{:.1f}", "Max weight": "{:.1%}"}
    disp = comp.copy()
    for c_, f_ in fmt.items():
        disp[c_] = comp[c_].map(f_.format)
    for line in disp.to_string().split("\n"):
        say("    " + line)

    rc = risk_contributions(current_w, sigma)
    rc_df = pd.DataFrame({
        "Weight": current_w,
        "Marginal risk": rc["mrc"],
        "Risk contribution": rc["crc"],
        "% of total risk": rc["pct"],
        "Risk minus weight": rc["pct"] - current_w,
    }, index=tickers).sort_values("% of total risk", ascending=False)

    say("")
    say("  Risk decomposition of YOUR portfolio:")
    say("")
    rc_disp = rc_df.copy()
    for c_ in ("Weight", "% of total risk", "Risk minus weight"):
        rc_disp[c_] = rc_df[c_].map("{:>8.2%}".format)
    for c_ in ("Marginal risk", "Risk contribution"):
        rc_disp[c_] = rc_df[c_].map("{:>8.4f}".format)
    for line in rc_disp.to_string().split("\n"):
        say("    " + line)
    say("")
    say(f"    Sum of risk contributions = {rc['crc'].sum():.4f} "
        f"= portfolio volatility {rc['total_vol']:.4f}  (Euler's theorem checks out)")

    fig_risk_contributions(tickers, current_w, rc["pct"], "07_risk_contributions.png")

    top3 = rc_df.head(3)
    # The headline finding is not "which assets carry the most risk" (a big
    # holding naturally carries a lot) but "which assets carry MORE risk than
    # their dollar weight" -- that is the concentration the eye misses.
    over = rc_df[rc_df["Risk minus weight"] > 0.005].sort_values(
        "Risk minus weight", ascending=False)
    under = rc_df[rc_df["Risk minus weight"] < -0.005].sort_values(
        "Risk minus weight")
    over_block = "\n".join(
        f"                {i:<7} {r['Weight']:>6.1%} of money  ->  "
        f"{r['% of total risk']:>6.1%} of risk   ({r['Risk minus weight']:+.1%})"
        for i, r in over.iterrows()) or "                (none)"
    under_block = "\n".join(
        f"                {i:<7} {r['Weight']:>6.1%} of money  ->  "
        f"{r['% of total risk']:>6.1%} of risk   ({r['Risk minus weight']:+.1%})"
        for i, r in under.iterrows()) or "                (none)"

    explain(
        "Reading the risk decomposition -- the most useful table here",
        f"""
        Formulas:

            sigma_p = sqrt(w' Sigma w)                    total portfolio risk
            MRC_i   = (Sigma w)_i / sigma_p               marginal contribution
            CRC_i   = w_i x MRC_i                         component contribution
            sum_i CRC_i = sigma_p                         (exact, by Euler)

        MRC answers "if I add a dollar of asset i, how much does total portfolio
        volatility rise?". CRC scales that by how much you actually hold, and
        the CRCs sum exactly to total volatility -- so the percentages are a
        complete, exhaustive split of your risk, not an approximation.

        YOUR RESULT -- holdings that carry MORE risk than money:

{over_block}

        Holdings that carry LESS risk than money (your genuine diversifiers):

{under_block}

        The top three risk sources overall are {top3.index[0]}, {top3.index[1]} and
        {top3.index[2]}, together {top3['% of total risk'].sum():.1%} of your total risk on
        {top3['Weight'].sum():.1%} of your money.

        Over-contribution happens on two fronts at once: the holding is
        individually volatile AND highly correlated with the rest of the
        portfolio, so its moves REINFORCE rather than offset your other
        positions. An asset's risk contribution depends on its covariance with
        the whole portfolio, not on its own volatility in isolation -- which is
        exactly why a volatile asset that moves independently (or negatively)
        can have a small or even negative risk contribution, acting as a hedge.
        Note that a large holding can still be an under-contributor if it is the
        thing everything else is measured against.

        DIVERSIFICATION RATIO = {stats_table['Current (yours)']['Diversification ratio']:.3f}
            = (weighted avg of individual vols) / (actual portfolio vol)
            The volatility that imperfect correlation cancelled out for you.
            1.0 = no benefit whatsoever; higher is better.

        EFFECTIVE NUMBER OF ASSETS = {stats_table['Current (yours)']['Effective N assets']:.2f}  (from {n} holdings)
            = 1 / sum(w_i^2), the inverse Herfindahl index.
            Equal weights across {n} assets would give exactly {n:.1f}.
            But note this only measures DOLLAR spread -- {n} different S&P 500
            index funds would also score {n:.1f} while being one single bet.
            Read it together with the diversification ratio, which measures
            whether that spread actually bought you anything.
        """
    )

    # ------------------------------------------------------------------ [7]
    header("STEP 7  --  ESTIMATION ERROR: THE LOOKBACK SENSITIVITY TEST")
    sens_table = None
    if RUN_LOOKBACK_SENSITIVITY:
        say("")
        # The universe for this test must be assets that actually HAVE data over
        # the longest window being compared. Otherwise a single young asset
        # silently truncates every window to the same short span, all three
        # solves see identical data, and the test appears to show perfect
        # stability -- the exact opposite of the truth.
        longest = max(LOOKBACK_SENSITIVITY_YEARS)
        cut_long = prices_all.index[-1] - pd.Timedelta(days=int(365.25 * longest))
        sens_universe = [
            t for t in tickers
            if prices_all[t].first_valid_index() is not None
            and prices_all[t].first_valid_index() <= cut_long + pd.Timedelta(days=7)
        ]
        dropped_sens = [t for t in tickers if t not in sens_universe]
        if dropped_sens:
            say(f"  Excluding {', '.join(dropped_sens)} from this test: they lack")
            say(f"  data covering the full {longest:.0f}-year window, and including")
            say(f"  them would force every window down to the same short span,")
            say(f"  making the comparison meaningless.")
            say("")

        cols = {}
        for yrs in LOOKBACK_SENSITIVITY_YEARS:
            cut = prices_all.index[-1] - pd.Timedelta(days=int(365.25 * yrs))
            px_w = (prices_all.loc[prices_all.index >= cut, sens_universe]
                    .dropna(how="any"))
            if len(px_w) < 60:
                say(f"  Skipping {yrs:.0f}-year window (only {len(px_w)} days).")
                continue
            r_w = compute_returns(px_w)
            mu_w, sig_w, _ = annualize_moments(r_w)
            w_w = max_sharpe_portfolio(mu_w.values, sig_w.values,
                                       RISK_FREE_RATE, 1.0, rng)
            cols[f"{yrs:.0f}-year"] = w_w
            say(f"  {yrs:.0f}-year window: {len(px_w)} trading days -- solved.")

        sens_table = pd.DataFrame(cols, index=sens_universe)
        fig_lookback_sensitivity(sens_table, "08_lookback_sensitivity.png")

        say("")
        say("  Max-Sharpe weights by lookback window:")
        say("")
        for line in sens_table.to_string(
                float_format=lambda v: f"{v:>8.2%}").split("\n"):
            say("    " + line)

        swing = (sens_table.max(axis=1) - sens_table.min(axis=1)).sort_values(
            ascending=False)
        swing_block = "\n".join(
            f"                {k:<8} moves {v:>6.1%}  "
            f"(from {sens_table.loc[k].min():>6.1%} to {sens_table.loc[k].max():>6.1%})"
            for k, v in swing.head(5).items())
        explain(
            "This is the single most important output in the file",
            f"""
            The same optimizer, the same assets, the same constraints -- the
            ONLY thing that changed is how far back we looked. If mean-variance
            optimization were extracting a durable signal, these columns would
            be nearly identical.

            Largest weight swings across windows:

{swing_block}

            None of these windows is more 'correct' than the others. So the
            differences between them are not information -- they are ESTIMATION
            ERROR being reported to two decimal places.

            The mechanism: the optimizer maximizes (w'mu - rf)/sqrt(w'Sigma w).
            It has no concept that mu is an estimate with a standard error. An
            asset that got lucky in the sample shows a high mu_i, and the
            optimizer reads that luck as a permanent property and allocates
            accordingly. Michaud's name for this is 'error maximization': of all
            possible portfolios, mean-variance reliably selects the one whose
            inputs are most favorably mis-estimated.

            This is why practitioners rarely use raw mean-variance output. The
            standard defenses -- all demonstrated below -- are: shrink the
            covariance matrix, resample the optimization, cap the weights, or
            abandon expected returns entirely and use minimum-variance or
            risk-parity instead.
            """
        )

    # ------------------------------------------------------------------ [8]
    header("STEP 8  --  STRETCH GOALS: SHRINKAGE, RESAMPLING, CONSTRAINTS")

    delta = None
    if RUN_SHRINKAGE:
        say("")
        say("  [Stretch 1] Ledoit-Wolf shrinkage covariance...")
        sigma_lw, delta = ledoit_wolf_shrinkage(returns)
        w_ms_lw = max_sharpe_portfolio(mu, sigma_lw, RISK_FREE_RATE, 1.0, rng)
        frontier_lw = efficient_frontier(mu, sigma_lw, N_FRONTIER_POINTS, 1.0, rng)
        fig_shrinkage(frontier, frontier_lw, w_maxsharpe, w_ms_lw, tickers,
                      delta, "09_shrinkage_comparison.png")

        turnover = np.abs(w_ms_lw - w_maxsharpe).sum() / 2
        explain(
            "Ledoit-Wolf shrinkage",
            f"""
            Shrinkage intensity chosen by the estimator: delta = {delta:.3f}

            That means the covariance matrix used is {delta:.1%} the structured
            'constant correlation' target and {1 - delta:.1%} the raw sample
            estimate:

                Sigma_shrunk = delta x F + (1 - delta) x S

            where F keeps each asset's own variance but replaces every pairwise
            correlation with the average correlation across all pairs.

            A high delta is the estimator telling you your sample covariance is
            noisy relative to how wrong the simple target is. Short windows and
            many assets both push delta up.

            Why it helps: with N assets you estimate N(N+1)/2 = {n * (n + 1) // 2} covariance
            parameters from {len(returns)} observations. The sample matrix's extreme
            eigenvalues are biased outward, and the optimizer effectively INVERTS
            this matrix -- so the smallest, most-biased eigenvalues dominate the
            result. The optimizer then discovers near-riskless-looking
            combinations that are nothing of the sort out of sample. Shrinking
            pulls those eigenvalues back toward the middle.

            Effect on the answer: the max-Sharpe weights moved by
            {turnover:.1%} (one-way turnover). The shrunk frontier sits
            slightly LOWER than the naive one -- which is the point. The naive
            frontier was partly fitted noise; some of that apparent free lunch
            was never real.
            """
        )

    resampled = None
    if RUN_RESAMPLING:
        say("")
        say(f"  [Stretch 2] Resampled frontier ({N_RESAMPLES} bootstraps)...")
        resampled = resampled_max_sharpe(returns, RISK_FREE_RATE, N_RESAMPLES,
                                         1.0, rng)
        fig_resampled(resampled, w_maxsharpe, "10_resampled_weights.png")

        say("")
        say("  Bootstrap distribution of max-Sharpe weights:")
        say("")
        rs_disp = resampled.copy()
        for c_ in rs_disp.columns:
            rs_disp[c_] = resampled[c_].map("{:>8.2%}".format)
        rs_disp.insert(0, "Naive", pd.Series(w_maxsharpe, index=tickers)
                       .map("{:>8.2%}".format))
        for line in rs_disp.to_string().split("\n"):
            say("    " + line)

        width = (resampled["p95"] - resampled["p05"]).sort_values(ascending=False)
        w_res = resampled["mean"].values
        res_stats = portfolio_stats(w_res, mu, sigma, RISK_FREE_RATE)
        spread_block = "\n".join(
            f"                {k:<8} {resampled.loc[k, 'p05']:>6.1%} to "
            f"{resampled.loc[k, 'p95']:>6.1%}    "
            f"(naive answer: {w_maxsharpe[tickers.index(k)]:>6.1%})"
            for k in width.head(5).index)
        explain(
            "The resampled portfolio -- estimation error, measured",
            f"""
            We drew {N_RESAMPLES} bootstrap samples from your own return history
            (sampling days with replacement), recomputed mu and Sigma from each,
            and re-solved for max Sharpe every time.

            The point is the SPREAD, not the average. Widest 90% intervals:

{spread_block}

            Read those intervals carefully. If an asset's optimal weight ranges
            from near zero to a large allocation depending on which days
            happened to be drawn, then the naive optimizer's confident-looking
            point estimate carries almost no information. Nothing about your
            actual portfolio changed between bootstraps -- only which random
            subset of history the optimizer saw.

            The AVERAGE of the resampled weights is itself a usable portfolio
            (Michaud's resampled efficiency):

                return {res_stats['Annual return']:.2%}   vol {res_stats['Annual volatility']:.2%}   Sharpe {res_stats['Sharpe ratio']:.3f}
                effective N = {res_stats['Effective N assets']:.2f}  vs  {stats_table['Max Sharpe']['Effective N assets']:.2f} for the naive optimum

            It is almost always more diversified, because an asset that only
            wins under some resamples receives a partial allocation instead of
            an all-or-nothing one. Its in-sample Sharpe is lower than the naive
            optimum's by construction -- but the naive optimum's in-sample
            Sharpe is precisely the number that does not survive contact with
            the future.
            """
        )

    if RUN_WEIGHT_CAP and w_capped is not None:
        say("")
        say(f"  [Stretch 3] Weight cap at {MAX_WEIGHT:.0%} per asset...")
        fig_weight_cap(w_maxsharpe, w_capped, tickers, MAX_WEIGHT,
                       "11_weight_cap_comparison.png")
        cap_stats = portfolio_stats(w_capped, mu, sigma, RISK_FREE_RATE)
        explain(
            "Position limits: the practitioner's blunt instrument",
            f"""
            Adding 0 <= w_i <= {MAX_WEIGHT:.0%} to the same optimization:

                Unconstrained ... Sharpe {stats_table['Max Sharpe']['Sharpe ratio']:.3f}   max weight {stats_table['Max Sharpe']['Max weight']:.1%}   effective N {stats_table['Max Sharpe']['Effective N assets']:.2f}
                Capped .......... Sharpe {cap_stats['Sharpe ratio']:.3f}   max weight {cap_stats['Max weight']:.1%}   effective N {cap_stats['Effective N assets']:.2f}

            The capped portfolio has a lower IN-SAMPLE Sharpe -- it must, since
            it is solving the same problem with strictly fewer options. The
            in-sample loss is {stats_table['Max Sharpe']['Sharpe ratio'] - cap_stats['Sharpe ratio']:.3f} Sharpe points.

            But that framing has it backwards. The unconstrained solution's
            extra Sharpe came from concentrating in whichever assets had the
            most favorably mis-estimated inputs. The cap prevents the optimizer
            from acting on estimates it has no right to be confident about. Jorion
            and others have shown empirically that constrained mean-variance
            portfolios routinely BEAT unconstrained ones out of sample, despite
            being strictly worse in sample.

            A weight cap is a crude, effective form of the same regularization
            that shrinkage and resampling achieve more elegantly -- and it is
            the one that shows up in nearly every real investment policy
            statement.
            """
        )

    # ------------------------------------------------------------------ [9]
    header("STEP 9  --  CAVEATS YOU SHOULD CARRY OUT OF THIS EXERCISE")

    # Short-history assets fall into two buckets, and BOTH deserve the caveat:
    # ones still in the model but thinly observed, and ones the history policy
    # removed entirely (which you still own -- the model just cannot see them).
    hist_years = {
        t: (prices_all[t].notna().sum() / TRADING_DAYS)
        for t in prices_all.columns
    }
    short_in_model = [t for t in tickers if hist_years.get(t, 0) < LOOKBACK_YEARS * 0.9]
    removed = [t for t in tickers_cfg if t not in tickers]
    short_history = short_in_model + removed

    if short_in_model and removed:
        short_desc = (
            f"{', '.join(short_in_model)} (in the model, thinly observed) and "
            f"{', '.join(removed)} (excluded entirely by HISTORY_POLICY)")
    elif removed:
        short_desc = (
            f"{', '.join(removed)} -- excluded from the model entirely by "
            f"HISTORY_POLICY, having only "
            f"{min(hist_years.get(t, 0) for t in removed):.1f} years of history")
    elif short_in_model:
        short_desc = f"{', '.join(short_in_model)}"
    else:
        short_desc = "(no asset in this run is unusually short-lived)"
    if not short_history:
        short_history = ["(none)"]

    explain(
        "What this model cannot tell you",
        f"""
        1. GARBAGE IN, GARBAGE OUT -- and mu is the garbage.
           Expected returns estimated from {len(returns) / TRADING_DAYS:.1f} years of daily data have
           enormous standard errors. For an asset with 30% annual volatility,
           the standard error of the estimated annual mean over 3 years is about
           30%/sqrt(3) ~ 17 percentage points. You cannot distinguish a 5%
           expected return from a 25% one at that precision. The optimizer does
           not know this and will act as though every mu_i is exact.

        2. SHORT HISTORY: {short_desc}.
           An asset with little history has unreliable means, variances, and --
           worst of all -- unreliable correlations with everything else.
           Correlation estimates need a lot of data to stabilize, and a wrong
           correlation is more damaging than a wrong mean, because it propagates
           through every cross term of w' Sigma w rather than affecting one
           entry. Treat the optimizer's opinion about such holdings as close to
           uninformative. Note that an EXCLUDED asset is not a solved problem
           either: you still own it, so the risk figures above describe only the
           part of your portfolio the model could see.

        3. TSLY IS STRUCTURALLY UNUSUAL.
           It is a covered-call fund: it sells TSLA upside for option premium
           and distributes the proceeds. Three consequences the model cannot
           see: (a) its return distribution is deliberately ASYMMETRIC -- capped
           upside, largely uncapped downside -- so a mean and a variance do not
           describe it; (b) its economics depend on option-implied volatility
           levels that vary hugely over time, so its historical risk/return does
           not extrapolate; (c) its correlation to TSLA is nonlinear and
           regime-dependent. Even with correct total-return data, mean-variance
           is the wrong lens for this holding.

        4. VARIANCE IS NOT RISK.
           Mean-variance assumes a mean and a variance fully describe a return
           distribution -- i.e. approximate normality. Real returns have fat
           tails (crashes occur far more often than a normal distribution
           permits) and negative skew (drops are larger than rallies). Variance
           also penalizes upside deviation exactly as much as downside, which no
           investor actually does. Alternatives that address this include
           semivariance, CVaR, and drawdown-based optimization.

        5. THE PAST IS NOT THE FUTURE.
           Correlations are not constant -- they famously spike toward 1 during
           crises, exactly when diversification is most needed. A portfolio
           optimized on a calm period can be far less diversified than it looks
           the moment it matters.

        6. NO COSTS, NO TAXES, NO CONSTRAINTS.
           This ignores transaction costs, bid-ask spreads, and (importantly for
           a taxable Robinhood account) capital gains tax on rebalancing. The
           gap between your portfolio and the frontier is not free money -- it
           is a gross figure before all of those.

        WHAT TO ACTUALLY TAKE FROM THIS:
        The ROBUST findings are the ones that depend on covariances rather than
        means -- your risk concentration, your diversification ratio, your
        effective N, and which holdings are redundant with each other. Those are
        estimated far more reliably and they answer the question you actually
        asked: how diversified am I really?

        The FRAGILE finding is the exact optimal weight vector. Read it as a
        description of what happened to work in this specific window, not as a
        target.
        """
    )

    # ------------------------------------------------------------------ [10]
    header("STEP 10  --  WRITING THE SUMMARY")
    ctx = {
        "tickers": tickers, "stats_table": stats_table, "risk_contrib": rc,
        "current_weights": current_w, "max_sharpe_weights": w_maxsharpe,
        "min_var_weights": w_minvar, "mu": mu_s, "sigma": sigma_s, "corr": corr,
        "rf": RISK_FREE_RATE, "frontier": frontier, "resampled": resampled,
        "short_history": short_history, "window_start": prices.index[0],
        "window_end": prices.index[-1], "n_obs": len(returns),
        "excluded": excluded,
    }
    write_summary(ctx)

    with open(os.path.join(OUTPUT_DIR, "analysis_log.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(_TRANSCRIPT) + "\n")
    print(f"  [saved] {OUTPUT_DIR}/analysis_log.txt -- this entire walkthrough")

    print("")
    print("=" * 78)
    print("  DONE. Start with output/05_efficient_frontier.png, then read SUMMARY.md")
    print("=" * 78)
    print("")


if __name__ == "__main__":
    main()
