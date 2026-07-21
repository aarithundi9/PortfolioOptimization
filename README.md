# Markowitz Portfolio Optimization

A hands-on, heavily-commented implementation of **Modern Portfolio Theory**
(Markowitz mean-variance optimization) applied to a real 10-holding Robinhood
portfolio.

This is a **learning exercise, not investment advice.** The "optimal" weights it
produces are historical estimates fitted with hindsight — the code is written
specifically to show you *why* you shouldn't trust them.

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
python portfolio_optimization.py
```

Runs in about a minute (the bootstrap resampling is the slow part). Everything
lands in `output/`, plus a written interpretation in `SUMMARY.md`.

> **Note on Python:** this project needs a native CPython (python.org, pyenv, or
> the Microsoft Store build). A MinGW/MSYS2 Python will try to compile numpy and
> scipy from source and fail — PyPI ships no wheels for it. Check with
> `python -c "import sys; print(sys.platform)"` — you want `win32` on Windows.

---

## Configuring it for your own portfolio

Everything you need to edit is in the **config block** at the top of
[portfolio_optimization.py](portfolio_optimization.py). Nothing below it should
need changing.

**Paste your exact share counts from Robinhood into `SHARES`:**

```python
SHARES = {
    "SPY":   0.95,
    "VOOG":  4.4,
    ...
}
```

Your *current* portfolio weights are then computed automatically from
`shares × latest price` — you never type a weight or a dollar amount. Add or
remove tickers freely; every chart, table, and optimization adapts.

Other settings worth knowing:

| Setting | Default | What it does |
|---|---|---|
| `LOOKBACK_YEARS` | `3.0` | Estimation window length |
| `HISTORY_POLICY` | `"drop_young"` | How to handle assets with short histories (see below) |
| `MIN_HISTORY_YEARS` | `2.0` | Threshold used by `drop_young` |
| `RISK_FREE_RATE` | `0.043` | Sets the Sharpe hurdle and anchors the Capital Market Line |
| `N_PORTFOLIOS` | `50_000` | Monte Carlo sample size |
| `MAX_WEIGHT` | `0.25` | Per-asset cap for the constrained run |
| `THEME` | `"light"` | `"light"` or `"dark"` figure styling |

### The short-history problem (worth understanding)

WQTM launched in October 2025. If you align all ten series to a common date
range, **the whole window collapses to WQTM's ~8 months** — and every covariance
in the model is then estimated from 8 months of data, including SPY's.

That is not a small effect. Sample covariance error scales roughly with
`1/√T`, so going from 753 trading days to 193 nearly doubles the standard error
on all 45 pairwise covariances. The optimizer, which is a machine for finding
and exploiting extreme values, will then happily build a portfolio on top of
that noise.

So you get a choice:

- **`"drop_young"`** (default) — drop assets under `MIN_HISTORY_YEARS`, keep the
  long window. Covariances stay trustworthy; the dropped holding is invisible to
  the model even though you still own it.
- **`"truncate"`** — keep all assets, shorten the window to the common overlap.
  Nothing is discarded; everything is estimated from very little data.

Neither is right. The script prints the consequences of whichever you pick.
**Try it both ways** — the contrast is the lesson.

---

## What each output file shows

| File | What it is | What to look for |
|---|---|---|
| `05_efficient_frontier.png` | **The main chart.** Start here. | How far your star sits *below and right* of the black curve. Vertical gap = return given up at your risk level; horizontal gap = risk taken without compensation. |
| `01_price_history.png` | Growth of $1, total-return basis | Context for everything else. Best and worst performers are highlighted. |
| `02_correlation_heatmap.png` | Pairwise correlations | The tech cluster (SPY/VOOG/SMH/NVDA) glowing red, GLD near-white against everything. Also: **the entire blue half is empty** — nothing here hedges anything. |
| `03_covariance_heatmap.png` | Same structure, unnormalized | Correlation scaled by volatility — shows which pairings actually move the portfolio variance. |
| `04_monte_carlo.png` | 50,000 random long-only portfolios | The hard upper-left boundary of the cloud. That edge *is* the efficient frontier, discovered rather than derived. |
| `06_weights_comparison.png` | Your weights vs. three optimized ones | How aggressively the optimizer reshapes things — and how little the four portfolios agree with each other. |
| `07_risk_contributions.png` | Dollar weight vs. share of risk | The concentration lesson, quantified. Bars where green towers over blue are punching above their allocation. |
| `08_lookback_sensitivity.png` | Max-Sharpe weights at 1 / 2 / 3-year lookbacks | **The most important chart in the project.** Same optimizer, same assets — only the window changed. The instability is estimation error, not signal. |
| `09_shrinkage_comparison.png` | Sample vs. Ledoit-Wolf covariance | The shrunk frontier sits *lower*. That's the point: some of the naive frontier was fitted noise. |
| `10_resampled_weights.png` | Bootstrap distribution of optimal weights | The error bars. A weight whose 5–95 band spans 0% to 70% is a coin flip, not an allocation. |
| `11_weight_cap_comparison.png` | Uncapped vs. 25%-capped optimum | A crude but effective regularizer — worse in sample, usually better out of sample. |
| `analysis_log.txt` | The full console walkthrough | Every explanation the script printed, saved for re-reading. |
| `SUMMARY.md` | Written interpretation | Plain-English reading of *your* results, generated from the actual numbers. |

---

## How the code is organized

One file, read top to bottom as a lesson:

| Section | Contents |
|---|---|
| **Config** | Everything you edit |
| **§0** | Plumbing: color system, explanation printer, output folder |
| **§1** | Data acquisition; the auto-adjusted-price argument; unequal-history handling |
| **§2** | Returns → the expected return vector `μ` and covariance matrix `Σ` |
| **§3** | Portfolio math: `w'μ`, `√(w'Σw)`, Sharpe — the three functions everything else is built from |
| **§4** | Monte Carlo simulation of the feasible set |
| **§5** | Formal optimization: min-variance, max-Sharpe, efficient frontier |
| **§6** | Diversification ratio, effective N, risk contributions |
| **§7** | Ledoit-Wolf shrinkage (implemented from the paper, not imported) |
| **§8** | Bootstrap / resampled optimization |
| **§9** | Figures |
| **§10** | `SUMMARY.md` generation |

Each stage prints a plain-English explanation of what it computed, the formula,
and how to read the result.

---

## The caveats, in short

The script explains each of these at length where it applies. In brief:

1. **Estimation error dominates.** Expected returns from 1–3 years of history
   have standard errors of *several percentage points a year*. The optimizer
   treats them as exact and piles into whatever got lucky. Michaud's name for
   this is "error maximization."
2. **Short histories are unreliable.** WQTM (~8 months) and TSLY (~3 years) have
   thin data. A wrong correlation is worse than a wrong mean — it propagates
   through every cross term of `w'Σw`.
3. **TSLY doesn't fit the model.** It's a covered-call fund: capped upside,
   uncapped downside, economics that depend on option-implied volatility levels
   that won't repeat. A mean and a variance don't describe that payoff, even
   with correct total-return data.
4. **Variance isn't risk.** Mean-variance assumes roughly normal returns. Real
   returns have fat tails and negative skew, and variance penalizes upside
   deviation exactly as much as downside.
5. **The frontier is hindsight.** It shows what *would have been* optimal knowing
   the outcome. Its honest use is as a diagnostic of redundancy and
   concentration in what you already own — not as a target.

**What survives all of that:** the findings that depend on *covariances* rather
than *means* — risk concentration, diversification ratio, effective N, and which
holdings are redundant with each other. Those are estimated far more reliably,
and they answer the question actually worth asking: how diversified am I really?
