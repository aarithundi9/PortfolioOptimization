# What This Analysis Says About My Portfolio

*Generated 2026-07-20 from 2023-07-18 to 2026-07-17 (752 trading days, 2.98 years), risk-free rate 4.30%.*

> This is a coursework exercise in mean-variance optimization. Every
> number below is an estimate from a specific slice of history, not a
> forecast and not advice. The final section explains exactly how much
> to distrust it, which is the real point of the exercise.

## 1. Where my portfolio actually sits

| | Annual return | Annual vol | Sharpe | Diversification ratio | Effective N |
|---|---|---|---|---|---|
| **Current (yours)** | 29.81% | 21.21% | 1.203 | 1.24 | 6.1 |
| **Equal weight** | 28.89% | 19.79% | 1.243 | 1.36 | 9.0 |
| **Min variance** | 18.17% | 11.39% | 1.218 | 1.37 | 2.4 |
| **Max Sharpe** | 30.07% | 15.25% | 1.690 | 1.59 | 3.6 |

My portfolio earned an estimated **29.81%** annualized with **21.21%** volatility, a Sharpe ratio of **1.203**.

**The distance to the frontier.** At my current risk level (21.21% volatility), the efficient frontier reached **38.64%** — about **8.8 percentage points** more return for the *same* risk. Read the other way: to earn my current 29.81% return, an efficient portfolio would have needed only **15.11%** volatility instead of 21.21% — about **6.1 points less risk**.

That gap is the cost of my particular mix — mostly the cost of holding several things that are really the same bet. It is *not* money I could have captured in real life, because the frontier was drawn with hindsight (see section 5).

## 2. The concentration problem: where my risk really is

The model sees **9 tickers** (of the 10 I hold — WQTM was excluded for lack of history), but the effective number of independent positions is only **6.1** by dollar weight — and the risk decomposition tells a sharper story still.

| Asset | Dollar weight | Share of risk | Difference |
|---|---|---|---|
| SMH | 22.8% | 36.5% | +13.7% ⚠️ |
| SPY | 25.6% | 17.1% | -8.5% |
| TSLY | 8.1% | 11.7% | +3.6% |
| VOOG | 12.9% | 11.3% | -1.6% |
| NVDA | 5.4% | 9.6% | +4.2% |
| GOOGL | 7.3% | 6.4% | -0.9% |
| VXUS | 9.1% | 5.1% | -4.0% |
| GLD | 7.6% | 2.0% | -5.6% |
| SCHD | 1.2% | 0.3% | -0.9% |

**Top three risk sources:** SMH (36.5% of total risk), SPY (17.1% of total risk), TSLY (11.7% of total risk) — together **65.3%** of all portfolio risk from **56.5%** of the money.

**Punching above their weight:** SMH (+13.7%), NVDA (+4.2%), TSLY (+3.6%). These holdings determine more of my outcome than their dollar allocation suggests, because they are both volatile *and* highly correlated with the rest of what I own — so their moves reinforce rather than offset everything else.

**Diversification ratio: 1.24.** This is low. It means my 9 holdings behave close to one big position — the correlations between them are high enough that spreading money across them bought relatively little risk reduction.

## 3. What the optimizer wants to change, and why

| Asset | My weight | Min-variance | Max-Sharpe | Avg. correlation with rest |
|---|---|---|---|---|
| SPY | 25.6% | 13.0% | 0.0% | 0.65 |
| VOOG | 12.9% | 0.0% | 0.0% | 0.65 |
| SCHD | 1.2% | 59.3% | 29.7% | 0.34 |
| VXUS | 9.1% | 1.9% | 0.0% | 0.56 |
| GLD | 7.6% | 23.7% | 35.7% | 0.17 |
| GOOGL | 7.3% | 2.2% | 19.5% | 0.41 |
| NVDA | 5.4% | 0.0% | 15.2% | 0.47 |
| SMH | 22.8% | 0.0% | 0.0% | 0.58 |
| TSLY | 8.1% | 0.0% | 0.0% | 0.41 |

**Wants more of:** SCHD (+28.5%), GLD (+28.0%), GOOGL (+12.2%), NVDA (+9.7%).

- **SCHD**: it is among the least correlated things I own (avg correlation 0.34), so it genuinely damps portfolio variance.
- **GLD**: it is among the least correlated things I own (avg correlation 0.17), so it genuinely damps portfolio variance.
- **GOOGL**: it is among the least correlated things I own (avg correlation 0.41), so it genuinely damps portfolio variance; and it posted one of the highest returns in this window (39.3% annualized) — which is precisely the kind of input that is mostly estimation noise.
- **NVDA**: it posted one of the highest returns in this window (59.7% annualized) — which is precisely the kind of input that is mostly estimation noise.

**Wants less of:** SPY (-25.6%), SMH (-22.8%), VOOG (-12.9%), VXUS (-9.1%).

Usually this is not because the optimizer dislikes the asset in isolation, but because it is *redundant*: another holding already supplies the same exposure with a better risk/return profile over this window. When two assets are ~0.9 correlated, mean-variance optimization will nearly always pick one and discard the other, even when their expected returns are nearly identical. That all-or-nothing behavior is a known pathology, not wisdom.

Note the min-variance column ignores expected returns entirely — it is built from the covariance matrix alone. It is the more trustworthy of the two optimized portfolios for exactly that reason (Sharpe 1.218 vs 1.690 in-sample, but far less dependent on return estimates that are mostly noise).

## 4. What the correlation matrix shows

**Most correlated pairs** (these are the redundant ones):

- SPY / VOOG: **0.96**
- VOOG / SMH: **0.86**
- SPY / SMH: **0.81**
- NVDA / SMH: **0.80**
- VOOG / NVDA: **0.78**

**Least correlated pairs** (these are what actually diversifies):

- GLD / GOOGL: **0.14**
- SCHD / NVDA: **0.13**
- SCHD / GLD: **0.11**
- GLD / NVDA: **0.09**
- GLD / TSLY: **0.08**

The single most independent holding by average correlation is **GLD** (0.17); the most redundant is **SPY** (0.65).

## 5. How much of this should I believe?

Honestly: the *structure* a lot, the *numbers* very little. In order of how much they should worry me —

**Estimation error is the dominant problem.** Mean-variance optimization has been called an "error maximizer" for good reason. The optimizer treats my estimated expected returns as facts, when a 3-year sample mean of a volatile asset has a standard error of several percentage points a year. Whatever happened to do best in this window gets loaded up on. See `08_lookback_sensitivity.png`: the same optimizer on the same assets produces materially different portfolios depending only on whether I look back 1, 2, or 3 years. None of those windows is more "correct" than the others — which means the weight differences between them are pure noise being reported as precision.

The bootstrap makes this concrete (`10_resampled_weights.png`). Across 300 resamples of my own return history, **GLD**'s max-Sharpe weight ranged from 0% to 69%. A weight that unstable is not an allocation decision; it is a coin flip reported to two decimals.

**Short history distorts specific holdings.** WQTM has far less history than the rest. Its estimated mean, variance, and — worst of all — correlation with everything else is built on very few observations. The optimizer has no way to know this and would state an opinion about it with exactly the same confidence it states one about SPY. Discount that view heavily — and note that excluding an asset does not make the problem go away, it just moves it: I still own it, the model simply cannot see it.

**TSLY is structurally unlike the others.** It is a covered-call / option-income fund: it sells away TSLA's upside in exchange for premium, and returns most of its economics as distributions. Even on a total-return basis, its historical distribution is *engineered* — capped on the upside, largely uncapped on the downside, with a return stream whose shape depends on option-implied volatility levels that will not repeat. A historical mean and variance describe what that structure did under one volatility regime; they do not extrapolate.

**Variance is not risk.** Mean-variance assumes returns are adequately described by a mean and a variance — effectively, that they are roughly normal. Real returns have fat tails (crashes far more often than a normal distribution allows) and negative skew (down moves bigger than up moves). Variance also penalizes upside deviation identically to downside, which no actual investor does. For a portfolio containing an option-income fund with deliberately asymmetric payoffs, this assumption is not a technicality — it is materially wrong for that holding.

**The frontier is drawn with hindsight.** It shows what *would have been* optimal, knowing the outcome. Nobody could have selected that portfolio in advance. The honest use of the frontier is as a diagnostic of redundancy and concentration in what I already own — not as a target.

## 6. What I'd actually take away

1. **The concentration finding is the robust one.** Risk contributions depend on covariances, which are estimated far more reliably than means. That SMH, SPY and TSLY account for 65% of my risk on 56% of my money is a real structural fact about my portfolio, not an artifact.
2. **Overlapping holdings are doing less than the ticker count suggests.** An effective N of 6.1 and a diversification ratio of 1.24 both say the same thing: several of these positions are the same bet wearing different names.
3. **Ignore the exact optimal weights.** They are the least reliable output here. The min-variance portfolio, which never touches expected returns, is the one worth studying.
4. **The diversifiers are the ones with low average correlation**, not the ones with the best returns — GLD and SCHD do more for portfolio risk per dollar than their standalone performance suggests.

---

*Figures in `output/`. Full console walkthrough in `output/analysis_log.txt`. Regenerate with `python portfolio_optimization.py`.*
