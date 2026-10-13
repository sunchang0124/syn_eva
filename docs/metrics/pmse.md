# `pmse` (congruence / core)

**Scope:** table-level. **Requires real data:** True. **Data types:** ['static'].

## What it measures

Whether a classifier can tell real rows from synthetic ones. Real and synthetic
rows are pooled and a logistic regression predicts which set each row came
from. If the two datasets look alike, the predicted probabilities stay near the
synthetic share of the pool; if they don't, the probabilities spread out
towards 0 and 1. The propensity mean squared error (pMSE) measures that spread.

## How it's computed

Numeric columns are used as they are, with missing values set to 0.
Categorical columns are one-hot encoded, with missing values as their own
level. Boolean, datetime and ID columns are left out. Each feature is
standardized over the pooled data, and a logistic regression (main effects
only, fixed seed) is fitted to label real rows 0 and synthetic rows 1.

With $N$ pooled rows, synthetic share $c = n_\text{syn} / N$ and fitted
probabilities $\hat p_i$ for the same rows the model was trained on,

$$
\text{pMSE} = \frac{1}{N} \sum_{i=1}^{N} \left(\hat p_i - c\right)^2 .
$$

A classifier that separates the two sets perfectly predicts 0 on every real
row and 1 on every synthetic row, giving the largest possible value,
$(1-c)\,c^2 + c\,(1-c)^2 = c(1-c)$. The score divides by that maximum:

$$
\text{score} = \operatorname{clip}\!\left(1 - \frac{\text{pMSE}}{c(1-c)},\ 0,\ 1\right).
$$

So fully separable data scores 0 whatever the real/synthetic ratio. With equal
sizes the maximum is 0.25; with 1000 real and 100 synthetic rows it is 0.083.

The metric is skipped when a dataset is empty. If the classifier fails to fit,
the error propagates and the run records it as a metric error, not a score.

## When to use it

Use it as a single table-level check that picks up shifts in column means and
category frequencies. Keep in mind:

- **It only sees what a linear model can separate.** Differences in spread or shape are invisible: synthetic data drawn from $N(0, 3^2)$ against real $N(0, 1)$ scores 1.00, because no straight cut in $x$ separates the two. Interactions between columns are missed for the same reason. Pair it with per-column metrics such as `ks_statistic` or `wasserstein`, which compare whole distributions. `c2st` uses the same logistic model, so it shares this blind spot.
- **A perfect generator still scores a little below 1.** The model is fitted and scored on the same rows, so it fits noise and pMSE sits above 0 even when both sets come from one distribution. The gap grows with the number of one-hot columns and shrinks with the number of rows: with 200 rows per side and three 30-level categorical columns, identically distributed data scores 0.80 ([#61](https://github.com/sunchang0124/syn_eva/issues/61)). Compare scores only between tables of similar shape.
- **Missing numeric values count as 0.** A difference in missingness shows up only as extra mass at 0 after the fill, not as missingness.

## Output scalars

- `score`: $1 - \text{pMSE} / (c(1-c))$, clipped to 0 to 1, higher is better.
- `pmse`: the raw pMSE, from 0 to $c(1-c)$. 0 means the classifier cannot distinguish real from synthetic; lower is better. Its upper bound depends on $c$, so compare raw values only between runs with the same real/synthetic sizes.

## Citation

Woo, M.-J., Reiter, J. P., Oganian, A., & Karr, A. F. (2009). Global measures
of data utility for microdata masked for disclosure limitation. *Journal of
Privacy and Confidentiality*, 1(1), 111–124.

Snoke, J., Raab, G. M., Nowok, B., Dibben, C., & Slavkovic, A. (2018). General
and specific utility measures for synthetic data. *Journal of the Royal
Statistical Society: Series A*, 181(3), 663–688.
