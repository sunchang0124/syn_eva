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

The model is fitted and scored on the same rows, so it fits noise: pMSE sits
above 0 even when both sets come from one distribution, and the more parameters
the model has relative to the number of rows, the higher it sits. The value it
would take for data that can't be told apart, $\text{pMSE}_0$, is estimated by
permutation (Snoke et al., 2018): the real/synthetic labels are shuffled 20
times (fixed seed), the same model is refitted on each shuffle, and
$\text{pMSE}_0$ is the mean pMSE over the shuffles.

At the other end, a classifier that separates the two sets perfectly predicts 0
on every real row and 1 on every synthetic row, giving the largest possible
value, $(1-c)\,c^2 + c\,(1-c)^2 = c(1-c)$. The score is rescaled between the
two:

$$
\text{score} = \operatorname{clip}\!\left(1 - \frac{\text{pMSE} - \text{pMSE}_0}{c(1-c) - \text{pMSE}_0},\ 0,\ 1\right).
$$

So data that a second sample from the same distribution would match scores
about 1, and fully separable data scores 0, whatever the table's shape and the
real/synthetic ratio. With equal sizes the maximum is 0.25; with 1000 real and
100 synthetic rows it is 0.083.

The metric is skipped when:

- a dataset is empty;
- no numeric or categorical column is left to classify on;
- the model has at least as many parameters (independent features plus the intercept) as pooled rows. A linear classifier can then split the rows in every possible way (Cover, 1965), so it fits any labelling, shuffled ones included, and pMSE says nothing about the data. This happens with categorical columns that have nearly one level per row.

If the classifier fails to fit, the error propagates and the run records it as
a metric error, not a score.

## When to use it

Use it as a single table-level check that picks up shifts in column means and
category frequencies. Keep in mind:

- **It only sees what a linear model can separate.** Differences in spread or shape are invisible: synthetic data drawn from $N(0, 3^2)$ against real $N(0, 1)$ scores 1.00, because no straight cut in $x$ separates the two. Interactions between columns are missed for the same reason. Pair it with per-column metrics such as `ks_statistic` or `wasserstein`, which compare whole distributions. `c2st` uses the same logistic model, so it shares this blind spot.
- **Large models leave less room.** The score's range runs from $\text{pMSE}_0$ to $c(1-c)$. With many one-hot columns and few rows, $\text{pMSE}_0$ is a large share of that range, so moderate differences move the score less. `pmse_null` shows how much room was left.
- **The permutation reference assumes all columns are synthesized.** Shuffling treats every column as exchangeable between the two sets. When some columns are copied unchanged from the real data, this is not the right reference (Snoke et al., 2018).
- **Missing numeric values count as 0.** A difference in missingness shows up only as extra mass at 0 after the fill, not as missingness.

## Output scalars

- `score`: $1 - (\text{pMSE} - \text{pMSE}_0) / (c(1-c) - \text{pMSE}_0)$, clipped to 0 to 1, higher is better.
- `pmse`: the raw pMSE, from 0 to $c(1-c)$. 0 means the classifier cannot distinguish real from synthetic; lower is better. Its upper bound depends on $c$, so compare raw values only between runs with the same real/synthetic sizes.
- `pmse_null`: $\text{pMSE}_0$, the permutation estimate of pMSE for data that can't be told apart.
- `pmse_ratio`: $\text{pMSE} / \text{pMSE}_0$ (Snoke et al., 2018). About 1 means the synthetic data is as close to the real data as a second sample from the same distribution would be; larger is worse. Well below 1 means the synthetic data is closer to the real data than an independent sample would be, as when synthetic rows are perturbed copies of real ones; check the privacy metrics such as `dcr` in that case. With few features, pMSE for data that can't be told apart varies a lot from sample to sample (its standard deviation is about $\sqrt{2/(k-1)}$ times its mean for $k-1$ independent features; Snoke et al., 2018), so the ratio can land far from 1 by chance. Read it together with `score`.

## Citation

Cover, T. M. (1965). Geometrical and statistical properties of systems of
linear inequalities with applications in pattern recognition. *IEEE
Transactions on Electronic Computers*, EC-14(3), 326–334.
https://doi.org/10.1109/PGEC.1965.264137

Woo, M.-J., Reiter, J. P., Oganian, A., & Karr, A. F. (2009). Global measures
of data utility for microdata masked for disclosure limitation. *Journal of
Privacy and Confidentiality*, 1(1), 111–124.

Snoke, J., Raab, G. M., Nowok, B., Dibben, C., & Slavkovic, A. (2018). General
and specific utility measures for synthetic data. *Journal of the Royal
Statistical Society: Series A*, 181(3), 663–688.
