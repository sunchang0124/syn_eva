# `sliced_wasserstein` (congruence / extended)

**Scope:** table-level. **Requires real data:** True. **Data types:** ['static'].

## What it measures

How far apart the real and synthetic data clouds are across all numeric columns
at once. The metric projects both datasets onto many random directions and
compares the two one-dimensional distributions along each direction, so it
picks up differences in how columns vary together as well as in each column on
its own. The distance is in units of real standard deviations, so it doesn't
depend on the columns' units.

## How it's computed

Each numeric column is standardized with the real mean and standard deviation,
$z = (x - \mu_\text{real}) / \sigma_\text{real}$, for both datasets. The scale
comes from the real data alone, so the synthetic data cannot move the yardstick
it is measured against. With $L = 50$ unit directions $\theta_\ell$ drawn
uniformly at random (fixed seed), the distance is the Monte Carlo estimate

$$
SW_1 = \frac{1}{L} \sum_{\ell=1}^{L} W_1\!\left(\theta_\ell^\top Z_\text{real},\ \theta_\ell^\top Z_\text{syn}\right),
$$

where $W_1$ is the one-dimensional Wasserstein-1 distance. The score is
$\max(0,\ 1 - SW_1 / 3)$.

Before projecting:

- Columns with no finite values in the real or the synthetic data are left out, with a note.
- Columns that are constant in the real data are left out, with a note: there is no spread to standardize by.
- Rows with a missing or infinite value in any remaining column are dropped (complete-case analysis). A note gives the count and share dropped on each side.
- A column that becomes constant once those rows are gone is left out too, with a note, and the rows are recomputed without it, so rows dropped only because of that column are kept.

The metric is skipped when:

- a dataset is empty;
- no usable column remains;
- fewer than 2 real rows or no synthetic rows are left;
- the distance is not finite, which happens when synthetic values are so far outside the real range that standardizing them overflows.

## When to use it

Use it as a single multivariate check of numeric columns that is cheaper than a
kernel or classifier test. Keep in mind:

- **Width dilutes the score.** A shift in one column is averaged over random directions, so its effect on the score shrinks as the number of columns grows: a 4-SD shift in one column scores 0.09 with 2 columns and 0.76 with 20 ([#58](https://github.com/sunchang0124/syn_eva/issues/58)). Pair it with per-column metrics such as `wasserstein` or `ks_statistic` to find which column is off.
- **Constant columns are invisible.** Columns that are constant in the real data are not scored here. Check them with `range_coverage`.
- **Only numeric columns count.** Categorical columns are ignored.

## Output scalars

- `score`: $\max(0,\ 1 - SW_1 / 3)$, from 0 to 1, higher is better.
- `sliced_wasserstein`: $SW_1$ in real standard deviations. 0 means identical data clouds; lower is better.

## Citation

Rabin, J., Peyré, G., Delon, J., & Bernot, M. (2011). Wasserstein barycenter
and its application to texture mixing. In *Scale Space and Variational Methods
in Computer Vision* (LNCS 6667, pp. 435–446). Springer.

Bonneel, N., Rabin, J., Peyré, G., & Pfister, H. (2015). Sliced and Radon
Wasserstein barycenters of measures. *Journal of Mathematical Imaging and
Vision*, 51(1), 22–45.
