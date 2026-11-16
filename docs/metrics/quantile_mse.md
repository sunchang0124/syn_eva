# `quantile_mse` (congruence / core)

**Scope:** per-column. **Requires real data:** True. **Data types:** ['static'].

## What it measures

How closely the shape of each numeric column in the synthetic data follows the
real column, from the centre out to the tails. The metric compares the two
columns quantile by quantile, so it picks up differences in spread and
skewness as well as shifts in location. Differences are measured in real
standard deviations, so the result doesn't depend on the column's units.

## How it's computed

For each numeric column, take the 19 quantiles $q = 0.05, 0.10, \dots, 0.95$ of
the real and the synthetic values, and standardise their differences by the
real (population) standard deviation $\sigma_\text{real}$:

$$
\text{QMSE} = \frac{1}{19} \sum_{q} \left(\frac{Q_\text{real}(q) - Q_\text{syn}(q)}{\sigma_\text{real}}\right)^2 .
$$

The score is $\max\!\left(0,\ 1 - \sqrt{\overline{\text{QMSE}}}\right)$, where
$\overline{\text{QMSE}}$ is the mean over the scored columns.

Missing values are dropped column by column. A column is left out, with a note,
when:

- it has no values left in the real or the synthetic data;
- it is constant in the real data. There is no spread to standardise by, so the
  difference is undefined ([#39](https://github.com/sunchang0124/syn_eva/issues/39)).

The metric is skipped when no numeric column is left. Columns that are not
numeric are not used.

## When to use it

Use it to check that numeric columns keep their shape, including the tails,
in a form that is easy to read per column. Keep in mind:

- **One column can dominate.** Per-column values are not capped and grow with the square of a shift, so one far-off column can bring the score to 0: a 3-SD shift in one of five columns already does ([#102](https://github.com/sunchang0124/syn_eva/issues/102)). Look at the per-column values to find it.
- **Constant columns are invisible.** Columns that are constant in the real data are not scored here. Check them with `range_coverage`.
- **Only the 5th to 95th percentiles count.** Differences in the outer 5% of either tail are not measured.

## Output scalars

- `score`: $\max(0,\ 1 - \sqrt{\overline{\text{QMSE}}})$, from 0 to 1, higher is better.
- `mean_quantile_mse`: $\overline{\text{QMSE}}$ in squared real standard deviations. 0 means identical quantiles; lower is better.

Per column:

- `quantile_mse`: QMSE for that column.

## Citation

Lautrup, A. D., Hyrup, T., Zimek, A., & Schneider-Kamp, P. (2024). SynthEval:
A framework for detailed utility and privacy evaluation of tabular synthetic
data. *Data Mining and Knowledge Discovery*.
https://doi.org/10.1007/s10618-024-01081-4
