# `dimension_wise_means` (congruence / core)

**Scope:** per-column. **Requires real data:** True. **Data types:** ['static'].

## What it measures

Whether each numeric column in the synthetic data is centred where the real
column is. The metric compares the two column means, measured in real standard
deviations, so the result doesn't depend on the column's units. It only looks
at location: a synthetic column with the right mean but the wrong spread or
shape scores perfectly.

## How it's computed

For each numeric column, with $\bar x$ the column mean and $s_\text{real}$ the
real sample standard deviation:

$$
d = \frac{\left|\bar x_\text{real} - \bar x_\text{syn}\right|}{s_\text{real}} .
$$

The score is $\max(0,\ 1 - \bar d)$, where $\bar d$ is the mean of $d$ over the
scored columns. This adapts the dimension-wise probability comparison of Choi
et al. (2017), which compares per-feature proportions of binary features, to
standardised differences in the means of numeric columns.

Missing values are dropped column by column. A column is left out, with a note,
when:

- it has fewer than 2 values in the real or the synthetic data;
- it is constant in the real data. There is no spread to standardise by, so the
  difference is undefined ([#39](https://github.com/sunchang0124/syn_eva/issues/39)).

The metric is skipped when no numeric column is left. Columns that are not
numeric are not used.

## When to use it

Use it as a quick first check that numeric columns are in the right place.
Keep in mind:

- **Location only.** Pair it with `quantile_mse`, `ks_statistic` or `wasserstein` to check spread and shape.
- **One column can dominate.** Per-column values are not capped, so one far-off column can bring the score to 0: a 10-SD shift in one of five columns does ([#102](https://github.com/sunchang0124/syn_eva/issues/102)). Look at the per-column values to find it.
- **Constant columns are invisible.** Columns that are constant in the real data are not scored here. Check them with `range_coverage`.

## Output scalars

- `score`: $\max(0,\ 1 - \bar d)$, from 0 to 1, higher is better.
- `mean_abs_std_diff`: $\bar d$ in real standard deviations. 0 means identical means; lower is better.

Per column:

- `std_mean_diff`: $d$ for that column.

## Citation

Choi, E., Biswal, S., Malin, B., Duke, J., Stewart, W. F., & Sun, J. (2017).
Generating multi-label discrete patient records using generative adversarial
networks. In *Proceedings of the 2nd Machine Learning for Healthcare
Conference* (PMLR 68, pp. 286–305).

Lautrup, A. D., Hyrup, T., Zimek, A., & Schneider-Kamp, P. (2024). SynthEval:
A framework for detailed utility and privacy evaluation of tabular synthetic
data. *Data Mining and Knowledge Discovery*.
https://doi.org/10.1007/s10618-024-01081-4
