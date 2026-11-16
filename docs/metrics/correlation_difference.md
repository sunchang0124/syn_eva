# `correlation_difference` (congruence / core)

**Scope:** pairwise. **Requires real data:** True. **Data types:** ['static'].

## What it measures

Whether the synthetic data keeps the relationships between columns that the real
data has. For every pair of numeric or categorical columns, the metric measures
how strongly the two columns are associated in each dataset and compares the
two. All three association measures it uses lie on a 0-to-1 strength scale, so
a change of the same size counts the same whichever kind of pair it is in.

## How it's computed

Each pair of columns gets one association measure, chosen by the column types:

| Pair | Measure | Range |
|---|---|---|
| numeric–numeric | Pearson correlation $r$ | $[-1, 1]$ |
| categorical–categorical | Cramér's $V$ (no continuity correction) | $[0, 1]$ |
| numeric–categorical | correlation ratio $\eta$ | $[0, 1]$ |

The correlation ratio is $\eta = \sqrt{SS_\text{between} / SS_\text{total}}$:
the share of the numeric column's spread that the categories explain, on the
square-root scale. For a two-level category, $\eta$ equals the absolute
point-biserial correlation, so it is directly comparable to $|r|$ and $V$.
($\eta^2$ is a share of variance, like $r^2$. Because
$|\eta_1^2 - \eta_2^2| = |\eta_1 - \eta_2|\,(\eta_1 + \eta_2)$, a difference on
that squared scale shrinks when $\eta_1 + \eta_2 < 1$ and grows when
$\eta_1 + \eta_2 > 1$, so these pairs would not be comparable with the others.)

For each pair the difference is

$$
d = \min\!\left(\left|a_\text{real} - a_\text{syn}\right|,\ 1\right),
$$

where $a$ is the pair's association measure. $V$ and $\eta$ differences are
already at most 1. For Pearson pairs the sign is kept, so a synthetic dataset
that reverses a relationship is penalised: $r$ going from $0.4$ to $-0.4$ gives
$d = 0.8$. Without the cap a sign flip could reach 2, so one reversed pair would
count up to twice as much as the worst categorical pair. When such pairs made up
most of the scored pairs, the mean difference could also exceed 1 and the score
would be clipped to 0, so those badly broken datasets would all score the same.
The cap is a design choice of this package
([#38](https://github.com/sunchang0124/syn_eva/issues/38)): any Pearson pair
with $|r_\text{real} - r_\text{syn}| \ge 1$ counts as the worst possible pair.

The score is $1 - \bar d$, where $\bar d$ is the mean of $d$ over all scored pairs.
Because every $d$ lies in $[0, 1]$, so does the score.

A pair is left out, with a note, when its association is undefined in the real
or the synthetic data:

- a numeric column in the pair is constant;
- a categorical column in the pair has a single level.

The metric is skipped when no pair is left. Columns that are neither numeric
nor categorical are not used.

## When to use it

Use it to check that the synthetic data reproduces the dependence structure of
the real data, pair by pair; per-column metrics such as `ks_statistic` or `tvd`
can't see this. Keep in mind:

- **Only linear and pairwise.** Pearson $r$ captures linear relationships only, and no measure here sees interactions among three or more columns.
- **Sign only for numeric pairs.** $V$ and $\eta$ have no direction, so a reversed relationship between a numeric and a categorical column (for example, group means swapping order) is only caught through the change in strength.
- **The mean hides single pairs.** With many columns, one badly broken pair moves the score little. Look at the per-pair `abs_corr_diff` values to find it.

## Output scalars

- `score`: $1 - \bar d$, from 0 to 1, higher is better.
- `mean_abs_corr_diff`: $\bar d$, from 0 to 1, lower is better.

Per pair (`"<col_a>|<col_b>"`):

- `abs_corr_diff`: $d$ for that pair, from 0 to 1.

## Citation

Pearson, K. (1895). Notes on regression and inheritance in the case of two
parents. *Proceedings of the Royal Society of London*, 58, 240–242.

Pearson, K. (1905). *On the general theory of skew correlation and non-linear
regression*. Drapers' Company Research Memoirs, Biometric Series II. Dulau and Co.

Cramér, H. (1946). *Mathematical Methods of Statistics*. Princeton University Press.
