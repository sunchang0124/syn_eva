# `jsd` (congruence / extended)

**Scope:** per-column. **Requires real data:** True. **Data types:** ['static'].

## What it measures

How far apart the category frequencies of each categorical column are in the
real and synthetic data. The Jensen-Shannon divergence (JSD) is a symmetric,
bounded version of the Kullback-Leibler divergence: it is 0 when both
columns have the same category frequencies and 1 when they share no
categories at all.

## How it's computed

For each categorical column, values are cast to strings and missing values
become their own category. Let $P$ and $Q$ be the real and synthetic category
frequencies over the union of categories seen in either set, and
$M = \tfrac{1}{2}(P + Q)$. Then

$$
\text{JSD}(P, Q) = \tfrac{1}{2} \sum_k P_k \log_2 \frac{P_k}{M_k}
                 + \tfrac{1}{2} \sum_k Q_k \log_2 \frac{Q_k}{M_k} .
$$

The logarithm is base 2, which bounds JSD to $[0, 1]$. With the natural
logarithm the upper bound would be $\ln 2 \approx 0.693$. $M_k > 0$ wherever
$P_k$ or $Q_k$ is, so a category present in only one set needs no smoothing.

The per-column values are averaged with equal weight, and

$$
\text{score} = 1 - \overline{\text{JSD}} .
$$

The metric is skipped when there are no categorical columns.

## When to use it

Use it to compare marginal category frequencies, column by column. Keep in mind:

- **It ignores relationships between columns.** Each column is compared on its own, so a generator that gets every marginal right but scrambles the joint distribution still scores 1.
- **Categories are matched by their string form.** `1` and `"1"` are the same category; `"Male"` and `"male"` are not.
- **Every column counts equally in the mean.** A badly generated column with two levels weighs the same as one with fifty.

## Output scalars

- `score`: $1 - \overline{\text{JSD}}$, from 0 to 1, higher is better.
- `mean_jsd`: the mean JSD over categorical columns, from 0 to 1, lower is better.
- Per column, `jsd`: that column's JSD, from 0 to 1.

## Citation

Lin, J. (1991). Divergence measures based on the Shannon entropy. *IEEE
Transactions on Information Theory*, 37(1), 145–151.
