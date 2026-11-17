# `novelty_rate` (coverage / core)

**Scope:** table-level. **Requires real data:** True. **Data types:** ['static'].

## What it measures

The share of synthetic rows that are not exact copies of a real row. A
generator that memorises and replays training records scores low; one that
produces new records scores high. This is *novelty* as it is defined for
generative models: the fraction of generated samples that do not occur in the
training data (Polykovskiy et al., 2020; Brown et al., 2019).

It is not *uniqueness*, the share of synthetic rows that are distinct from
each other. Synthetic rows that repeat one another but copy no real row all
count as novel, so a generator that outputs the same new row over and over
scores 1.0.

## How it's computed

Each row is turned into a key over the real data's columns, in the real data's
order. ID columns are left out, because a fresh identifier would make a copied
row look new; if every column is an ID column, all columns are kept. Missing
values are normalised so that two rows with a missing value in the same place
compare equal. Values must match exactly: `1.0` and `1.0000001` differ.

With $n$ synthetic rows, of which $c$ have a key that occurs in the real data,
`novelty_rate` is

$$
\frac{n - c}{n} .
$$

Every synthetic row counts, including repeats. The keys are the same ones
`identical_match_rate` uses, which is $c/n$, so the two metrics always add up
to 1. The metric
is skipped when the synthetic data is empty.

## When to use it

Use it to check that the generator isn't replaying training records. Keep in
mind:

- **Same signal as `identical_match_rate`.** It is that metric's complement,
  reported under coverage. Reading both adds no information.
- **Exact matches only.** A synthetic row that differs from a real row in one
  decimal place is novel. Use `dcr` or `nndr` to find rows that are near-copies.
- **Says nothing about diversity.** A collapsed generator that outputs one new
  row many times scores 1.0. Use `category_coverage`, `range_coverage` and
  `entropy_ratio` to check variety.

## Output scalars

- `score`: the same value as `novelty_rate`.
- `novelty_rate`: $(n - c)/n$, from 0 to 1, higher is better.
- `duplicates`: $c$, the number of synthetic rows that are exact copies of a
  real row. Despite the name, it does not count rows that repeat each other
  within the synthetic data.

## Citation

Polykovskiy, D., Zhebrak, A., Sanchez-Lengeling, B., Golovanov, S., Tatanov,
O., Belyaev, S., Kurbanov, R., Artamonov, A., Aladinskiy, V., Veselov, M.,
Kadurin, A., Johansson, S., Chen, H., Nikolenko, S., Aspuru-Guzik, A., &
Zhavoronkov, A. (2020). Molecular Sets (MOSES): A benchmarking platform for
molecular generation models. *Frontiers in Pharmacology*, 11, 565644.
https://doi.org/10.3389/fphar.2020.565644

Brown, N., Fiscato, M., Segler, M. H. S., & Vaucher, A. C. (2019). GuacaMol:
Benchmarking models for de novo molecular design. *Journal of Chemical
Information and Modeling*, 59(3), 1096–1108.
https://doi.org/10.1021/acs.jcim.8b00839
