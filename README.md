# syneva — 7 Cs scorecard for tabular synthetic data

`syneva` evaluates synthetic tabular data against its real counterpart using the
**7 Cs framework** ([Zamzmi et al. 2025](https://doi.org/10.1038/s44172-025-00450-1))
plus a parallel **Task-based ML utility** section. v0.1 supports static
(single-table) data with 14 core + 11 extended metrics.

## Install

```bash
pip install syneva
# optional extras: PDF rendering, interactive HTML
pip install 'syneva[pdf,plotly]'
```

## Quickstart

```python
import pandas as pd
import syneva

real = pd.read_parquet("real.parquet")
synthetic = pd.read_parquet("synthetic.parquet")

report = syneva.evaluate(real, synthetic)

print(report.aggregated)  # per-C score summary
report.to_html("scorecard.html")
report.to_pdf("scorecard.pdf")   # needs syneva[pdf]
report.to_json("scorecard.json")
```

## CLI

```bash
syn-eva evaluate --real real.parquet --synthetic synthetic.parquet --out ./report --pdf
```

## What's in the box

- **Congruence** (distribution alignment): KS, TVD, Wasserstein-1, correlation
  difference, pMSE. Extended: sliced Wasserstein, Jensen-Shannon, C2ST.
- **Coverage** (variability, novelty): category coverage, range coverage,
  novelty rate, entropy ratio. Extended: alpha-precision/beta-recall,
  authenticity, PCA scatter.
- **Compliance** (privacy/disclosure): DCR, NNDR, k-anonymity, identical
  match rate. Extended: MIA AUC, DP ledger pass-through.
- **Task-based utility**: TSTR/TRTR/utility ratio per `UtilityTask`. Extended:
  multi-target sweep, feature-importance Spearman correlation, discriminative
  score.

See `docs/superpowers/specs/2026-05-21-syneva-package-design.md` for the
complete design.

## License

Apache-2.0.
