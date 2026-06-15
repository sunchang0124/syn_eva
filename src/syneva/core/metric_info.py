"""Plain-language descriptions for every metric and C dimension.

Single source of truth for the labels shown in the UI sidebar, the HTML
scorecard, and the downloadable reports.

- METRIC_INFO: a neutral "what it measures" sentence with no direction claim,
  because the direction of "good" differs between the normalized score and the
  raw statistic.
- RAW_HINT: how to read the *raw* statistic, including its scale and which
  direction is better (e.g. a distance is better when lower; an AUC is best
  near 0.5). Shown only in the "actual measured values" view.
- The normalized score is always 0-1 with 1 = ideal (higher is better); that
  direction is conveyed by the score view's banner and verdict badges.
"""

from __future__ import annotations

C_INFO: dict[str, str] = {
    "congruence": "Do the synthetic data's distributions and column relationships match the real data?",
    "coverage": "Does the synthetic data span the same categories, ranges, and variety as the real data?",
    "compliance": "Does the synthetic data protect privacy, without copying or exposing real individuals?",
    "utility": "Is the synthetic data as useful as the real data for training models?",
}

METRIC_INFO: dict[str, str] = {
    # Congruence — distribution and relationship match
    "ks_statistic": "How closely each numeric column's distribution matches the real data.",
    "tvd": "How closely the category frequencies match the real data.",
    "wasserstein": "How far numeric values would have to shift to match the real distribution.",
    "correlation_difference": "Whether the relationships (correlations) between columns are preserved.",
    "pmse": "Whether a model can tell real and synthetic rows apart from their distributions.",
    "sliced_wasserstein": "The overall distance between the real and synthetic data clouds across many directions.",
    "jsd": "How different the category distributions are, on an information scale.",
    "c2st": "Whether a trained classifier can separate real from synthetic rows.",
    # Coverage — variety and novelty
    "category_coverage": "Whether the synthetic data includes every category present in the real data.",
    "range_coverage": "Whether numeric values span the same range as the real data.",
    "novelty_rate": "The share of synthetic rows that are not exact duplicates of each other.",
    "entropy_ratio": "Whether categories are as varied as in the real data.",
    "alpha_precision_beta_recall": "Whether synthetic points land where real ones do and also cover the real spread.",
    "authenticity": "The share of synthetic rows that are not near-copies of a real row.",
    "pca_scatter": "A two-dimensional overlay of the real and synthetic data for visual inspection.",
    # Compliance — privacy and disclosure
    "dcr": "How far synthetic rows sit from the nearest real record.",
    "nndr": "Whether synthetic rows are closer to real records than real records are to one another.",
    "k_anonymity": "The size of the smallest group sharing the same sensitive values.",
    "identical_match_rate": "The share of synthetic rows that are exact copies of a real row.",
    "mia_auc": "How easily an attacker could guess whether a record was in the real data.",
    "dp_ledger": "The differential-privacy budget the generator declared, if any.",
    # Utility — usefulness for modeling
    "tstr_suite": "How well a model trained on synthetic data performs on real data, relative to training on real data.",
    "multi_target_utility": "The train-on-synthetic, test-on-real check applied to every column as a prediction target.",
    "feature_importance_spearman": "Whether a model finds the same features important in synthetic and real data.",
    "discriminative_score": "Whether a model can distinguish real from synthetic rows.",
}

# How to read the RAW statistic (scale + which direction is better). Shown only
# in the "actual measured values" view, where the raw value's direction often
# differs from the normalized score's "higher is better".
RAW_HINT: dict[str, str] = {
    "ks_statistic": "Ranges 0 to 1: 0 = identical distributions, larger = more different. Lower is better.",
    "tvd": "Ranges 0 to 1: 0 = identical frequencies, larger = more different. Lower is better.",
    "wasserstein": "A distance in standardized units: 0 = identical, larger = further apart. Lower is better.",
    "correlation_difference": "Average absolute difference in correlations: 0 = identical structure, larger = more distortion. Lower is better.",
    "pmse": "0 means the model cannot distinguish real from synthetic; larger = easier to tell apart. Lower is better.",
    "sliced_wasserstein": "A distance: 0 = identical data clouds, larger = further apart. Lower is better.",
    "jsd": "Ranges 0 to 1: 0 = identical distributions, larger = more different. Lower is better.",
    "c2st": "Classifier AUC: 0.5 = indistinguishable (ideal), 1.0 = perfectly separable. Closer to 0.5 is better.",
    "category_coverage": "Fraction of real categories present in the synthetic data, 0 to 1. Higher is better.",
    "range_coverage": "Fraction of the real numeric range that is covered, 0 to 1. Higher is better.",
    "novelty_rate": "Fraction of non-duplicate rows, 0 to 1. Higher is better.",
    "entropy_ratio": "Synthetic-to-real variety ratio, 0 to 1. Higher (closer to 1) is better.",
    "alpha_precision_beta_recall": "Two fractions from 0 to 1 (precision and recall). Higher is better on both.",
    "authenticity": "Fraction of rows that are not near-copies, 0 to 1. Higher is better.",
    "pca_scatter": "Visualization only; there is no numeric value to compare.",
    "dcr": "Distance to the nearest real record, in standardized units. Larger = more privacy. Higher is better.",
    "nndr": "Distance ratio near 1 means synthetic rows are no closer to real records than reals are to each other. Higher is safer.",
    "k_anonymity": "The smallest matching-group size; larger groups hide individuals better. Higher is better.",
    "identical_match_rate": "Fraction of exact copies of real rows, 0 to 1. Fewer copies is safer. Lower is better.",
    "mia_auc": "Attack AUC: 0.5 = no privacy leakage (ideal), 1.0 = full leakage. Lower is better.",
    "dp_ledger": "Privacy budget epsilon: smaller epsilon = stronger privacy guarantee. Informational.",
    "tstr_suite": "Synthetic-trained performance as a fraction of real-trained, around 1 when just as useful. Higher is better.",
    "multi_target_utility": "Average utility ratio across targets, around 1 when just as useful. Higher is better.",
    "feature_importance_spearman": "Rank correlation from -1 to 1; 1 = identical importance ordering. Higher is better.",
    "discriminative_score": "Classifier AUC: 0.5 = indistinguishable (ideal), 1.0 = separable. Closer to 0.5 is better.",
}


# Full, human-readable names (the code stays available as a small secondary tag).
METRIC_NAMES: dict[str, str] = {
    "ks_statistic": "Kolmogorov-Smirnov statistic",
    "tvd": "Total variation distance",
    "wasserstein": "Wasserstein-1 distance",
    "correlation_difference": "Correlation difference",
    "pmse": "Propensity mean squared error",
    "sliced_wasserstein": "Sliced Wasserstein distance",
    "jsd": "Jensen-Shannon divergence",
    "c2st": "Classifier two-sample test",
    "category_coverage": "Category coverage",
    "range_coverage": "Range coverage",
    "novelty_rate": "Novelty rate",
    "entropy_ratio": "Entropy ratio",
    "alpha_precision_beta_recall": "Alpha-precision and beta-recall",
    "authenticity": "Authenticity",
    "pca_scatter": "PCA scatter plot",
    "dcr": "Distance to closest record",
    "nndr": "Nearest-neighbor distance ratio",
    "k_anonymity": "k-anonymity",
    "identical_match_rate": "Identical match rate",
    "mia_auc": "Membership inference attack",
    "dp_ledger": "Differential-privacy ledger",
    "tstr_suite": "Train on synthetic, test on real",
    "multi_target_utility": "Multi-target utility",
    "feature_importance_spearman": "Feature-importance correlation",
    "discriminative_score": "Discriminative score",
}

# Readable labels for the per-metric detail rows. Anything not listed falls back
# to a generic prettifier (underscores to spaces, known acronyms upper-cased).
_SCALAR_LABELS: dict[str, str] = {
    "score": "Score (0-1, higher is better)",
    "p_value": "p-value",
    "p05_dcr": "5th-percentile distance",
    "median_dcr": "Median distance",
    "min_k": "Smallest group size (k)",
    "unique_groups": "Distinct groups",
    "identical_match_rate": "Exact-copy rate",
    "identical_matches": "Exact copies",
    "nndr_median": "Median distance ratio",
    "c2st_auc": "Classifier AUC",
    "alpha_precision": "Alpha-precision",
    "beta_recall": "Beta-recall",
    "spearman_rho": "Spearman correlation",
    "epsilon": "Epsilon",
    "delta": "Delta",
}
_SCALAR_ACRONYMS: dict[str, str] = {
    "ks": "KS",
    "tvd": "TVD",
    "pmse": "pMSE",
    "dcr": "DCR",
    "nndr": "NNDR",
    "jsd": "JSD",
    "mia": "MIA",
    "auc": "AUC",
    "c2st": "C2ST",
    "tstr": "TSTR",
    "trtr": "TRTR",
    "pca": "PCA",
    "dp": "DP",
}


def describe_metric(name: str) -> str:
    """Return a plain-language description for a metric, or '' if unknown."""
    return METRIC_INFO.get(name, "")


def display_name(name: str) -> str:
    """Return the full human-readable name for a metric, or the code itself."""
    return METRIC_NAMES.get(name, name)


def raw_hint(name: str) -> str:
    """Return how to read the raw statistic (scale + direction), or '' if unknown."""
    return RAW_HINT.get(name, "")


def verdict(score: float) -> tuple[str, str]:
    """Map a 0-1 score to a plain verdict word and a CSS class.

    Excellent >= 0.9, Good >= 0.75, Fair >= 0.5, otherwise Poor.
    """
    if score >= 0.9:
        return ("Excellent", "v-excellent")
    if score >= 0.75:
        return ("Good", "v-good")
    if score >= 0.5:
        return ("Fair", "v-fair")
    return ("Poor", "v-poor")


def humanize_scalar(key: str) -> str:
    """Turn a scalar key like 'mean_ks_statistic' into 'Mean KS statistic'."""
    if key in _SCALAR_LABELS:
        return _SCALAR_LABELS[key]
    words = [_SCALAR_ACRONYMS.get(p, p) for p in key.split("_")]
    text = " ".join(words)
    return text[:1].upper() + text[1:] if text else text


def describe_c(c: str) -> str:
    """Return a plain-language description for a C dimension, or '' if unknown."""
    return C_INFO.get(c, "")
