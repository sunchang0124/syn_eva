"""Plain-language descriptions for every metric and C dimension.

Single source of truth for the hover tooltips shown in the UI sidebar, the
HTML scorecard, and the downloadable reports. All scores are normalized so
that 1.0 is the ideal, so every description ends by noting that higher is
better (except visualization-only and informational metrics).
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
    "ks_statistic": "How closely each numeric column's distribution matches the real data. Higher is better.",
    "tvd": "How closely the category frequencies match the real data. Higher is better.",
    "wasserstein": "How far numeric values would have to shift to match the real distribution. Higher means less shift is needed.",
    "correlation_difference": "Whether the relationships between columns are preserved. Higher means the correlations match the real data.",
    "pmse": "Whether a model can tell real and synthetic rows apart. Higher means they look alike.",
    "sliced_wasserstein": "The overall distance between the real and synthetic data clouds across many directions. Higher means closer.",
    "jsd": "How similar the category distributions are on an information scale. Higher means more similar.",
    "c2st": "Whether a trained classifier can separate real from synthetic rows. Higher means it cannot tell them apart.",
    # Coverage — variety and novelty
    "category_coverage": "Whether the synthetic data includes every category present in the real data. Higher means fewer categories are missing.",
    "range_coverage": "Whether numeric values span the same range as the real data. Higher means the full range is covered.",
    "novelty_rate": "The share of synthetic rows that are not exact duplicates of each other. Higher means more genuinely distinct rows.",
    "entropy_ratio": "Whether categories are as varied as in the real data. Higher means similar variety; low means the synthetic data collapsed onto a few values.",
    "alpha_precision_beta_recall": "Whether synthetic points land where real ones do and also cover the real spread. Higher is better on both counts.",
    "authenticity": "The share of synthetic rows that are not near-copies of a real row. Higher means less memorization.",
    "pca_scatter": "A two-dimensional overlay of the real and synthetic data for visual inspection. No score.",
    # Compliance — privacy and disclosure
    "dcr": "How far synthetic rows sit from the nearest real person. Higher means more privacy distance.",
    "nndr": "Whether synthetic rows are suspiciously closer to real people than real people are to one another. Higher is safer.",
    "k_anonymity": "The size of the smallest group sharing the same sensitive values. Higher means individuals are harder to single out.",
    "identical_match_rate": "Whether synthetic rows are exact copies of real rows. Higher means fewer exact copies.",
    "mia_auc": "How easily an attacker could guess whether a record was in the real data. Higher means harder to guess.",
    "dp_ledger": "Reports the differential-privacy budget if the generator declared one. Informational.",
    # Utility — usefulness for modeling
    "tstr_suite": "How well a model trained on synthetic data performs on real data, compared with training on real data. Higher means the synthetic data is just as useful.",
    "multi_target_utility": "The same train-on-synthetic, test-on-real check applied to every column as a prediction target. Higher means broadly useful.",
    "feature_importance_spearman": "Whether a model finds the same features important in synthetic and real data. Higher means the signal is preserved.",
    "discriminative_score": "Whether a model can distinguish real from synthetic rows. Higher means they are indistinguishable.",
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
