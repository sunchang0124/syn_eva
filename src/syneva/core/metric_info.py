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


def describe_metric(name: str) -> str:
    """Return a plain-language description for a metric, or '' if unknown."""
    return METRIC_INFO.get(name, "")


def describe_c(c: str) -> str:
    """Return a plain-language description for a C dimension, or '' if unknown."""
    return C_INFO.get(c, "")
