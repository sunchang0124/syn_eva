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
    "fairness": "Does the synthetic data preserve the real data's fairness across protected groups?",
    "preservation": "Does the synthetic data preserve rare categories, distribution tails, and small subgroups of the real data?",
}

METRIC_INFO: dict[str, str] = {
    # Congruence — distribution and relationship match
    "ks_statistic": "How closely each numeric column's distribution matches the real data.",
    "tvd": "How closely the category frequencies match the real data.",
    "wasserstein": "How far numeric values would have to shift to match the real distribution.",
    "correlation_difference": "Whether the relationships (correlations) between columns are preserved.",
    "pmse": "Whether a model can tell real and synthetic rows apart from their distributions.",
    "ci_overlap": "How much the 95% confidence intervals of column means overlap between real and synthetic data.",
    "hellinger": "Per-column distributional distance between real and synthetic data.",
    "quantile_mse": "Agreement of the quantile functions of numeric columns, including the tails.",
    "dimension_wise_means": "Whether each numeric column's mean is preserved.",
    "sliced_wasserstein": "The overall distance between the real and synthetic data clouds across many directions.",
    "jsd": "How different the category distributions are, on an information scale.",
    "c2st": "Whether a trained classifier can separate real from synthetic rows.",
    "mutual_information_difference": "Whether the pairwise dependency structure between columns is preserved.",
    "mmd": "Overall multivariate distributional discrepancy between real and synthetic data, via a kernel.",
    # Coverage — variety and novelty
    "category_coverage": "Whether the synthetic data includes every category present in the real data.",
    "range_coverage": "Whether numeric values span the same range as the real data.",
    "novelty_rate": "The share of synthetic rows that are not exact duplicates of each other.",
    "entropy_ratio": "Whether categories are as varied as in the real data.",
    "alpha_precision_beta_recall": "Whether synthetic points land where real ones do and also cover the real spread.",
    "authenticity": "The share of synthetic rows that are not near-copies of a real row.",
    "pca_scatter": "A two-dimensional overlay of the real and synthetic data for visual inspection.",
    "nn_adversarial_accuracy": "Whether synthetic points are too close (memorization) or too far (poor coverage), judged by nearest-neighbour separability.",
    # Compliance — privacy and disclosure
    "dcr": "How far synthetic rows sit from the nearest real record.",
    "nndr": "Whether synthetic rows sit much closer to one real record than to any other, a sign of memorization.",
    "k_anonymity": "The size of the smallest group sharing the same sensitive values.",
    "identical_match_rate": "The share of synthetic rows that are exact copies of a real row.",
    "hitting_rate": "How often a real record has a near-identical synthetic counterpart.",
    "epsilon_identifiability": "Whether real records are closer to a synthetic record than to their own nearest real neighbour.",
    "attribute_disclosure": "Whether a sensitive attribute can be inferred from the quasi-identifiers via the nearest synthetic record.",
    "mia_auc": "How easily an attacker could guess whether a record was in the real data.",
    "dp_ledger": "The differential-privacy budget the generator declared, if any.",
    # Fairness — outcome-rate preservation across protected groups
    "statistical_parity": "Whether the synthetic data preserves the real data's outcome-rate gap between protected groups.",
    # Preservation — minorities, tails, and subgroups
    "rare_category_retention": "Whether categories that are rare in the real data survive into the synthetic data.",
    "tail_coverage": "Whether synthetic values reach into the extreme tails of each numeric column.",
    "minority_class_density": "Whether the least-frequent class of each categorical column keeps its share of the data.",
    "subgroup_fidelity": "Whether declared subgroups keep their size and their internal distributions in the synthetic data.",
    "minority_utility_gap": "Whether a model trained on synthetic data serves declared subgroups as well as it serves the overall population.",
    "minority_privacy_risk": "Whether disclosure risk concentrates on declared subgroups rather than spreading evenly across the data.",
    # Utility — usefulness for modeling
    "tstr_suite": "How well a model trained on synthetic data performs on real data, relative to training on real data.",
    "multi_target_utility": "The train-on-synthetic, test-on-real check applied to every column as a prediction target.",
    "model_panel_utility": "Trains linear, random-forest, and gradient-boosting models on the synthetic data and on the real data, then compares how well each performs on a real test set; reports the average ratio and the per-family breakdown.",
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
    "ci_overlap": "Overlap fraction from 0 to 1: 1 = fully overlapping intervals, 0 = disjoint. Higher is better.",
    "hellinger": "Ranges 0 to 1: 0 = identical distributions, larger = more different. Lower is better.",
    "quantile_mse": "Standardized mean squared error across quantiles: 0 = identical, larger = more deviation. Lower is better.",
    "dimension_wise_means": "Standardized gap between real and synthetic column means: 0 = identical, larger = more shifted. Lower is better.",
    "sliced_wasserstein": "A distance: 0 = identical data clouds, larger = further apart. Lower is better.",
    "jsd": "Ranges 0 to 1: 0 = identical distributions, larger = more different. Lower is better.",
    "c2st": "Classifier AUC: 0.5 = indistinguishable (ideal), 1.0 = perfectly separable. Closer to 0.5 is better.",
    "mutual_information_difference": "Mean absolute difference in normalized mutual information: 0 = identical structure, larger = more distortion. Lower is better.",
    "mmd": "0 = identical distributions, larger = more different. Lower is better.",
    "category_coverage": "Fraction of real categories present in the synthetic data, 0 to 1. Higher is better.",
    "range_coverage": "Fraction of the real numeric range that is covered, 0 to 1. Higher is better.",
    "novelty_rate": "Fraction of non-duplicate rows, 0 to 1. Higher is better.",
    "entropy_ratio": "Synthetic-to-real variety ratio, 0 to 1. Higher (closer to 1) is better.",
    "alpha_precision_beta_recall": "Two fractions from 0 to 1 (precision and recall). Higher is better on both.",
    "authenticity": "Fraction of rows that are not near-copies, 0 to 1. Higher is better.",
    "pca_scatter": "Visualization only; there is no numeric value to compare.",
    "nn_adversarial_accuracy": "Adversarial accuracy from 0 to 1: 0.5 = indistinguishable (ideal), toward 1 = too separable, toward 0 = memorized. Closer to 0.5 is better.",
    "dcr": "Distance to the nearest real record, in standardized units. Larger = more privacy. Higher is better.",
    "nndr": "Nearest over second-nearest real-record distance, 0 to 1. Near 0 means synthetic rows hug single real records. Higher is safer.",
    "k_anonymity": "The smallest matching-group size; larger groups hide individuals better. Higher is better.",
    "identical_match_rate": "Fraction of exact copies of real rows, 0 to 1. Fewer copies is safer. Lower is better.",
    "hitting_rate": "Fraction of real records reproduced by the synthetic data, 0 to 1. Lower is better.",
    "epsilon_identifiability": "Fraction of real records that are identifiable, 0 to 1. Lower is better.",
    "attribute_disclosure": "Fraction of records whose sensitive attribute is correctly inferred, 0 to 1. Lower is better.",
    "mia_auc": "Attack AUC: 0.5 = no privacy leakage (ideal), 1.0 = full leakage. Lower is better.",
    "dp_ledger": "Privacy budget epsilon: smaller epsilon = stronger privacy guarantee. Informational.",
    "tstr_suite": "Synthetic-trained performance as a fraction of real-trained, around 1 when just as useful. Higher is better.",
    "multi_target_utility": "Average utility ratio across targets, around 1 when just as useful. Higher is better.",
    "model_panel_utility": "Average utility ratio across model families, around 1 when synthetic is just as useful as real. Higher is better.",
    "feature_importance_spearman": "Rank correlation from -1 to 1; 1 = identical importance ordering. Higher is better.",
    "discriminative_score": "Classifier AUC: 0.5 = indistinguishable (ideal), 1.0 = separable. Closer to 0.5 is better.",
    "statistical_parity": "Drift = |synthetic parity gap - real parity gap|, 0 to 1. 0 means the real fairness structure is preserved. Lower is better.",
    "rare_category_retention": "Mean retention of rare categories, 0 to 1: 1 = every rare category keeps its real frequency, 0 = all rare categories lost. Higher is better.",
    "tail_coverage": "Fraction of the expected tail mass the synthetic data reproduces, 0 to 1: 1 = both tails fully populated, 0 = tails empty. Higher is better.",
    "minority_class_density": "Symmetric density ratio of each column's least-frequent class, 0 to 1: 1 = share preserved, 0 = class vanished; over-representation is penalized the same as under-representation. Higher is better.",
    "subgroup_fidelity": "Even blend of subgroup-size preservation and within-subgroup distribution match, 0 to 1: 1 = subgroup fully preserved, 0 = subgroup erased. Higher is better.",
    "minority_utility_gap": "Excess gap = synthetic-trained performance gap minus real-trained gap, floored at 0. 0 means training on synthetic data costs the subgroup nothing beyond what real data already would. Lower is better.",
    "minority_privacy_risk": "Ratio of the subgroup's median distance-to-nearest-synthetic-record to the overall median, capped at 1: 1 = no concentrated risk, near 0 = subgroup members are much closer to synthetic records than average. Higher is better.",
}


# Full, human-readable names (the code stays available as a small secondary tag).
METRIC_NAMES: dict[str, str] = {
    "ks_statistic": "Kolmogorov-Smirnov statistic",
    "tvd": "Total variation distance",
    "wasserstein": "Wasserstein-1 distance",
    "correlation_difference": "Correlation difference",
    "pmse": "Propensity mean squared error",
    "ci_overlap": "Confidence-interval overlap",
    "hellinger": "Hellinger distance",
    "quantile_mse": "Quantile MSE",
    "dimension_wise_means": "Dimension-wise means",
    "sliced_wasserstein": "Sliced Wasserstein distance",
    "jsd": "Jensen-Shannon divergence",
    "c2st": "Classifier two-sample test",
    "mutual_information_difference": "Mutual-information difference",
    "mmd": "Maximum mean discrepancy",
    "category_coverage": "Category coverage",
    "range_coverage": "Range coverage",
    "novelty_rate": "Novelty rate",
    "entropy_ratio": "Entropy ratio",
    "alpha_precision_beta_recall": "Alpha-precision and beta-recall",
    "authenticity": "Authenticity",
    "pca_scatter": "PCA scatter plot",
    "nn_adversarial_accuracy": "Nearest-neighbour adversarial accuracy",
    "dcr": "Distance to closest record",
    "nndr": "Nearest-neighbor distance ratio",
    "k_anonymity": "k-anonymity",
    "identical_match_rate": "Identical match rate",
    "hitting_rate": "Hitting rate",
    "epsilon_identifiability": "Epsilon identifiability risk",
    "attribute_disclosure": "Attribute disclosure risk",
    "mia_auc": "Membership inference attack",
    "dp_ledger": "Differential-privacy ledger",
    "tstr_suite": "Train on synthetic, test on real",
    "multi_target_utility": "Multi-target utility",
    "model_panel_utility": "Model-panel utility",
    "feature_importance_spearman": "Feature-importance correlation",
    "discriminative_score": "Discriminative score",
    "statistical_parity": "Statistical parity difference",
    "rare_category_retention": "Rare-category retention",
    "tail_coverage": "Tail coverage",
    "minority_class_density": "Minority-class density",
    "subgroup_fidelity": "Subgroup fidelity",
    "minority_utility_gap": "Minority utility gap",
    "minority_privacy_risk": "Minority privacy risk",
}

# Readable labels for the per-metric detail rows. Anything not listed falls back
# to a generic prettifier (underscores to spaces, known acronyms upper-cased).
_SCALAR_LABELS: dict[str, str] = {
    "score": "Score (0-1, higher is better)",
    "p_value": "p-value",
    "p05_dcr": "5th-percentile distance",
    "median_dcr": "Median distance",
    "median_dcr_holdout": "Median distance (holdout)",
    "p05_dcr_holdout": "5th-percentile distance (holdout)",
    "min_k": "Smallest group size (k)",
    "unique_groups": "Distinct groups",
    "identical_match_rate": "Exact-copy rate",
    "identical_matches": "Exact copies",
    "nndr_median": "Median distance ratio",
    "nndr_median_holdout": "Median distance ratio (holdout)",
    "c2st_auc": "Classifier AUC",
    "mean_mi_diff": "Mean MI difference",
    "mmd": "MMD",
    "alpha_precision": "Alpha-precision",
    "beta_recall": "Beta-recall",
    "nn_adversarial_accuracy": "Adversarial accuracy",
    "spearman_rho": "Spearman correlation",
    "mean_ci_overlap": "Mean CI overlap",
    "mean_hellinger": "Mean Hellinger distance",
    "mean_quantile_mse": "Mean quantile MSE",
    "mean_abs_std_diff": "Mean standardized mean gap",
    "epsilon": "Epsilon",
    "delta": "Delta",
    "hit_rate": "Hit rate",
    "identifiability_risk": "Identifiability risk",
    "disclosure_rate": "Disclosure rate",
    "spd_synthetic": "Parity gap (synthetic)",
    "spd_real": "Parity gap (real)",
    "parity_drift": "Parity drift",
    "utility_ratio_mean": "Mean utility ratio",
    "ratio_linear": "Utility ratio — linear model",
    "ratio_random_forest": "Utility ratio — random forest",
    "ratio_hist_gbdt": "Utility ratio — gradient boosting",
    "ratio_spread": "Spread across model families (std)",
    "n_rare_categories": "Rare categories found",
    "pct_rare_missing": "Share of rare categories missing",
    "retention": "Mean retention",
    "lower_tail_coverage": "Lower-tail coverage",
    "upper_tail_coverage": "Upper-tail coverage",
    "tail_coverage": "Tail coverage",
    "worst_density_ratio": "Worst density ratio",
    "density_ratio": "Density ratio",
    "worst_subgroup_score": "Worst subgroup score",
    "mean_share_drift": "Mean subgroup-share drift",
    "share_real": "Subgroup share (real)",
    "share_syn": "Subgroup share (synthetic)",
    "shape_score": "Within-subgroup shape score",
    "worst_excess_gap": "Worst excess gap",
    "mean_gap_synthetic": "Mean gap (synthetic-trained)",
    "mean_gap_real": "Mean gap (real-trained)",
    "gap_synthetic": "Gap (synthetic-trained)",
    "gap_real": "Gap (real-trained)",
    "excess_gap": "Excess gap",
    "mean_risk_ratio": "Mean risk ratio",
    "worst_risk_ratio": "Worst risk ratio",
    "risk_ratio": "Risk ratio",
    "median_dcr_subgroup": "Median distance (subgroup)",
    "median_dcr_overall": "Median distance (overall)",
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
