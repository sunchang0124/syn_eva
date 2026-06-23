"""CI-enforced SynthEval coverage manifest.

Maps each SynthEval metric code to its syneva status. Flip "planned" -> "implemented"
as later phases (0b/0c) land. The test asserts every "implemented" entry resolves to a
registered syneva metric, making "superset of SynthEval" a CI invariant.
"""

import syneva  # noqa: F401  populate the global registry
from syneva.core.registry import registry

# code -> (status, syneva_metric_name_or_phase)
SYNTHEVAL_PARITY: dict[str, tuple[str, str]] = {
    "corr_diff": ("implemented", "correlation_difference"),
    "ks_test": ("implemented", "ks_statistic"),
    "p_MSE": ("implemented", "pmse"),
    "fio": ("implemented", "feature_importance_spearman"),
    "cls_acc": ("implemented", "tstr_suite"),
    "auroc_diff": ("implemented", "tstr_suite"),
    "pca": ("implemented", "pca_scatter"),
    "dwm": ("implemented", "dimension_wise_means"),
    "cio": ("implemented", "ci_overlap"),
    "h_dist": ("implemented", "hellinger"),
    "q_mse": ("implemented", "quantile_mse"),
    "mi_diff": ("implemented", "mutual_information_difference"),
    "mmd": ("implemented", "mmd"),
    "nnaa": ("implemented", "nn_adversarial_accuracy"),
    "nndr": ("implemented", "nndr"),
    "dcr": ("implemented", "dcr"),
    "mia": ("implemented", "mia_auc"),
    "hit_rate": ("planned", "0b"),
    "eps_risk": ("planned", "0b"),
    "att_discl": ("planned", "0b"),
    "statistical_parity": ("planned", "0b"),
}


def test_implemented_parity_metrics_are_registered():
    registered = {cls.spec.name for cls in registry.metrics()}
    missing = [
        f"{code} -> {name}"
        for code, (status, name) in SYNTHEVAL_PARITY.items()
        if status == "implemented" and name not in registered
    ]
    assert not missing, f"SynthEval-parity metrics claimed but not registered: {missing}"


def test_no_unknown_status():
    for code, (status, _) in SYNTHEVAL_PARITY.items():
        assert status in {"implemented", "planned", "na"}, f"{code} has bad status {status}"
