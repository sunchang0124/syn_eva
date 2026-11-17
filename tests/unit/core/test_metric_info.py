import syneva  # noqa: F401  populate the global registry
from syneva.core.metric_info import (
    describe_c,
    describe_metric,
    display_name,
    humanize_scalar,
    raw_hint,
    verdict,
)
from syneva.core.registry import registry


def test_every_registered_metric_has_a_description():
    missing = [c.spec.name for c in registry.metrics() if not describe_metric(c.spec.name)]
    assert not missing, f"metrics missing a plain-language description: {missing}"


def test_every_registered_metric_has_a_raw_hint():
    missing = [c.spec.name for c in registry.metrics() if not raw_hint(c.spec.name)]
    assert not missing, f"metrics missing a raw-statistic hint: {missing}"


def test_base_description_makes_no_direction_claim():
    # The neutral description must not bake in "higher/lower is better"; that
    # belongs to the score banner (normalized) or the raw hint (actual).
    for c in registry.metrics():
        text = describe_metric(c.spec.name).lower()
        assert "higher is better" not in text
        assert "lower is better" not in text


def test_every_registered_metric_has_a_full_name():
    # display_name must differ from the bare code for every registered metric.
    missing = [c.spec.name for c in registry.metrics() if display_name(c.spec.name) == c.spec.name]
    assert not missing, f"metrics missing a full display name: {missing}"


def test_verdict_thresholds():
    assert verdict(0.98)[0] == "Excellent"
    assert verdict(0.80)[0] == "Good"
    assert verdict(0.60)[0] == "Fair"
    assert verdict(0.30)[0] == "Poor"
    # the second element is a CSS class used by the scorecard template
    assert verdict(0.98)[1] == "v-excellent"


def test_humanize_scalar_examples():
    assert humanize_scalar("mean_ks_statistic") == "Mean KS statistic"
    assert humanize_scalar("p05_dcr") == "5th-percentile distance"
    assert humanize_scalar("score") == "Score (0-1, higher is better)"
    assert humanize_scalar("c2st_auc") == "Classifier AUC"


def test_every_active_c_has_a_description():
    cs = {c.spec.c for c in registry.metrics()}
    missing = [c for c in cs if not describe_c(c)]
    assert not missing, f"C dimensions missing a description: {missing}"


def test_describe_unknown_returns_empty():
    assert describe_metric("does_not_exist") == ""
    assert describe_c("does_not_exist") == ""


def test_every_option_row_names_a_constructor_option_with_a_scalar_default():
    import inspect

    from syneva.core.metric_info import OPTION_INFO

    by_name = {c.spec.name: c for c in registry.metrics()}
    for metric, options in OPTION_INFO.items():
        assert metric in by_name, f"OPTION_INFO names unknown metric '{metric}'"
        params = inspect.signature(by_name[metric]).parameters
        for option, info in options.items():
            assert option in params, f"'{metric}' has no constructor option '{option}'"
            default = params[option].default
            assert type(default) in (bool, int, float), f"{metric}.{option} default {default!r}"
            assert info.label and info.help
