import syneva  # noqa: F401  populate the global registry
from syneva.core.metric_info import describe_c, describe_metric
from syneva.core.registry import registry


def test_every_registered_metric_has_a_description():
    missing = [c.spec.name for c in registry.metrics() if not describe_metric(c.spec.name)]
    assert not missing, f"metrics missing a plain-language description: {missing}"


def test_every_active_c_has_a_description():
    cs = {c.spec.c for c in registry.metrics()}
    missing = [c for c in cs if not describe_c(c)]
    assert not missing, f"C dimensions missing a description: {missing}"


def test_describe_unknown_returns_empty():
    assert describe_metric("does_not_exist") == ""
    assert describe_c("does_not_exist") == ""
