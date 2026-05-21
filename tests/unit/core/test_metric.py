from syneva.core.metric import MetricResult, MetricSpec


def test_spec_is_frozen():
    s = MetricSpec(
        name="x",
        c="congruence",
        tier="core",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="per-column",
    )
    from dataclasses import FrozenInstanceError

    import pytest

    with pytest.raises(FrozenInstanceError):
        s.name = "y"  # type: ignore[misc]


def test_result_has_default_notes_list():
    spec = MetricSpec(
        name="x",
        c="congruence",
        tier="core",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="per-column",
    )
    r = MetricResult(spec=spec, scalars={"v": 0.5})
    assert r.notes == []
    assert r.error is None
    assert r.plot_payload is None
