import pytest

from syneva.core.errors import RegistryError
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import MetricRegistry


class _FakeMetric:
    spec = MetricSpec(
        name="fake",
        c="congruence",
        tier="core",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
    )

    def compute(self, real, synthetic, meta):
        return MetricResult(spec=self.spec, scalars={"x": 1.0})


def test_register_and_select():
    r = MetricRegistry()
    r.register(_FakeMetric)
    out = r.select(tiers=["core"], data_type="static")
    assert _FakeMetric in out


def test_select_filters_by_c():
    r = MetricRegistry()
    r.register(_FakeMetric)
    out = r.select(tiers=["core"], data_type="static", cs=["coverage"])
    assert out == []


def test_unknown_tier_raises():
    r = MetricRegistry()
    with pytest.raises(RegistryError):
        r.select(tiers=["bogus"], data_type="static")


def test_unknown_data_type_raises():
    r = MetricRegistry()
    with pytest.raises(RegistryError, match="unknown data_type"):
        r.select(tiers=["core"], data_type="bogus")


def test_longitudinal_raises_v02_message():
    r = MetricRegistry()
    with pytest.raises(RegistryError, match=r"v0\.2"):
        r.select(tiers=["core"], data_type="longitudinal")


def test_double_register_is_idempotent():
    r = MetricRegistry()
    r.register(_FakeMetric)
    r.register(_FakeMetric)
    assert len(r.select(tiers=["core"], data_type="static")) == 1
