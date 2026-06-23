from syneva import FairnessSpec
from syneva.core.metric_info import describe_c


def test_fairness_spec_fields():
    sp = FairnessSpec(protected_attribute="sex", outcome="y")
    assert sp.protected_attribute == "sex"
    assert sp.outcome == "y"
    assert sp.favorable_outcome is None


def test_fairness_dimension_described():
    assert describe_c("fairness") != ""
