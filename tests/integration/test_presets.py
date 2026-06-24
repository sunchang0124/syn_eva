import pytest

import syneva
from syneva.core.errors import SynevaError


def _names(rep):
    return {r.spec.name for r in rep.results}


def test_fast_preset_core_only_no_utility(real_df, syn_good_df, metadata):
    rep = syneva.evaluate(real_df, syn_good_df, metadata, preset="fast")
    names = _names(rep)
    assert "tstr_suite" not in names
    assert all(r.spec.tier == "core" for r in rep.results)


def test_full_preset_includes_extended_and_utility(real_df, syn_good_df, metadata):
    rep = syneva.evaluate(real_df, syn_good_df, metadata, preset="full")
    tiers = {r.spec.tier for r in rep.results}
    assert "extended" in tiers
    assert "tstr_suite" in _names(rep)


def test_privacy_preset_compliance_only(real_df, syn_good_df, metadata):
    rep = syneva.evaluate(real_df, syn_good_df, metadata, preset="privacy")
    assert {r.spec.c for r in rep.results} == {"compliance"}


def test_explicit_overrides_preset(real_df, syn_good_df, metadata):
    from syneva.utility.task import UtilityTask

    rep = syneva.evaluate(
        real_df,
        syn_good_df,
        metadata,
        preset="fast",
        run_utility=True,
        utility_tasks=[UtilityTask(target="high_income", task_type="classification")],
    )
    assert "tstr_suite" in _names(rep)


def test_no_preset_matches_default_selection(real_df, syn_good_df, metadata):
    a = syneva.evaluate(real_df, syn_good_df, metadata)
    b = syneva.evaluate(real_df, syn_good_df, metadata, preset=None)
    assert _names(a) == _names(b)


def test_unknown_preset_raises(real_df, syn_good_df, metadata):
    with pytest.raises(SynevaError, match="unknown preset"):
        syneva.evaluate(real_df, syn_good_df, metadata, preset="nope")
