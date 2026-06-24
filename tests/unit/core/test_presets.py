import dataclasses

import pytest

from syneva.core.errors import SynevaError
from syneva.core.presets import PRESETS, Preset, get_preset


def test_known_presets_exist():
    assert set(PRESETS) == {"fast", "full", "privacy"}


def test_fast_preset_values():
    p = get_preset("fast")
    assert isinstance(p, Preset)
    assert p.tiers == ("core",)
    assert p.cs is None
    assert p.run_utility is False
    assert p.run_fairness is False
    assert p.distance == "euclidean"


def test_full_preset_values():
    p = get_preset("full")
    assert p.tiers == ("core", "extended")
    assert p.cs is None
    assert p.run_utility is True
    assert p.run_fairness is True


def test_privacy_preset_values():
    p = get_preset("privacy")
    assert p.tiers == ("core", "extended")
    assert p.cs == ("compliance",)
    assert p.run_utility is False


def test_unknown_preset_raises():
    with pytest.raises(SynevaError, match="unknown preset"):
        get_preset("nope")


def test_preset_is_frozen():
    p = get_preset("fast")
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.tiers = ("extended",)  # frozen dataclass -> FrozenInstanceError
