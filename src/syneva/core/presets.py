from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from syneva.core.errors import SynevaError

# Sentinel marking "argument not supplied" so preset/explicit resolution can tell
# an explicit value from an omitted one (None is a legitimate value for `cs`).
_UNSET = object()


@dataclass(frozen=True)
class Preset:
    tiers: tuple[str, ...]
    cs: tuple[str, ...] | None  # None = all Cs
    run_utility: bool
    run_fairness: bool
    distance: str
    # Per-metric constructor options; explicit evaluate() options override them.
    # Left out of the hash: a dict can't be hashed, and the other fields identify
    # the preset.
    metric_options: Mapping[str, Mapping[str, Any]] = field(default_factory=dict, hash=False)


PRESETS: dict[str, Preset] = {
    "fast": Preset(
        tiers=("core",),
        cs=None,
        run_utility=False,
        run_fairness=False,
        distance="euclidean",
    ),
    "full": Preset(
        tiers=("core", "extended"),
        cs=None,
        run_utility=True,
        run_fairness=True,
        distance="euclidean",
    ),
    "privacy": Preset(
        tiers=("core", "extended"),
        cs=("compliance",),
        run_utility=False,
        run_fairness=False,
        distance="euclidean",
    ),
}


def get_preset(name: str) -> Preset:
    if name not in PRESETS:
        raise SynevaError(f"unknown preset '{name}'; valid: {sorted(PRESETS)}")
    return PRESETS[name]
