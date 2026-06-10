from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from syneva.core.errors import RegistryError

if TYPE_CHECKING:
    from syneva.core.metric import Metric


_VALID_TIERS = {"core", "extended", "custom"}
_VALID_DATA_TYPES = {"static", "longitudinal", "relational"}


class MetricRegistry:
    def __init__(self) -> None:
        self._metrics: dict[str, type[Metric]] = {}

    def register(self, cls: type[Metric]) -> type[Metric]:
        self._metrics[cls.spec.name] = cls
        return cls

    def select(
        self,
        *,
        tiers: Iterable[str],
        data_type: str,
        cs: Iterable[str] | None = None,
    ) -> list[type[Metric]]:
        tiers_set = set(tiers)
        if not tiers_set.issubset(_VALID_TIERS):
            raise RegistryError(
                f"unknown tier(s): {tiers_set - _VALID_TIERS}; valid: {_VALID_TIERS}"
            )
        if data_type not in _VALID_DATA_TYPES:
            raise RegistryError(f"unknown data_type '{data_type}'; valid: {_VALID_DATA_TYPES}")
        if data_type == "longitudinal":
            raise RegistryError("longitudinal not yet implemented; coming in v0.2")
        if data_type == "relational":
            raise RegistryError("relational not yet implemented; coming in v0.3")
        cs_set = set(cs) if cs is not None else None
        out: list[type[Metric]] = []
        for cls in self._metrics.values():
            s = cls.spec
            if s.tier not in tiers_set:
                continue
            if data_type not in s.data_types:
                continue
            if cs_set is not None and s.c not in cs_set:
                continue
            out.append(cls)
        return out


registry = MetricRegistry()
