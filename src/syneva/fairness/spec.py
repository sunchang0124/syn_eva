from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FairnessSpec:
    """Declares a fairness comparison: a protected attribute, an outcome column,
    and (optionally) which outcome value is the favorable one (defaults to the
    maximum outcome value, e.g. 1 / True)."""

    protected_attribute: str
    outcome: str
    favorable_outcome: Any = None
