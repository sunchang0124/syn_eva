from __future__ import annotations

from syneva.core.errors import SynevaError

_MODES = ("absolute", "linear", "normal", "quantile")


def _average_ranks(values: list[float]) -> list[float]:
    """0-based ranks, ties averaged (e.g. two tied lowest -> 0.5, 0.5)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = sum(range(i, j + 1)) / (j - i + 1)
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def normalize_across(values: list[float], mode: str) -> list[float]:
    """Transform one metric's absolute scores across the candidate cohort.

    Used for ranking order ONLY; displayed scores stay absolute. `values` are in
    candidate order; the return list is in the same order.
    """
    n = len(values)
    if mode == "absolute":
        return list(values)
    if mode == "linear":
        lo, hi = min(values), max(values)
        if hi == lo:
            return [1.0] * n
        span = hi - lo
        return [round((v - lo) / span, 14) for v in values]
    if mode == "quantile":
        if n < 2:
            return [0.5] * n
        ranks = _average_ranks(values)
        return [r / (n - 1) for r in ranks]
    if mode == "normal":
        mean = sum(values) / n
        std = (sum((v - mean) ** 2 for v in values) / n) ** 0.5
        if std < 1e-14:
            return [0.0] * n
        return [(v - mean) / std for v in values]
    raise SynevaError(f"unknown normalization '{mode}'; use one of {_MODES}")
