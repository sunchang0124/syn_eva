from __future__ import annotations

from typing import ClassVar

from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry


@registry.register
class DPLedger:
    spec: ClassVar[MetricSpec] = MetricSpec(
        name="dp_ledger",
        c="compliance",
        tier="extended",
        data_types=frozenset({"static"}),
        requires_real=False,
        scope="table-level",
        scoring=False,  # reports a declared budget; it does not measure the data
    )

    def compute(self, real, synthetic, meta) -> MetricResult:
        ledger = synthetic.attrs.get("dp_ledger") if hasattr(synthetic, "attrs") else None
        if not ledger:
            return MetricResult(
                spec=self.spec,
                skip_reason=(
                    "The synthetic data declares no differential-privacy ledger "
                    "(synthetic.attrs['dp_ledger'])."
                ),
            )
        return MetricResult(
            spec=self.spec,
            scalars={
                "epsilon": float(ledger["epsilon"]),
                "delta": float(ledger.get("delta", 0.0)),
            },
        )
