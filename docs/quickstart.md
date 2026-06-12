# Quickstart

## Minimal example

```python
import pandas as pd, syneva

real = pd.read_parquet("real.parquet")
syn  = pd.read_parquet("synthetic.parquet")
rep  = syneva.evaluate(real, syn)
rep.to_html("scorecard.html")
```

## Declaring sensitive columns (for k-anonymity and DCR)

```python
from syneva import Metadata, ColumnMetadata, ColumnType

meta = Metadata.infer(real).override(
    income={"sensitive": True}, sex={"sensitive": True},
)
rep = syneva.evaluate(real, syn, meta)
```

## Running utility tasks

```python
from syneva import UtilityTask
rep = syneva.evaluate(
    real, syn,
    utility_tasks=[UtilityTask(target="high_income", task_type="classification")],
    run_utility=True,
)
print(rep.aggregated["utility"])
```

## Registering a custom metric

```python
from syneva import registry, MetricSpec, MetricResult

@registry.register
class MyMetric:
    spec = MetricSpec(
        name="my_metric", c="congruence", tier="custom",
        data_types=frozenset({"static"}),
        requires_real=True, scope="table-level",
    )
    def compute(self, real, synthetic, meta) -> MetricResult:
        return MetricResult(spec=self.spec, scalars={"score": 0.5})

rep = syneva.evaluate(real, syn, tiers=("core", "custom"))
```
