# Quickstart

## Minimal example

```python
import pandas as pd, syneva

real = pd.read_parquet("real.parquet")
syn  = pd.read_parquet("synthetic.parquet")
rep  = syneva.evaluate(real, syn)
rep.to_html("scorecard.html")
```

## Declaring privacy roles (for k-anonymity and attribute disclosure)

Columns can take two independent privacy roles (Sweeney, 2002; Machanavajjhala
et al., 2007):

- **`quasi_identifier`**: an attribute an attacker could link on, such as age,
  sex or zip code. `k_anonymity` groups the synthetic rows on these columns and
  is skipped when none are marked.
- **`sensitive`**: an attribute whose disclosure is the harm, such as a
  diagnosis or income. `attribute_disclosure` tries to infer these columns from
  the quasi-identifiers, and never links on a sensitive column. If no
  non-sensitive column is a quasi-identifier, it infers from every other numeric
  or categorical column instead.

```python
from syneva import Metadata, ColumnMetadata, ColumnType

meta = Metadata.infer(real).override(
    age={"quasi_identifier": True},
    sex={"quasi_identifier": True},
    income={"sensitive": True},
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
