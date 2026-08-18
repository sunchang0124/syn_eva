from syneva._version import __version__
from syneva.benchmark import BenchmarkResult, benchmark
from syneva.core.errors import (
    MetadataError,
    MetricError,
    RegistryError,
    SchemaError,
    SynevaError,
)
from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata
from syneva.core.metric import Metric, MetricResult, MetricSpec
from syneva.core.registry import MetricRegistry, registry
from syneva.core.report import Report
from syneva.core.run_info import RunInfo
from syneva.core.runner import evaluate
from syneva.fairness.spec import FairnessSpec
from syneva.preservation.spec import SubgroupSpec
from syneva.utility.task import UtilityTask

__all__ = [
    "BenchmarkResult",
    "ColumnMetadata",
    "ColumnType",
    "FairnessSpec",
    "Metadata",
    "MetadataError",
    "Metric",
    "MetricError",
    "MetricRegistry",
    "MetricResult",
    "MetricSpec",
    "RegistryError",
    "Report",
    "RunInfo",
    "SchemaError",
    "SubgroupSpec",
    "SynevaError",
    "UtilityTask",
    "__version__",
    "benchmark",
    "evaluate",
    "registry",
]

# Register the built-in metrics as a side effect of importing the package, so
# `import syneva; syneva.evaluate(...)` sees the full core scorecard.
from syneva import compliance as _compliance  # noqa: F401
from syneva import congruence as _congruence  # noqa: F401
from syneva import coverage as _coverage  # noqa: F401
from syneva import fairness as _fairness  # noqa: F401
from syneva import preservation as _preservation  # noqa: F401
from syneva import utility as _utility  # noqa: F401
