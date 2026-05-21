# tests/unit/core/test_errors.py
from syneva.core.errors import (
    MetadataError,
    MetricError,
    RegistryError,
    SchemaError,
    SynevaError,
)


def test_hierarchy():
    assert issubclass(SchemaError, SynevaError)
    assert issubclass(MetadataError, SynevaError)
    assert issubclass(RegistryError, SynevaError)
    assert issubclass(MetricError, SynevaError)


def test_schema_error_carries_diff():
    err = SchemaError("columns mismatch", extra=["c"], missing=["d"])
    assert err.extra == ["c"]
    assert err.missing == ["d"]
    assert "columns mismatch" in str(err)


def test_metric_error_wraps_original():
    inner = ValueError("nope")
    err = MetricError("ks failed", original=inner)
    assert err.original is inner
