from __future__ import annotations


class SynevaError(Exception):
    """Base class for all syneva errors."""


class SchemaError(SynevaError):
    def __init__(
        self, msg: str, *, extra: list[str] | None = None, missing: list[str] | None = None
    ) -> None:
        super().__init__(msg)
        self.extra: list[str] = extra or []
        self.missing: list[str] = missing or []


class MetadataError(SynevaError):
    pass


class RegistryError(SynevaError):
    pass


class MetricError(SynevaError):
    def __init__(self, msg: str, *, original: BaseException | None = None) -> None:
        super().__init__(msg)
        self.original = original
