from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, cast

import pandas as pd

if TYPE_CHECKING:
    from syneva.fairness.spec import FairnessSpec
    from syneva.preservation.spec import SubgroupSpec
    from syneva.utility.task import UtilityTask

from syneva.core.errors import MetricError, SchemaError, SynevaError
from syneva.core.metadata import Metadata
from syneva.core.metric import MetricResult
from syneva.core.presets import _UNSET, get_preset
from syneva.core.registry import MetricRegistry
from syneva.core.registry import registry as _default_registry
from syneva.core.report import Report
from syneva.core.run_info import RunInfo


def _instantiate(
    cls, utility_tasks, fairness_specs, subgroup_specs, distance, holdout, random_state
):
    """Instantiate a metric, passing only the kwargs its constructor accepts."""
    params = inspect.signature(cls).parameters
    kwargs = {}
    if "tasks" in params:
        kwargs["tasks"] = utility_tasks
    if "specs" in params:
        kwargs["specs"] = fairness_specs if fairness_specs is not None else []
    if "subgroup_specs" in params:
        kwargs["subgroup_specs"] = subgroup_specs if subgroup_specs is not None else []
    if "distance" in params:
        kwargs["distance"] = distance
    if "holdout" in params:
        kwargs["holdout"] = holdout
    if "random_state" in params:
        kwargs["random_state"] = random_state
    return cls(**kwargs)


def evaluate(
    real: pd.DataFrame | None,
    synthetic: pd.DataFrame,
    metadata: Metadata | None = None,
    *,
    tiers: Sequence[str] | object = _UNSET,
    data_type: str = "static",
    cs: Sequence[str] | None | object = _UNSET,
    utility_tasks: list[UtilityTask] | None = None,
    run_utility: bool | object = _UNSET,
    fairness_specs: list[FairnessSpec] | None = None,
    run_fairness: bool | object = _UNSET,
    subgroup_specs: list[SubgroupSpec] | None = None,
    distance: str | object = _UNSET,
    holdout: pd.DataFrame | None = None,
    preset: str | None = None,
    random_state: int = 42,
    nan_policy: Literal["drop", "explicit_na", "raise"] = "drop",
) -> Report:
    """Evaluate synthetic tabular data against real data, return a Report."""
    return evaluate_with(
        _default_registry,
        real=real,
        synthetic=synthetic,
        metadata=metadata,
        tiers=tiers,
        data_type=data_type,
        cs=cs,
        utility_tasks=utility_tasks,
        run_utility=run_utility,
        fairness_specs=fairness_specs,
        run_fairness=run_fairness,
        subgroup_specs=subgroup_specs,
        distance=distance,
        holdout=holdout,
        preset=preset,
        random_state=random_state,
        nan_policy=nan_policy,
    )


def evaluate_with(
    reg: MetricRegistry,
    *,
    real: pd.DataFrame | None,
    synthetic: pd.DataFrame,
    metadata: Metadata | None = None,
    tiers: Sequence[str] | object = _UNSET,
    data_type: str = "static",
    cs: Sequence[str] | None | object = _UNSET,
    utility_tasks: list[UtilityTask] | None = None,
    run_utility: bool | object = _UNSET,
    fairness_specs: list[FairnessSpec] | None = None,
    run_fairness: bool | object = _UNSET,
    subgroup_specs: list[SubgroupSpec] | None = None,
    distance: str | object = _UNSET,
    holdout: pd.DataFrame | None = None,
    preset: str | None = None,
    random_state: int = 42,
    nan_policy: str = "drop",
    # nan_policy is reserved for v0.1.x; metrics handle NaNs per their own policy for now
) -> Report:
    from syneva.utility.task import suggest_tasks

    preset_obj = get_preset(preset) if preset is not None else None

    def _resolve(value, attr, hard_default):
        if value is not _UNSET:
            return value
        if preset_obj is not None:
            return getattr(preset_obj, attr)
        return hard_default

    tiers = cast("Sequence[str]", _resolve(tiers, "tiers", ("core",)))
    cs = cast("Sequence[str] | None", _resolve(cs, "cs", None))
    run_utility = _resolve(run_utility, "run_utility", False)
    run_fairness = _resolve(run_fairness, "run_fairness", False)
    distance = _resolve(distance, "distance", "euclidean")

    if distance not in ("euclidean", "gower"):
        raise SynevaError(f"unknown distance '{distance}'; use 'euclidean' or 'gower'")

    if real is not None:
        _check_schema(real, synthetic)

    if holdout is not None and set(holdout.columns) != set(synthetic.columns):
        raise SynevaError("holdout columns must match the synthetic/real columns")

    if metadata is None:
        metadata = Metadata.infer(real if real is not None else synthetic)
    metadata.validate_against(synthetic)
    if real is not None:
        metadata.validate_against(real)

    info = RunInfo.capture(random_state=random_state)
    selected = reg.select(tiers=list(tiers), data_type=data_type, cs=cs)
    if real is None:
        selected = [c for c in selected if not c.spec.requires_real]

    if not run_utility:
        selected = [c for c in selected if c.spec.c != "utility"]
    elif utility_tasks is None:
        utility_tasks = suggest_tasks(metadata)
        info.warnings.append(
            "no utility_tasks declared; auto-suggested "
            + ", ".join(f"{t.target}({t.task_type})" for t in utility_tasks)
        )

    if not run_fairness:
        selected = [c for c in selected if c.spec.c != "fairness"]

    if not subgroup_specs:
        # preservation metrics that need subgroups are dropped, not skipped,
        # so spec-less runs stay noise-free
        selected = [c for c in selected if "subgroup_specs" not in inspect.signature(c).parameters]

    if not selected:
        raise SynevaError(
            "no runnable metrics for this selection "
            f"(real={'None' if real is None else 'df'}, tiers={list(tiers)}, "
            f"data_type={data_type}, cs={list(cs) if cs is not None else 'all'})"
        )

    results: list[MetricResult] = []
    for cls in selected:
        try:
            inst = _instantiate(
                cls, utility_tasks, fairness_specs, subgroup_specs, distance, holdout, random_state
            )
            results.append(inst.compute(real, synthetic, metadata))
        except Exception as e:
            results.append(
                MetricResult(
                    spec=cls.spec,
                    error=MetricError(
                        f"{cls.spec.name} failed: {e}",
                        original=e,
                    ),
                )
            )

    info.finish()
    return Report(metadata=metadata, results=results, run_info=info)


def _check_schema(real: pd.DataFrame, synthetic: pd.DataFrame) -> None:
    real_cols, syn_cols = list(real.columns), list(synthetic.columns)
    extra = [c for c in syn_cols if c not in real_cols]
    missing = [c for c in real_cols if c not in syn_cols]
    if extra or missing:
        raise SchemaError(
            f"column mismatch: extra={extra}, missing={missing}",
            extra=extra,
            missing=missing,
        )
