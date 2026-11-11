"""`evaluate` runs on frames whose column labels aren't strings (#35)."""

import numpy as np
import pandas as pd
import pytest

import syneva
from syneva.utility.task import UtilityTask


def _scored(real: pd.DataFrame, synthetic: pd.DataFrame, target) -> list[str]:
    report = syneva.evaluate(
        real,
        synthetic,
        tiers=("core", "extended"),
        run_utility=True,
        utility_tasks=[UtilityTask(target=target, task_type="classification")],
    )
    assert [(r.spec.name, str(r.error)) for r in report.results if r.error] == []
    return [r.spec.name for r in report.results if r.scalars]


@pytest.mark.parametrize(
    "labels",
    [
        pytest.param([0, 1, 2, 3], id="integers"),  # pd.DataFrame(ndarray), header=None
        pytest.param([0, "b", 2, "d"], id="mixed"),
    ],
)
def test_evaluate_scores_unnamed_columns_like_named_ones(labels):
    rng = np.random.default_rng(0)

    def frame() -> pd.DataFrame:
        x = rng.normal(size=(200, 3))
        # The target comes first, so a label is not its column's position among the
        # features: a model that reads integer labels as positions picks the wrong ones.
        target = np.where(x[:, 0] + rng.normal(0, 0.5, 200) > 0, "yes", "no")
        return pd.DataFrame(dict(zip(labels, [target, *x.T], strict=True)))

    real, synthetic = frame(), frame()
    names = dict(zip(labels, ["y", "x0", "x1", "x2"], strict=True))
    unnamed = _scored(real, synthetic, labels[0])
    named = _scored(real.rename(columns=names), synthetic.rename(columns=names), "y")
    # The same metrics score either way: none is skipped for the labels alone.
    assert unnamed == named
    assert "tstr_suite" in unnamed
