"""`evaluate` runs on frames whose column labels aren't strings (#35)."""

import numpy as np
import pandas as pd

import syneva


def _scored(real: pd.DataFrame, synthetic: pd.DataFrame) -> list[str]:
    report = syneva.evaluate(real, synthetic, tiers=("core", "extended"))
    assert [r.spec.name for r in report.results if r.error] == []
    return [r.spec.name for r in report.results if r.scalars]


def test_evaluate_scores_array_built_frames_like_named_ones():
    rng = np.random.default_rng(0)

    def frame() -> pd.DataFrame:
        # Built from an array, so the columns are labelled 0, 1, 2, 3.
        df = pd.DataFrame(rng.normal(size=(200, 3)))
        df[3] = rng.choice(["a", "b", "c"], 200)
        return df

    real, synthetic = frame(), frame()
    names = {0: "x", 1: "y", 2: "z", 3: "c"}
    positional = _scored(real, synthetic)
    # The same metrics score either way: none is skipped for the labels alone.
    assert positional == _scored(real.rename(columns=names), synthetic.rename(columns=names))
    assert positional
