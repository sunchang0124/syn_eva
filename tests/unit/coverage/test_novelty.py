# tests/unit/coverage/test_novelty.py
import pandas as pd
import pytest

from syneva.compliance.identical_match import IdenticalMatchRate
from syneva.core.metadata import Metadata
from syneva.coverage.novelty import NoveltyRate


def test_all_novel_score_one():
    real = pd.DataFrame({"x": list(range(100))})
    syn = pd.DataFrame({"x": list(range(100, 200))})
    r = NoveltyRate().compute(real, syn, Metadata.infer(real))
    assert r.scalars["novelty_rate"] == 1.0


def test_all_duplicates_score_zero():
    real = pd.DataFrame({"x": list(range(100))})
    syn = real.copy()
    r = NoveltyRate().compute(real, syn, Metadata.infer(real))
    assert r.scalars["novelty_rate"] == 0.0


def test_id_columns_ignored():
    real = pd.DataFrame({"id": [1, 2, 3], "x": [1.0, 2.0, 3.0]})
    syn = real.assign(id=[101, 102, 103])
    r = NoveltyRate().compute(real, syn, Metadata.infer(real))
    assert r.scalars["novelty_rate"] == 0.0


def test_novelty_is_complement_of_identical_match_rate():
    # 2 synthetic rows copy a real row; 3 are new but identical to each other.
    # Novelty counts rows not copied from real (3/5), not distinct synthetic rows (2/5).
    real = pd.DataFrame({"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"]})
    syn = pd.DataFrame({"a": [1, 1, 9, 9, 9], "b": ["w", "w", "q", "q", "q"]})
    meta = Metadata.infer(real)
    nov = NoveltyRate().compute(real, syn, meta).scalars
    imr = IdenticalMatchRate().compute(real, syn, meta).scalars
    assert nov is not None and imr is not None
    assert nov["novelty_rate"] == pytest.approx(0.6)
    assert nov["novelty_rate"] == pytest.approx(1 - imr["identical_match_rate"])
    assert nov["duplicates"] == imr["identical_matches"] == 2.0
