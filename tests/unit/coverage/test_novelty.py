# tests/unit/coverage/test_novelty.py
import pandas as pd

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
