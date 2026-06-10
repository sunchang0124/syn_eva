# tests/fixtures/build_fixtures.py
"""Build deterministic toy fixtures.

Run from repo root: `uv run python tests/fixtures/build_fixtures.py`
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
SEED = 17
N = 500


def _make_real(rng: np.random.Generator) -> pd.DataFrame:
    age = rng.normal(40, 12, N).clip(18, 90).round().astype(int)
    sex = rng.choice(["Male", "Female"], size=N, p=[0.55, 0.45])
    edu = rng.choice(["HS", "BSc", "MSc", "PhD"], size=N, p=[0.4, 0.35, 0.2, 0.05])
    hours = rng.normal(40, 8, N).clip(1, 99).round().astype(int)
    income = (
        age * 600 + (edu == "PhD") * 25000 + (edu == "MSc") * 12000 + rng.normal(0, 8000, N)
    ).round(2)
    high_income = (income > 60_000).astype(int)
    return pd.DataFrame(
        {
            "age": age,
            "sex": sex,
            "education": edu,
            "hours_per_week": hours,
            "income": income,
            "high_income": high_income,
        }
    )


def _make_syn_good(real: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Mild distributional noise — should score high on Congruence."""
    df = real.copy()
    df["age"] = (df["age"] + rng.normal(0, 1.5, N)).clip(18, 90).round().astype(int)
    df["hours_per_week"] = (
        (df["hours_per_week"] + rng.normal(0, 1, N)).clip(1, 99).round().astype(int)
    )
    df["income"] = (df["income"] + rng.normal(0, 1500, N)).round(2)
    # categorical jitter: 5% flip
    flip = rng.random(N) < 0.05
    df.loc[flip, "education"] = rng.choice(["HS", "BSc", "MSc", "PhD"], size=flip.sum())
    df["high_income"] = (df["income"] > 60_000).astype(int)
    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)


def _make_syn_shifted(real: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Deliberate distribution shift — should score worse on Congruence/Coverage."""
    df = real.copy()
    df["age"] = (df["age"] + 15).clip(18, 90).round().astype(int)  # mean shift
    df["education"] = rng.choice(["BSc"], size=N)  # mode collapse
    df["income"] = (df["income"] * 0.7).round(2)
    df["high_income"] = (df["income"] > 60_000).astype(int)
    return df


def _make_syn_leaky(real: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """80 verbatim real rows + 20 noise rows → privacy must fire."""
    leaky = real.sample(n=80, random_state=SEED).copy()
    extras = real.sample(n=20, random_state=SEED + 1).copy()
    extras["income"] = (extras["income"] + rng.normal(0, 1000, len(extras))).round(2)
    return pd.concat([leaky, extras], ignore_index=True)


def main() -> None:
    rng = np.random.default_rng(SEED)
    real = _make_real(rng)
    syn_good = _make_syn_good(real, np.random.default_rng(SEED + 1))
    syn_shifted = _make_syn_shifted(real, np.random.default_rng(SEED + 2))
    syn_leaky = _make_syn_leaky(real, np.random.default_rng(SEED + 3))

    real.to_parquet(HERE / "adult_income_real_500.parquet", index=False)
    syn_good.to_parquet(HERE / "adult_income_syn_good_500.parquet", index=False)
    syn_shifted.to_parquet(HERE / "adult_income_syn_shifted_500.parquet", index=False)
    syn_leaky.to_parquet(HERE / "adult_income_syn_leaky_100.parquet", index=False)

    metadata = {
        "columns": {
            "age": {"dtype": "numeric"},
            "sex": {"dtype": "categorical", "sensitive": True},
            "education": {"dtype": "categorical"},
            "hours_per_week": {"dtype": "numeric"},
            "income": {"dtype": "numeric", "sensitive": True},
            "high_income": {"dtype": "categorical"},
        },
        "primary_key": None,
    }
    (HERE / "metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
