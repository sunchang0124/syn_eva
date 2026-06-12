import json
from pathlib import Path

import syneva

GOLDEN = Path(__file__).parent.parent / "golden" / "adult_income_good.scorecard.json"


def _scrub(d: dict) -> dict:
    """Strip volatile fields before diffing."""
    d = json.loads(json.dumps(d))
    d["run_info"]["started_at"] = 0
    d["run_info"]["finished_at"] = 0
    d["run_info"]["library_versions"] = {k: "PINNED" for k in d["run_info"]["library_versions"]}
    return d


def test_adult_income_good_matches_golden(real_df, syn_good_df, metadata, update_golden):
    rep = syneva.evaluate(real_df, syn_good_df, metadata, tiers=("core",), run_utility=False)
    actual = _scrub(rep.to_dict())
    if update_golden or not GOLDEN.exists():
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(actual, indent=2, sort_keys=True))
        return
    expected = _scrub(json.loads(GOLDEN.read_text()))
    assert actual == expected, (
        "Golden mismatch. If intentional, run "
        "`uv run pytest tests/integration/test_golden.py --update-golden`"
    )
