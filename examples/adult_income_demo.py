"""End-to-end demo: evaluate adult_income synthetic vs real."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import syneva

HERE = Path(__file__).parent.parent / "tests" / "fixtures"


def main() -> None:
    real = pd.read_parquet(HERE / "adult_income_real_500.parquet")
    syn = pd.read_parquet(HERE / "adult_income_syn_good_500.parquet")
    rep = syneva.evaluate(
        real,
        syn,
        utility_tasks=[syneva.UtilityTask(target="high_income", task_type="classification")],
        run_utility=True,
    )
    out = Path("syneva-report-demo")
    out.mkdir(exist_ok=True)
    rep.to_html(out / "scorecard.html")
    rep.to_json(out / "scorecard.json")
    print("aggregated:", rep.aggregated)
    print(f"wrote {out}/scorecard.html")


if __name__ == "__main__":
    main()
