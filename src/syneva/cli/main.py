from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer

import syneva
from syneva.core.errors import SynevaError
from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata

app = typer.Typer(add_completion=False, help="syneva - 7 Cs scorecard for tabular synthetic data")


def _load_df(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise typer.BadParameter(f"unsupported extension: {path.suffix}")


def _load_metadata(path: Path | None) -> Metadata | None:
    if path is None:
        return None
    raw = json.loads(path.read_text())
    cols = {
        n: ColumnMetadata(
            name=n,
            dtype=ColumnType(c["dtype"]),
            sensitive=c.get("sensitive", False),
        )
        for n, c in raw["columns"].items()
    }
    return Metadata(columns=cols, primary_key=raw.get("primary_key"))


@app.callback()
def _main() -> None:
    """syneva command-line interface."""


@app.command()
def evaluate(
    real: Path | None = typer.Option(None, exists=True, dir_okay=False),
    synthetic: Path = typer.Option(..., exists=True, dir_okay=False),
    metadata: Path | None = typer.Option(None, exists=True, dir_okay=False),
    out: Path = typer.Option(Path("./syneva-report")),
    tiers: str = typer.Option("core", help="comma-separated tiers"),
    cs: str | None = typer.Option(None, help="comma-separated Cs to restrict to"),
    run_utility: bool = typer.Option(False, help="run utility (TSTR) tasks"),
    pdf: bool = typer.Option(False, help="also write scorecard.pdf"),
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    try:
        real_df = _load_df(real) if real else None
        syn_df = _load_df(synthetic)
        meta = _load_metadata(metadata)
        rep = syneva.evaluate(
            real_df,
            syn_df,
            meta,
            tiers=tuple(t.strip() for t in tiers.split(",")),
            cs=tuple(c.strip() for c in cs.split(",")) if cs else None,
            run_utility=run_utility,
        )
        rep.to_json(out / "scorecard.json")
        rep.to_html(out / "scorecard.html")
        if pdf:
            rep.to_pdf(out / "scorecard.pdf")
        typer.echo(f"wrote scorecard to {out}/")
    except SynevaError as e:
        typer.echo(f"syneva error: {e}", err=True)
        raise typer.Exit(code=2) from e


if __name__ == "__main__":
    app()
