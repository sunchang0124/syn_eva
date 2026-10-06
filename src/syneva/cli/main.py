from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import typer

import syneva
from syneva.core.errors import SynevaError
from syneva.core.metadata import ColumnMetadata, ColumnType, Metadata
from syneva.core.presets import _UNSET

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
    holdout: Path | None = typer.Option(None, exists=True, dir_okay=False),
    tiers: str | None = typer.Option(None, help="comma-separated tiers"),
    cs: str | None = typer.Option(None, help="comma-separated Cs to restrict to"),
    run_utility: bool = typer.Option(False, help="run utility (TSTR) tasks"),
    preset: str | None = typer.Option(None, help="evaluation preset: fast | full | privacy"),
    pdf: bool = typer.Option(False, help="also write scorecard.pdf"),
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    try:
        real_df = _load_df(real) if real else None
        syn_df = _load_df(synthetic)
        meta = _load_metadata(metadata)
        holdout_df = _load_df(holdout) if holdout else None
        rep = syneva.evaluate(
            real_df,
            syn_df,
            meta,
            holdout=holdout_df,
            tiers=(tuple(t.strip() for t in tiers.split(",")) if tiers else _UNSET),
            cs=(tuple(c.strip() for c in cs.split(",")) if cs else _UNSET),
            run_utility=(True if run_utility else _UNSET),
            preset=preset,
        )
        rep.to_json(out / "scorecard.json")
        rep.to_html(out / "scorecard.html")
        if pdf:
            rep.to_pdf(out / "scorecard.pdf")
        typer.echo(f"wrote scorecard to {out}/")
    except SynevaError as e:
        typer.echo(f"syneva error: {e}", err=True)
        raise typer.Exit(code=2) from e


@app.command()
def benchmark(
    candidate: list[str] = typer.Option(
        ..., "--candidate", help="named synthetic dataset as name=path (repeatable)"
    ),
    real: Path | None = typer.Option(None, exists=True, dir_okay=False),
    metadata: Path | None = typer.Option(None, exists=True, dir_okay=False),
    out: Path = typer.Option(Path("./syneva-benchmark")),
    holdout: Path | None = typer.Option(None, exists=True, dir_okay=False),
    preset: str | None = typer.Option(None, help="evaluation preset: fast | full | privacy"),
    tiers: str | None = typer.Option(None, help="comma-separated tiers"),
    cs: str | None = typer.Option(None, help="comma-separated Cs to restrict to"),
    run_utility: bool = typer.Option(False, help="run utility (TSTR) tasks"),
    normalization: str = typer.Option("absolute", help="ranking: absolute|linear|normal|quantile"),
) -> None:
    candidates: dict[str, pd.DataFrame] = {}
    for entry in candidate:
        if "=" not in entry:
            raise typer.BadParameter(f"candidate must be name=path, got '{entry}'")
        name, _, path_str = entry.partition("=")
        if not name:
            raise typer.BadParameter(f"candidate name is empty in '{entry}'")
        if name in candidates:
            raise typer.BadParameter(f"duplicate candidate name '{name}'")
        cand_path = Path(path_str)
        if not cand_path.exists():
            raise typer.BadParameter(f"candidate '{name}' path does not exist: {path_str}")
        candidates[name] = _load_df(cand_path)
    out.mkdir(parents=True, exist_ok=True)
    try:
        real_df = _load_df(real) if real else None
        meta = _load_metadata(metadata)
        holdout_df = _load_df(holdout) if holdout else None
        res = syneva.benchmark(
            real_df,
            candidates,
            meta,
            holdout=holdout_df,
            preset=preset,
            tiers=(tuple(t.strip() for t in tiers.split(",")) if tiers else _UNSET),
            cs=(tuple(c.strip() for c in cs.split(",")) if cs else _UNSET),
            run_utility=(True if run_utility else _UNSET),
        )
        ranked = res.ranking(normalization=normalization)  # fail fast on bad normalization
        res.to_html(out / "leaderboard.html", normalization=normalization)
        res.to_json(out / "benchmark.json")
        for rank, name, score in ranked:
            typer.echo(f"{rank}. {name}  {score:.4f}")
        typer.echo(f"wrote leaderboard to {out}/")
    except SynevaError as e:
        typer.echo(f"syneva error: {e}", err=True)
        raise typer.Exit(code=2) from e


@app.command()
def ui() -> None:
    """Launch the Streamlit UI."""
    try:
        import streamlit  # noqa: F401
    except ImportError:
        typer.echo("The UI needs Streamlit: pip install 'syneva[ui]'", err=True)
        sys.exit(2)
    app_path = Path(__file__).parent.parent / "ui" / "app.py"
    proc = subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])
    sys.exit(proc.returncode)


if __name__ == "__main__":
    app()
