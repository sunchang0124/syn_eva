"""Every random component is seeded from `random_state`, never a literal (#42).

`evaluate(random_state=...)` reaches each metric through its `random_state`
parameter, so a seed written into a call (or a generator with no seed at all)
silently ignores the run's seed and makes `RunInfo.random_state` wrong. Defaults
such as `random_state: int = 42` are parameters, not calls, and are allowed.
"""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "syneva"

_SEED_KEYWORDS = {"random_state", "seed"}
_SEEDED_CONSTRUCTORS = {"default_rng", "RandomState", "seed"}
# Estimators and splitters that draw random numbers unless given `random_state`. A
# call to one without that keyword falls back to an unseeded generator.
_NEEDS_RANDOM_STATE = {
    "LogisticRegression",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    "HistGradientBoostingClassifier",
    "HistGradientBoostingRegressor",
    "MLPClassifier",
    "MLPRegressor",
    "KMeans",
    "PCA",
    "TSNE",
    "train_test_split",
}
# Splitters that are random only when asked to shuffle.
_SHUFFLING_SPLITTERS = {"KFold", "StratifiedKFold", "GroupKFold", "RepeatedKFold"}
# np.random attributes that build or describe a generator rather than draw from the
# hidden global one.
_GENERATOR_API = {"default_rng", "RandomState", "Generator", "SeedSequence", "BitGenerator"}


def _is_int_literal(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    )


def _call_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _is_np_random(node: ast.AST) -> bool:
    # np.random / numpy.random
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "random"
        and isinstance(node.value, ast.Name)
        and node.value.id in {"np", "numpy"}
    )


def _violations(tree: ast.AST) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            for kw in node.keywords:
                if kw.arg in _SEED_KEYWORDS and _is_int_literal(kw.value):
                    found.append((node.lineno, f"{kw.arg}={ast.unparse(kw.value)} literal"))
            if name in _NEEDS_RANDOM_STATE and not any(
                kw.arg == "random_state" for kw in node.keywords
            ):
                found.append((node.lineno, f"{name}(...) without random_state"))
            if (
                name in _SHUFFLING_SPLITTERS
                and any(
                    kw.arg == "shuffle"
                    and not (isinstance(kw.value, ast.Constant) and kw.value.value is False)
                    for kw in node.keywords
                )
                and not any(kw.arg == "random_state" for kw in node.keywords)
            ):
                found.append((node.lineno, f"{name}(shuffle=...) without random_state"))
            if name in _SEEDED_CONSTRUCTORS:
                if node.args and _is_int_literal(node.args[0]):
                    found.append((node.lineno, f"{name}({ast.unparse(node.args[0])}) literal"))
                elif name == "default_rng" and not node.args and not node.keywords:
                    found.append((node.lineno, "unseeded default_rng()"))
        if (
            isinstance(node, ast.Attribute)
            and _is_np_random(node.value)
            and node.attr not in _GENERATOR_API
        ):
            found.append((node.lineno, f"global RNG np.random.{node.attr}"))
    return found


def test_no_seed_literals_or_unseeded_generators_in_src():
    problems = [
        f"{path.relative_to(SRC.parent)}:{line}: {what}"
        for path in sorted(SRC.rglob("*.py"))
        for line, what in _violations(ast.parse(path.read_text(encoding="utf-8")))
    ]
    assert not problems, "seed the component from `random_state`:\n" + "\n".join(problems)


def test_guard_flags_each_pattern():
    # The guard itself: each pattern it should catch, and the defaults it must allow.
    code = """
import numpy as np
def f(random_state: int = 42, *, seed: int = 0):
    np.random.default_rng(42)
    np.random.default_rng()
    np.random.RandomState(7)
    np.random.seed(1)
    np.random.choice([1, 2])
    Model(random_state=42)
    np.random.default_rng(random_state)
    Model(random_state=random_state)
    LogisticRegression(max_iter=1000)
    train_test_split(X, y, random_state=random_state)
    StratifiedKFold(n_splits=5)
    KFold(n_splits=5, shuffle=True)
    StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
"""
    lines = sorted(line for line, _ in _violations(ast.parse(code)))
    # A splitter without shuffle is deterministic and needs no seed.
    assert lines == [4, 5, 6, 7, 7, 8, 9, 12, 15]
