# tests/unit/backends/test_placeholders.py
import pandas as pd
import pytest

from syneva import evaluate
from syneva.core.errors import RegistryError


def test_longitudinal_not_yet_implemented():
    df = pd.DataFrame({"x": [1, 2]})
    with pytest.raises(RegistryError, match="longitudinal"):
        evaluate(df, df, data_type="longitudinal")


def test_relational_not_yet_implemented():
    df = pd.DataFrame({"x": [1, 2]})
    with pytest.raises(RegistryError, match="relational"):
        evaluate(df, df, data_type="relational")
