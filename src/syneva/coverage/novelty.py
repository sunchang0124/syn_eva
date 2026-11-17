from __future__ import annotations

from typing import ClassVar

import pandas as pd

from syneva.core.metadata import Metadata
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry
from syneva.core.rows import comparable_row_keys


@registry.register
class NoveltyRate:
    """Share of synthetic rows that are not exact copies of a real row.

    Novelty is the fraction of generated samples that do not occur in the training
    data (Polykovskiy et al., 2020; Brown et al., 2019). It is not uniqueness, the
    fraction of generated samples that are distinct from each other: synthetic rows
    that repeat one another but copy no real row are all novel.

    Rows are compared on the same keys as ``identical_match_rate``: columns aligned,
    missing values normalised and ID columns left out. Every synthetic row counts,
    including repeats, so ``novelty_rate == 1 - identical_match_rate`` and
    ``duplicates`` is the number of synthetic rows that copy a real row.

    References
    ----------
    Polykovskiy, D., Zhebrak, A., Sanchez-Lengeling, B., Golovanov, S., Tatanov, O.,
    Belyaev, S., Kurbanov, R., Artamonov, A., Aladinskiy, V., Veselov, M., Kadurin,
    A., Johansson, S., Chen, H., Nikolenko, S., Aspuru-Guzik, A., & Zhavoronkov, A.
    (2020). Molecular Sets (MOSES): A benchmarking platform for molecular generation
    models. *Frontiers in Pharmacology*, 11, 565644.
    https://doi.org/10.3389/fphar.2020.565644

    Brown, N., Fiscato, M., Segler, M. H. S., & Vaucher, A. C. (2019). GuacaMol:
    Benchmarking models for de novo molecular design. *Journal of Chemical
    Information and Modeling*, 59(3), 1096-1108.
    https://doi.org/10.1021/acs.jcim.8b00839
    """

    spec: ClassVar[MetricSpec] = MetricSpec(
        name="novelty_rate",
        c="coverage",
        tier="core",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
    )

    def compute(
        self,
        real: pd.DataFrame | None,
        synthetic: pd.DataFrame,
        meta: Metadata,
    ) -> MetricResult:
        assert real is not None

        if len(synthetic) == 0:
            return MetricResult(
                spec=self.spec,
                skip_reason="The synthetic data is empty.",
            )

        # Column alignment, NaN normalization and ID-column exclusion live in
        # comparable_row_keys, shared with IdenticalMatchRate.
        real_keys, syn_keys = comparable_row_keys(real, synthetic, meta)
        real_set: set[tuple] = set(real_keys)
        novel = sum(1 for t in syn_keys if t not in real_set)
        n_syn = len(synthetic)
        duplicates = float(n_syn - novel)
        rate = float(novel / n_syn)
        # clamp to [0, 1] to guard against floating-point edge cases
        rate = min(1.0, max(0.0, rate))

        return MetricResult(
            spec=self.spec,
            scalars={"score": rate, "novelty_rate": rate, "duplicates": duplicates},
        )
