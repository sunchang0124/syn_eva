from __future__ import annotations

from typing import ClassVar

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from syneva.compliance._encode import encode_pair
from syneva.core.metric import MetricResult, MetricSpec
from syneva.core.registry import registry


@registry.register
class PMSE:
    """Propensity mean squared error (pMSE) of a real-vs-synthetic classifier.

    A main-effects logistic regression is fitted to the pooled rows (real 0,
    synthetic 1) and pMSE is the mean squared distance of its in-sample
    probabilities from the synthetic share c (Woo et al., 2009). An in-sample
    model fits noise, so pMSE sits above 0 even for two samples from one
    distribution. Its value under that null is estimated by refitting the same
    model with the labels permuted (Snoke et al., 2018), and the score is
    rescaled between that null (1) and the maximum c(1 - c) of a perfect
    classifier (0). ``pmse_ratio`` is pMSE divided by the null.

    The metric is skipped when the model has at least as many parameters as rows:
    a linear classifier can then split the rows in every possible way (Cover,
    1965), so pMSE carries no information.

    References
    ----------
    Cover, T. M. (1965). Geometrical and statistical properties of systems of linear
    inequalities with applications in pattern recognition. *IEEE Transactions on
    Electronic Computers*, EC-14(3), 326-334.

    Woo, M.-J., Reiter, J. P., Oganian, A., & Karr, A. F. (2009). Global measures
    of data utility for microdata masked for disclosure limitation. *Journal of
    Privacy and Confidentiality*, 1(1), 111-124.

    Snoke, J., Raab, G. M., Nowok, B., Dibben, C., & Slavkovic, A. (2018). General
    and specific utility measures for synthetic data. *Journal of the Royal
    Statistical Society: Series A*, 181(3), 663-688.
    """

    spec: ClassVar[MetricSpec] = MetricSpec(
        name="pmse",
        c="congruence",
        tier="core",
        data_types=frozenset({"static"}),
        requires_real=True,
        scope="table-level",
    )

    def __init__(self, random_state: int = 42, n_permutations: int = 20) -> None:
        if n_permutations < 1:
            raise ValueError(f"n_permutations must be at least 1, got {n_permutations}")
        self.random_state = random_state
        self.n_permutations = n_permutations

    def _pmse(self, X: np.ndarray, y: np.ndarray, c: float) -> float:
        # Fitted and scored on the same rows, as in Snoke et al. (2018).
        # A classifier failure propagates; the runner records it as a metric error.
        clf = LogisticRegression(max_iter=1000, random_state=self.random_state).fit(X, y)
        p = clf.predict_proba(X)[:, 1]
        return float(np.mean((p - c) ** 2))

    def compute(self, real, synthetic, meta) -> MetricResult:
        assert real is not None
        if len(real) == 0 or len(synthetic) == 0:
            return MetricResult(
                spec=self.spec,
                skip_reason="A dataset is empty, so there are no rows to classify.",
            )
        X_real, X_syn = encode_pair(real, synthetic, meta)
        if X_real.shape[1] == 0:
            return MetricResult(
                spec=self.spec,
                skip_reason="There are no numeric or categorical columns to classify on.",
            )
        X = np.vstack([X_real, X_syn])
        y = np.concatenate([np.zeros(len(X_real)), np.ones(len(X_syn))])
        # encode_pair scales only the numeric columns; scale the one-hot columns too
        X = StandardScaler().fit_transform(X)
        # A linear classifier with an intercept and d independent features can split
        # N points in general position in every possible way when N <= d + 1 (Cover,
        # 1965). With that many parameters the model fits any labelling, permuted ones
        # included, so pMSE says nothing about the data.
        # The rank (an SVD) is only needed when the column count allows a skip.
        n_params = X.shape[1] + 1
        if n_params >= len(X):
            n_params = int(np.linalg.matrix_rank(X)) + 1
        if n_params >= len(X):
            return MetricResult(
                spec=self.spec,
                skip_reason=(
                    f"The classifier has {n_params} parameters for {len(X)} rows, so it "
                    "can fit any labelling. Use fewer high-cardinality categorical "
                    "columns or more rows."
                ),
            )
        c = len(X_syn) / len(X)
        pmse = self._pmse(X, y, c)
        # Null reference: the pMSE the same model reaches when the real/synthetic
        # labels carry no information, estimated by permuting them (Snoke et al., 2018).
        rng = np.random.default_rng(self.random_state)
        null_pmse = float(
            np.mean([self._pmse(X, rng.permutation(y), c) for _ in range(self.n_permutations)])
        )
        # A perfect classifier (p = 0 on real rows, 1 on synthetic) attains the
        # maximum pMSE of c(1 - c), which maps fully separable data to 0.
        max_pmse = c * (1.0 - c)
        score = float(np.clip(1.0 - (pmse - null_pmse) / (max_pmse - null_pmse), 0.0, 1.0))
        # With only constant features the model fits just the intercept, so pMSE and
        # its null are both exactly 0: the data can't be told apart.
        ratio = pmse / null_pmse if null_pmse > 0 else 1.0
        return MetricResult(
            spec=self.spec,
            scalars={
                "score": score,
                "pmse": pmse,
                "pmse_null": null_pmse,
                "pmse_ratio": ratio,
            },
        )
