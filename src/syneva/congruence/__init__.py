from syneva.congruence import ci_overlap as ci_overlap  # side-effect: registers CIOverlap
from syneva.congruence import (
    correlation_diff as correlation_diff,  # side-effect: registers CorrelationDifference
)
from syneva.congruence import (
    dimension_wise_means as dimension_wise_means,  # side-effect: registers DimensionWiseMeans
)
from syneva.congruence import extended as extended  # side-effect: registers extended metrics
from syneva.congruence import hellinger as hellinger  # side-effect: registers Hellinger
from syneva.congruence import ks as ks  # side-effect: registers KSStatistic
from syneva.congruence import pmse as pmse  # side-effect: registers PMSE
from syneva.congruence import quantile_mse as quantile_mse  # side-effect: registers QuantileMSE
from syneva.congruence import tvd as tvd  # side-effect: registers TotalVariationDistance
from syneva.congruence import wasserstein as wasserstein  # side-effect: registers Wasserstein1
