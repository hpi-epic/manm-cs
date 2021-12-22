from abc import ABC

from ..prob_distribution import ProbDistribution
from ...variables import VariableType


class ContinuousDistribution(ProbDistribution, ABC):
    type = VariableType.CONTINUOUS
