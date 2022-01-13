from abc import ABC

from manm_cs.prob_distributions.prob_distribution import ProbDistribution
from manm_cs.variables import VariableType


class ContinuousDistribution(ProbDistribution, ABC):
    type = VariableType.CONTINUOUS
