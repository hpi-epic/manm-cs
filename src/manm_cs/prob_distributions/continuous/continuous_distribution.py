from abc import ABC

from src.manm_cs.prob_distributions.prob_distribution import ProbDistribution
from src.manm_cs.variables import VariableType


class ContinuousDistribution(ProbDistribution, ABC):
    type = VariableType.CONTINUOUS
