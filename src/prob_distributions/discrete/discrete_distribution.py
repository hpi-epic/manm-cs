from abc import ABC

from src.prob_distributions.prob_distribution import ProbDistribution
from src.variables import VariableType


class DiscreteDistribution(ProbDistribution, ABC):
    type = VariableType.DISCRETE

    def get_num_values(self) -> int:
        raise NotImplementedError()
