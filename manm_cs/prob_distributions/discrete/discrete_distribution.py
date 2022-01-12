from abc import ABC

from manm_cs.prob_distributions.prob_distribution import ProbDistribution
from manm_cs.variables import VariableType


class DiscreteDistribution(ProbDistribution, ABC):
    type = VariableType.DISCRETE
    num_values: int

    def __init__(self, num_values: int):
        self.num_values = num_values

    def get_num_values(self) -> int:
        return self.num_values
