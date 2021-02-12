from abc import ABC

from src.prob_distributions.prob_distribution import ProbDistribution
from src.variables import VariableType


class DiscreteDistribution(ProbDistribution, ABC):
    type = VariableType.DISCRETE
    num_values: int

    def __init__(self, num_values: int):
        self.num_values = num_values

    def get_num_values(self) -> int:
        return self.num_values
