from typing import List

import pandas as pd

from src.prob_distributions.prob_distribution import ProbDistribution
from src.variables.discrete_variable import DiscreteVariable
from src.variables.variable import VariableType


class DiscreteRootVariable(DiscreteVariable):
    type = VariableType.DISCRETE
    num_values: int
    prob_distribution: ProbDistribution

    def __init__(self, idx: int, prob_distribution: ProbDistribution):
        num_values = prob_distribution.get_num_values()
        super(DiscreteRootVariable, self).__init__(idx=idx, num_values=num_values, parents=[],
                                                   noise=None)
        self.prob_distribution = prob_distribution

    def sample(self, df: pd.DataFrame, num_observations: int) -> List[float]:
        return self.prob_distribution.sample(num_observations=num_observations)
