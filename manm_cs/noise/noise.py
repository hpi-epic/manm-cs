from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from manm_cs.variables import VariableType

if TYPE_CHECKING:
    from manm_cs.prob_distributions.prob_distribution import ProbDistribution


class Noise:
    prob_distribution: 'ProbDistribution'

    def __init__(self, prob_distribution: 'ProbDistribution'):
        self.prob_distribution = prob_distribution

    def get_type(self) -> VariableType:
        return self.prob_distribution.type

    def __sample(self, num: int) -> np.array:
        return self.prob_distribution.sample(num_observations=num)

    def __add__(self, other) -> pd.Series:
        if not isinstance(other, pd.Series):
            raise ValueError(f'Cannot add other to Noise')

        return other.apply(lambda x: x + self.__sample(num=1)[0])
