from typing import List, Optional

import pandas as pd

from src.noise.noise import Noise
from src.variables.variable import Variable, VariableType


class DiscreteVariable(Variable):
    type = VariableType.DISCRETE
    num_values: int

    def __init__(self, idx: int, num_values: int, parents: List['Variable'],
                 noise: Optional[Noise] = None):
        super(DiscreteVariable, self).__init__(idx=idx, parents=parents, noise=noise)
        self.num_values = num_values

    def sample(self, df: pd.DataFrame, num_observations: int):
        mapping = {
            0: 1,
            1: 2
        }
        return self.noise + df[self.parents[0].idx].map(mapping)
