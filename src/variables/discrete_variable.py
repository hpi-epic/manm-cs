from typing import List, Optional, Dict, Tuple

import pandas as pd
import numpy as np

from src.noise.discrete_noise import DiscreteNoise
from src.variables.variable import Variable, VariableType


class DiscreteVariable(Variable):
    type = VariableType.DISCRETE
    num_values: int
    mapping: Dict[Tuple[int, ...], int]

    def __init__(self, idx: int, num_values: int, noise: DiscreteNoise, parents: Optional[List['DiscreteVariable']] = None,
                 mapping: Optional[Dict[Tuple[int, ...], int]] = None):
        parents = [] if parents is None else parents
        mapping = {} if mapping is None else mapping
        super(DiscreteVariable, self).__init__(idx=idx, parents=parents, noise=noise)
        self.num_values = num_values
        self.mapping = mapping

        # Input parameter validation
        if len(self._get_continous_parents()) > 0:
            raise ValueError(f'The discrete variable {self.id} must only ' \
                                f'have discrete parents, but were {self._get_continous_parents()}')
        if self.noise.get_num_values() != self.num_values:
            raise ValueError(f'The noise term must define a probability distribution over all possible values. ' \
                             f'Expected num_values equal to {self.num_values}, but received {self.noise.get_num_values()}')

    def sample(self, df: pd.DataFrame, num_observations: int) -> pd.Series:
        if self._is_root():
            # If the variable is a root variable, the sampling is determined by the noise term only
            signal = pd.Series(np.zeros(num_observations, dtype=int))
        else:
            # If the variable has one or more parent variables, the sampling is driven 
            # by a combination of signal and noise term
            parent_idxs = [p.idx for p in self.parents]
            signal = df[parent_idxs].apply(lambda x: self.mapping[tuple(x)], axis=1)

        return (self.noise + signal) % self.num_values
