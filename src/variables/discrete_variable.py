from typing import List, Optional, Dict, Tuple

import pandas as pd
import numpy as np

from src.noise.noise import Noise
from src.variables.variable import Variable, VariableType


class DiscreteVariable(Variable):
    type = VariableType.DISCRETE
    num_values: int
    mapping: Dict[Tuple[int, ...], int]

    def __init__(self, idx: int, num_values: int, parents: Optional[List[Variable]] = None,
                 mapping: Optional[Dict[Tuple[int, ...], int]] = None, noise: Optional[Noise] = None):
        parents = [] if parents is None else parents
        mapping = {} if mapping is None else mapping
        super(DiscreteVariable, self).__init__(idx=idx, parents=parents, noise=noise)
        self.num_values = num_values
        self.mapping = mapping

    def sample(self, df: pd.DataFrame, num_observations: int):
        if self._is_root():
            signal = pd.Series(np.zeros(num_observations, dtype=int))
        else:
            parent_idxs = [p.idx for p in self.parents]
            signal = df[parent_idxs].apply(lambda x: self.mapping[tuple(x)], axis=1)

        return (self.noise + signal) % self.num_values
