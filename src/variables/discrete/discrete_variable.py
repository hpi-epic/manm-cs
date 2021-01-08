from typing import List, Optional, Dict, Tuple

import pandas as pd

from src.noise.noise import Noise
from src.variables.variable import Variable, VariableType


class DiscreteVariable(Variable):
    type = VariableType.DISCRETE
    num_values: int
    mapping: Dict[Tuple[int, ...], int]

    def __init__(self, idx: int, num_values: int, parents: List[Variable],
                 mapping: Dict[Tuple[int, ...], int], noise: Optional[Noise] = None):
        super(DiscreteVariable, self).__init__(idx=idx, parents=parents, noise=noise)
        self.num_values = num_values
        self.mapping = mapping

    def sample(self, df: pd.DataFrame, num_observations: int):
        parent_idxs = [p.idx for p in self.parents]

        def mapper(x):
            return self.mapping[tuple(x)]

        return (self.noise + df[parent_idxs].apply(mapper, axis=1)) % self.num_values
