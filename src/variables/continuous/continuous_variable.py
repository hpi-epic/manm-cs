from typing import List, Optional, Callable

import pandas as pd

from src.noise.continous_noise import ContinuousNoise
from src.variables.variable import Variable, VariableType


class ContinuousVariable(Variable):
    type = VariableType.CONTINUOUS
    func: Callable[..., float]

    def __init__(self, idx: int, parents: List['Variable'], func: Callable[[float], float],
                 noise: Optional[ContinuousNoise] = None):
        super().__init__(idx=idx, parents=parents, noise=noise)
        self.func = func

    def sample(self, df: pd.DataFrame, num_observations: int):
        parent_idxs = [p.idx for p in self.parents]

        def mapper(x):
            return self.func(*x)

        return self.noise + df[parent_idxs].apply(mapper, axis=1)
