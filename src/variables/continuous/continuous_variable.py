from typing import List, Optional, Callable

import numpy as np
import pandas as pd

from src.noise.continous_noise import ContinuousNoise
from src.variables.variable import Variable, VariableType


class ContinuousVariable(Variable):
    type = VariableType.CONTINUOUS
    func: Callable[..., float]

    def __init__(self, idx: int, parents: List[Variable], betas: List[float],
                 noise: Optional[ContinuousNoise] = None):
        super(ContinuousVariable, self).__init__(idx=idx, parents=parents, noise=noise)
        self.func = self.__create_func(betas=betas)

    def __create_func(self, betas: List[float]):
        def func(parent_values):
            return np.sum([x * betas[i] for i, x in enumerate(parent_values)])

        return func

    def sample(self, df: pd.DataFrame, num_observations: int):
        parent_idxs = [p.idx for p in self.parents]

        return self.noise + df[parent_idxs].apply(self.func, axis=1)
