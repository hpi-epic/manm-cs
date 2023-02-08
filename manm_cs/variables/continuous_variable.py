from typing import List, Optional, Callable, Dict, Tuple

import numpy as np
import pandas as pd

from manm_cs.noise.continous_noise import ContinuousNoise
from manm_cs.variables.variable import Variable, VariableType


class ContinuousVariable(Variable):
    type = VariableType.CONTINUOUS
    continuous_mapper_func: Callable[..., float]
    def __init__(self, idx: int, noise: ContinuousNoise, parents: Optional[List[Variable]] = None,
                 functions: Optional[List[Callable[..., float]]] = None, betas: Optional[List[float]] = None,
                 scale_parents: Optional[bool] = False):
        parents = [] if parents is None else parents
        functions = [] if functions is None else functions
        super(ContinuousVariable, self).__init__(idx=idx, parents=parents, noise=noise, functions=functions,
                                                 scale_parents = scale_parents)
        self.betas = betas
        self.continuous_mapper_func = self.__create_continuous_mapper_func(functions=functions, betas=betas)

        # Input parameter validation
        if len(functions) != len(self._get_continuous_parents()):
            raise ValueError(f'There must be one function for each continuous parent. Expected '
                             f'{len(self._get_continuous_parents())}, but were {len(functions)}')

    def __create_continuous_mapper_func(self, functions: List[Callable[..., float]], betas: List[float]) -> Callable[[pd.Series], float]:
        """Creates the mapper function for continuous parents

        """

        def func(parent_values: pd.Series):
            return np.sum([betas[i] * functions[i](value) for i, value in enumerate(parent_values)])

        return func

    def get_non_root_signal(self, df: pd.DataFrame) -> pd.Series:
        # Compute signal for continuous parent variables
        continuous_parent_idxs = [p.idx for p in self._get_continuous_parents()]
        continuous_signal = df[continuous_parent_idxs].apply(self.continuous_mapper_func, axis=1) \
            if len(continuous_parent_idxs) > 0 \
            else pd.Series(np.zeros(len(df)))

        # Compute signal for discrete parent variables
        discrete_parent_idxs = [p.idx for p in self._get_discrete_parents()]
        discrete_signal = df[discrete_parent_idxs].apply(sum, axis=1) \
            if len(discrete_parent_idxs) > 0 \
            else pd.Series(np.zeros(len(df)))

        # Aggregate continuous and discrete signal terms into overall signal term
        return (continuous_signal + discrete_signal) / len(self.parents) \
            if self.scale_parents else continuous_signal + discrete_signal

    def sample(self, df: pd.DataFrame, num_observations: int) -> pd.Series:
        if self._is_root():
            # If the variable is a root variable, the sampling is determined by the noise term only
            signal = pd.Series(np.zeros(num_observations, dtype=float))
        else:
            # If the variable has one or more parent variables, the sampling is driven
            # by a combination of signal and noise term
            signal = self.get_non_root_signal(df=df)

        # Add noise and signal terms
        return self.noise + signal
