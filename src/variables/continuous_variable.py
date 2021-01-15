from typing import List, Optional, Callable, Dict, Tuple

import numpy as np
import pandas as pd

from src.noise.continous_noise import ContinuousNoise
from src.variables.variable import Variable, VariableType


class ContinuousVariable(Variable):
    type = VariableType.CONTINUOUS
    continuous_mapper_func: Callable[..., float]

    def __init__(self, idx: int, noise: ContinuousNoise, parents: Optional[List[Variable]] = None, betas: Optional[List[float]] = None, mapping: Optional[Dict[Tuple[int], int]] = None):
        parents = [] if parents is None else parents
        betas = [] if betas is None else betas
        mapping = {} if mapping is None else mapping
        super(ContinuousVariable, self).__init__(idx=idx, parents=parents, noise=noise)
        self.continuous_mapper_func = self.__create_continuous_mapper_func(betas=betas)
        self.mapping = mapping

        # Input parameter validation
        if len(betas) != len(self._get_continous_parents()):
            raise ValueError(f'There must be one beta value for each continuous parent. ' \
                                f'Expected {len(self._get_continous_parents())}, but were {len(betas)}')

    def __create_continuous_mapper_func(self, betas: List[float]) -> Callable[[pd.Series], float]:
        """Creates the mapper function for continuous parents

        """
        def func(parent_values: pd.Series):
            return np.sum([x * betas[i] for i, x in enumerate(parent_values)])

        return func

    def get_non_root_signal(self, df: pd.DataFrame) -> pd.Series:
        # Compute signal for continous parent variables
        continuous_parent_idxs = [p.idx for p in self._get_continous_parents()]
        continuous_signal = df[continuous_parent_idxs].apply(self.continuous_mapper_func, axis=1)

        # Compute signal for discrete parent variables
        discrete_parent_idxs = [p.idx for p in self._get_discrete_parents()]
        discrete_signal = df[discrete_parent_idxs].apply(lambda x: self.mapping[tuple(x)], axis=1) \
                            if len(discrete_parent_idxs) > 0 \
                            else pd.Series(np.zeros(len(df)))

        # Aggregate continuous and discrete signal terms into overall signal term
        return continuous_signal + discrete_signal

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
