from abc import ABC
from typing import List, Callable, Optional

import pandas as pd

from manm_cs.noise.noise import Noise
from manm_cs.variables import VariableType


class Variable(ABC):
    idx: int
    type: VariableType
    parents: List['Variable']
    noise: Noise
    functions: List[Callable[..., float]]
    scale_parents: bool

    def __init__(self, idx: int, parents: List['Variable'], noise: Noise, functions: Optional[List[Callable[..., float]]] = None,
                 scale_parents: Optional[bool] = False):
        self.idx = idx
        self.parents = parents
        self.noise = noise
        self.functions = functions
        self.scale_parents = scale_parents

        # Input parameter validation
        if noise.get_type() != self.type:
            raise ValueError(f'Expected noise to be of type {self.type}, '
                             f'but was {noise.get_type()}')

    def _is_root(self) -> bool:
        return len(self.parents) == 0

    def _get_continuous_parents(self) -> List['Variable']:
        return list(filter(lambda p: p.type == VariableType.CONTINUOUS, self.parents))

    def _get_discrete_parents(self) -> List['Variable']:
        return list(filter(lambda p: p.type == VariableType.DISCRETE, self.parents))

    def sample(self, df: pd.DataFrame, num_observations: int) -> pd.Series:
        raise NotImplementedError()
