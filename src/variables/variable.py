from abc import ABC
from typing import List, Optional

import pandas as pd

from src.noise.noise import Noise
from src.variables import VariableType


class Variable(ABC):
    idx: int
    type: VariableType
    parents: List['Variable']
    noise: Noise

    def __init__(self, idx: int, parents: List['Variable'], noise: Noise):
        self.idx = idx
        self.parents = parents
        self.noise = noise

        if noise.get_type() != self.type:
            raise ValueError(f'Expected noise to be of type {self.type}, '
                             f'but was {noise.get_type()}')

    def _is_root(self) -> bool:
        return len(self.parents) == 0

    def sample(self, df: pd.DataFrame, num_observations: int) -> pd.Series:
        raise NotImplementedError()
