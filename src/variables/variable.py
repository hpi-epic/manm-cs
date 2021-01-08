from abc import ABC
from enum import Enum
from typing import List, Optional

import pandas as pd

from src.noise.noise import Noise


class VariableType(Enum):
    DISCRETE = 1
    CONTINUOUS = 2


class Variable(ABC):
    idx: int
    type: VariableType
    parents: List['Variable']
    noise: Noise

    def __init__(self, idx: int, parents: List['Variable'], noise: Optional[Noise] = None):
        self.idx = idx
        self.parents = parents
        self.noise = noise

    def __func(self, variable: 'Variable'):
        raise NotImplementedError()

    def sample(self, df: pd.DataFrame, num_observations: int) -> List[float]:
        raise NotImplementedError()
