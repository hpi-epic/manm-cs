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

    def __init__(self, idx: int, parents: List['Variable'], noise: Optional[Noise] = None):
        self.idx = idx
        self.parents = parents
        self.noise = noise

    def sample(self, df: pd.DataFrame, num_observations: int) -> List[float]:
        raise NotImplementedError()
