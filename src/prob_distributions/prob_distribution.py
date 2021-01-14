from typing import List

import numpy as np

from src.variables import VariableType


class ProbDistribution:
    type: VariableType

    def sample(self, num_observations: int) -> np.array:
        raise NotImplementedError()
