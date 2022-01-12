from abc import ABC

import numpy as np

from manm_cs.variables import VariableType


class ProbDistribution(ABC):
    type: VariableType

    def sample(self, num_observations: int) -> np.array:
        raise NotImplementedError()
