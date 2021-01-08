from typing import List

from src.variables.variable import VariableType


class ProbDistribution:
    type: VariableType

    def sample(self, num_observations: int) -> List[float or int]:
        raise NotImplementedError()
