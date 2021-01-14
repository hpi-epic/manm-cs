from abc import ABC
from typing import List

import pandas as pd

from src.prob_distributions.prob_distribution import ProbDistribution

class RootVariable(ABC):
    prob_distribution: ProbDistribution

    def __init__(self, prob_distribution: ProbDistribution):
        self.prob_distribution = prob_distribution

    def sample(self, df: pd.DataFrame, num_observations: int) -> List[float]:
        return self.prob_distribution.sample(num_observations=num_observations)
