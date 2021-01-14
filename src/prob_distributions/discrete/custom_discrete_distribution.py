from typing import List

import numpy as np
from numpy.random import random_sample

from src.prob_distributions.discrete.discrete_distribution import DiscreteDistribution


class CustomDiscreteDistribution(DiscreteDistribution):
    probs: List[float]

    def __init__(self, probs: List[float]):
        self.probs = probs

    def get_num_values(self) -> int:
        return len(self.probs)

    def sample(self, num_observations: int) -> np.array:
        bins = np.add.accumulate(self.probs)
        return np.digitize(random_sample(self.get_num_values()), bins)
