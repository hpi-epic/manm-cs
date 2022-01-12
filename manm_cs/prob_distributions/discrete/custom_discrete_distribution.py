from typing import List

import numpy as np
from numpy.random import random_sample

from manm_cs.prob_distributions.discrete.discrete_distribution import DiscreteDistribution


class CustomDiscreteDistribution(DiscreteDistribution):
    probs: List[float]

    def __init__(self, probs: List[float]):
        if abs(sum(probs) - 1.0) > 0.001:
            raise ValueError(f'Expected the provided probabilities to approximately sum up to 1, '
                             f'but sum was {sum(probs)} '
                             f'[diff={abs(sum(probs) - 1.0)}, tolerance={0.001}]')
        self.probs = probs
        super(CustomDiscreteDistribution, self).__init__(num_values=len(self.probs))

    def sample(self, num_observations: int) -> np.array:
        bins = np.add.accumulate(self.probs)
        # example bins = [0.8,0.9,1.0]
        # index of interval random number is within 0 = [0,0.8] 1=[0.8,0.9] 2= [0.9,1]
        return np.digitize(random_sample(num_observations), bins)
