import numpy as np

from src.prob_distributions.discrete.discrete_distribution import DiscreteDistribution


class BinomialDistribution(DiscreteDistribution):
    probability: float

    def __init__(self, probability: float):
        self.probability = probability

    def get_num_values(self):
        return 2

    def sample(self, num_observations: int):
        return np.random.binomial(1, self.probability, num_observations)
