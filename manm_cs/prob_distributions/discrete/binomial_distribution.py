import numpy as np

from manm_cs.prob_distributions.discrete.discrete_distribution import DiscreteDistribution


class BinomialDistribution(DiscreteDistribution):
    probability: float

    def __init__(self, probability: float):
        self.probability = probability
        super(BinomialDistribution, self).__init__(num_values=2)

    def sample(self, num_observations: int) -> np.array:
        return np.random.binomial(1, self.probability, num_observations)
