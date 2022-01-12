import numpy as np

from manm_cs.prob_distributions.discrete.discrete_distribution import DiscreteDistribution


class UniformDiscreteDistribution(DiscreteDistribution):
    num_values: int

    def __init__(self, num_values: int):
        super(UniformDiscreteDistribution, self).__init__(num_values=num_values)

    def sample(self, num_observations: int) -> np.array:
        return np.random.randint(low=0, high=self.num_values - 1, size=num_observations)
