import numpy as np

from manm_cs.prob_distributions.discrete import CustomDiscreteDistribution
from manm_cs.prob_distributions.discrete import DiscreteDistribution


class RandomDiscreteDistribution(DiscreteDistribution):
    distribution: CustomDiscreteDistribution

    def __init__(self, num_values: int):
        super(RandomDiscreteDistribution, self).__init__(num_values=num_values)
        self.distribution = self.__generate_prob_distribution()

    def __generate_prob_distribution(self):
        probs = []
        for i in range(self.num_values - 1):
            next_prob = np.random.uniform(0, 1 - sum(probs))
            probs.append(next_prob)
        probs.append(1 - sum(probs))

        return CustomDiscreteDistribution(probs=probs)

    def sample(self, num_observations: int) -> np.array:
        return self.distribution.sample(num_observations=num_observations)
