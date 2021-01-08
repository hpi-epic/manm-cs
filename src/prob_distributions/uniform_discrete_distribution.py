import numpy as np

from src.prob_distributions.prob_distribution import ProbDistribution


class UniformDiscreteDistribution(ProbDistribution):
    num_values: int

    def __init__(self, num_values: int):
        self.num_values = num_values

    def get_num_values(self):
        return self.num_values

    def sample(self, num_observations: int):
        return np.random.randint(low=0, high=self.num_values-1, size=num_observations)
