import numpy as np

from manm_cs.prob_distributions.continuous import ContinuousDistribution


class GaussianDistribution(ContinuousDistribution):
    mu: float
    sigma: float

    def __init__(self, mu: float, sigma: float):
        super(GaussianDistribution, self).__init__()
        self.mu = mu
        self.sigma = sigma

    def sample(self, num_observations: int) -> np.array:
        return np.random.normal(loc=self.mu, scale=self.sigma, size=num_observations)
