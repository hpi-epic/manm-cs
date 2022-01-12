from manm_cs.noise import ContinuousNoise
from manm_cs.prob_distributions.continuous import GaussianDistribution


class GaussianNoiseBuilder:
    mu: float = .0
    sigma: float

    def with_sigma(self, sigma: float) -> 'GaussianNoiseBuilder':
        self.sigma = sigma
        return self

    def build(self) -> ContinuousNoise:
        prob_distribution = GaussianDistribution(mu=self.mu, sigma=self.sigma)
        return ContinuousNoise(prob_distribution=prob_distribution)
