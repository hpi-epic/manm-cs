from src.prob_distributions.continuous import GaussianDistribution
from src.noise import ContinuousNoise

class GaussianNoiseBuilder:
    mu: float = .0
    sigma: float

    def with_sigma(self, sigma: float) -> 'GaussianNoiseBuilder':
        self.sigma = sigma
        return self

    def build(self) -> ContinuousNoise:
        prob_distribution = GaussianDistribution(mu=self.mu, sigma=self.sigma)
        return ContinuousNoise(prob_distribution=prob_distribution)
