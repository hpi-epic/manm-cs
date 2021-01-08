from src.noise.noise import Noise
from src.prob_distributions.discrete import DiscreteDistribution


class DiscreteNoise(Noise):

    def __init__(self, prob_distribution: DiscreteDistribution):
        super(DiscreteNoise, self).__init__(prob_distribution=prob_distribution)
