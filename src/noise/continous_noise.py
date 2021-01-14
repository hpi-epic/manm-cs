from src.noise.noise import Noise
from src.prob_distributions.continuous import ContinuousDistribution


class ContinuousNoise(Noise):

    def __init__(self, prob_distribution: ContinuousDistribution):
        super(ContinuousNoise, self).__init__(prob_distribution=prob_distribution)