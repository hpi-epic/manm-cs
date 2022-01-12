from manm_cs.noise.noise import Noise
from manm_cs.prob_distributions.discrete import DiscreteDistribution


class DiscreteNoise(Noise):

    def __init__(self, prob_distribution: DiscreteDistribution):
        super(DiscreteNoise, self).__init__(prob_distribution=prob_distribution)

    def get_num_values(self):
        return self.prob_distribution.get_num_values()
