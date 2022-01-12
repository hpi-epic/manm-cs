import numpy as np

from manm_cs.prob_distributions.continuous import ContinuousDistribution, GaussianDistribution


class BimodalDistribution(ContinuousDistribution):
    prob_dist1: GaussianDistribution
    prob_dist2: GaussianDistribution

    def __init__(self, prob_dist1: GaussianDistribution, prob_dist2: GaussianDistribution):
        super(BimodalDistribution, self).__init__()
        self.prob_dist1 = prob_dist1
        self.prob_dist2 = prob_dist2

    def sample(self, num_observations: int) -> np.array:
        num_obs1 = int(num_observations / 2)
        samples1 = self.prob_dist1.sample(num_observations=num_obs1)
        samples2 = self.prob_dist2.sample(num_observations=num_observations - num_obs1)

        samples = np.concatenate([samples1, samples2])
        np.random.shuffle(samples)

        return samples
