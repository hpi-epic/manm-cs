from manm_cs.noise import DiscreteNoise
from manm_cs.prob_distributions import CustomDiscreteDistribution


class DiscreteNoiseBuilder:
    signal_to_noise_ratio: float
    num_discrete_values: int

    def with_signal_to_noise_ratio(self, signal_to_noise_ratio: float) -> 'DiscreteNoiseBuilder':
        self.signal_to_noise_ratio = signal_to_noise_ratio
        return self

    def with_num_discrete_values(self, num_discrete_values: int) -> 'DiscreteNoiseBuilder':
        self.num_discrete_values = num_discrete_values
        return self

    def build(self) -> DiscreteNoise:
        signal = self.signal_to_noise_ratio
        num_noise_classes = self.num_discrete_values - 1
        avg_noise = (1 - signal) / num_noise_classes
        probs = [signal] + [avg_noise for _ in range(num_noise_classes)]

        return DiscreteNoise(prob_distribution=CustomDiscreteDistribution(probs=probs))
