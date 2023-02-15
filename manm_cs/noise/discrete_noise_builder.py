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
        # Signal_noise = portions of uniform noise over all discrete values, i.e., signal_to_noise_ratio=0 for no noise and signal_to_noise_ratio =1 for uniform noise
        signal_noise = self.signal_to_noise_ratio / self.num_discrete_values
        
        # Probabilities
        probs = [(1-self.signal_to_noise_ratio)+signal_noise] + [(signal_noise) for _ in range(self.num_discrete_values - 1)]

        return DiscreteNoise(prob_distribution=CustomDiscreteDistribution(probs=probs))
