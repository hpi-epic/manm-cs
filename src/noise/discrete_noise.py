from typing import List

import numpy as np
import pandas as pd
from numpy.random import random_sample

from src.noise.noise import Noise


class DiscreteNoise(Noise):
    prob_distribution: List[float]

    def __init__(self, prob_distribution: List[float]):
        self.prob_distribution = prob_distribution

    def __sample(self, num: int) -> List[int]:
        bins = np.add.accumulate(self.prob_distribution)
        return np.digitize(random_sample(num), bins)

    def __add__(self, other):
        if not isinstance(other, pd.Series):
            raise ValueError(f'Cannot add other to Noise')

        return other.apply(lambda x: x + self.__sample(num=1)[0])
