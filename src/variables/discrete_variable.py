from typing import List, Dict, Tuple, Callable

import numpy as np
import pandas as pd

from src.noise.discrete_noise import DiscreteNoise
from src.prob_distributions import CustomDiscreteDistribution
from src.variables.variable import Variable, VariableType


class DiscreteVariable(Variable):
    type = VariableType.DISCRETE
    num_values: int
    mapping: Dict[Tuple[int, ...], int]

    def __init__(self, idx: int, num_values: int, noise: DiscreteNoise,
                 parents: List['DiscreteVariable'] = None):
        parents = [] if parents is None else parents
        super(DiscreteVariable, self).__init__(idx=idx, parents=parents, noise=noise)
        self.num_values = num_values
        self.logit_mapper_func = self.__create_logit_mapper_func(functions=functions)

        # Input parameter validation
        if self.noise.get_num_values() != self.num_values:
            raise ValueError(
                f'The noise term must define a probability distribution over all possible values. '
                f'Expected num_values equal to {self.num_values}, but received {self.noise.get_num_values()}')
    
    def mult_logit_function(self, value) -> pd.Series:
        categories =  list(range(1, self.num_values))
                
        e_sum = np.sum([np.exp(k*value) for  k in enumerate(categories)])
        softmax = [np.exp(k*value) / e_sum for  k in enumerate(categories)]
        
        return softmax
     
    def __create_logit_mapper_func(self) -> Callable[[pd.Series], int]:
        """Creates the a multinomial logit function for each of the continuous parents

        """            
        def func(parent_values: pd.Series):
            return np.sum([np.where(np.random.multinomial(1,mult_logit_function(value=value)) == 1)[0][0] for  value in enumerate(parent_values)])

        return func

    def get_non_root_signal(self, df: pd.DataFrame) -> pd.Series:
        # Compute signal for continuous parent variables
        continuous_parent_idxs = [p.idx for p in self._get_continuous_parents()]
        continuous_signal = df[continuous_parent_idxs].apply(self.logit_mapper_func, axis=1) \
            if len(continuous_parent_idxs) > 0 \
            else pd.Series(np.zeros(len(df)))

        # Compute signal for discrete parent variables
        discrete_parent_idxs = [p.idx for p in self._get_discrete_parents()]
        discrete_signal = df[discrete_parent_idxs].apply(sum, axis=1) \
            if len(discrete_parent_idxs) > 0 \
            else pd.Series(np.zeros(len(df)))

        # Aggregate continuous and discrete signal terms into overall signal term
        return continuous_signal + discrete_signal

    def sample(self, df: pd.DataFrame, num_observations: int) -> pd.Series:
        if self._is_root():
            # If the variable is a root variable, the sampling is determined by the noise term only
            props = [1] + list(np.zeros(self.num_values - 1))
            signal = pd.Series(CustomDiscreteDistribution(props).sample(
                num_observations=num_observations)) # This will return a list of zeros
        else:
            # If the variable has one or more parent variables, the sampling is driven 
            # by a combination of signal and noise term
            signal = self.get_non_root_signal(df=df)
            
        # Combine noise and signal terms and apply ring transformation
        return (self.noise + signal) % self.num_values
