from src.prob_distributions.prob_distribution import ProbDistribution
from src.variables.discrete_variable import DiscreteVariable
from src.variables.root_variable import RootVariable


class DiscreteRootVariable(RootVariable, DiscreteVariable):

    def __init__(self, idx: int, prob_distribution: ProbDistribution):
        if prob_distribution.type != self.type:
            raise ValueError(f'Expected prob_distribution to be of type {self.type}, '
                             f'but was {prob_distribution.type}')

        num_values = prob_distribution.get_num_values()
        DiscreteVariable.__init__(self, idx=idx, num_values=num_values, parents=[],
                                  mapping=dict())
        RootVariable.__init__(self, prob_distribution=prob_distribution)
