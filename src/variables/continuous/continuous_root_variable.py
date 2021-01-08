from src.prob_distributions.continuous import ContinuousDistribution
from src.variables.continuous import ContinuousVariable
from src.variables.root_variable import RootVariable


class ContinuousRootVariable(RootVariable, ContinuousVariable):

    def __init__(self, idx: int, prob_distribution: ContinuousDistribution):
        if prob_distribution.type != self.type:
            raise ValueError(f'Expected prob_distribution to be of type {self.type}, '
                             f'but was {prob_distribution.type}')

        ContinuousVariable.__init__(self, idx=idx, parents=[], func=lambda x: x)
        RootVariable.__init__(self, prob_distribution=prob_distribution)
