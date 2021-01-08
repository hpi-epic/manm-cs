from src.prob_distributions.prob_distribution import ProbDistribution
from src.variables.variable import VariableType


class DiscreteDistribution(ProbDistribution):
    type = VariableType.DISCRETE
