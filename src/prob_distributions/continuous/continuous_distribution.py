from src.prob_distributions.prob_distribution import ProbDistribution
from src.variables.variable import VariableType


class ContinuousDistribution(ProbDistribution):
    type = VariableType.CONTINUOUS
