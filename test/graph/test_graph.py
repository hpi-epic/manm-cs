from manm_cs.graph.graph import Graph
from manm_cs.noise import DiscreteNoise
from manm_cs.prob_distributions import BinomialDistribution, CustomDiscreteDistribution, UniformDiscreteDistribution
from manm_cs.variables.discrete_variable import DiscreteVariable
import networkx as nx


def create_simple_discrete_model2():
    A = DiscreteVariable(idx=0, num_values=2, noise=DiscreteNoise(prob_distribution=BinomialDistribution(probability=.25)))
    B = DiscreteVariable(idx=1, num_values=3, noise=DiscreteNoise(UniformDiscreteDistribution(num_values=3)))
    C = DiscreteVariable(idx=2, num_values=3, parents=[A, B],
                         noise=DiscreteNoise(
                             prob_distribution=CustomDiscreteDistribution(probs=[.5, .2, .3])))
    variables = [A, B, C]
    return Graph(variables=variables)


def test_to_networkx_graph():
    expected = nx.DiGraph()
    expected.add_node(0)
    expected.add_node(1)
    expected.add_node(2)
    expected.add_edge(0, 2)
    expected.add_edge(1, 2)

    graph = create_simple_discrete_model2()
    actual = graph.to_networkx_graph()

    assert nx.is_isomorphic(expected, actual)
