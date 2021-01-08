from src.graph.graph import Graph
from src.noise.continous_noise import ContinuousNoise
from src.noise.discrete_noise import DiscreteNoise
from src.prob_distributions import BinomialDistribution, UniformDiscreteDistribution
from src.prob_distributions.continuous.gaussian_distribution import GaussianDistribution
from src.prob_distributions.discrete.custom_discrete_distribution import CustomDiscreteDistribution
from src.variables.continuous import ContinuousRootVariable
from src.variables.continuous import ContinuousVariable
from src.variables.discrete.discrete_root_variable import DiscreteRootVariable
from src.variables.discrete.discrete_variable import DiscreteVariable


def create_simple_discrete_model1():
    A = DiscreteRootVariable(idx=0, prob_distribution=BinomialDistribution(probability=.75))
    B = DiscreteVariable(idx=1, num_values=3, parents=[A], mapping={(0,): 1, (1,): 2},
                         noise=DiscreteNoise(
                             prob_distribution=CustomDiscreteDistribution(probs=[.5, .2, .3, .5])))
    variables = [A, B]
    return Graph(variables=variables)


def create_simple_discrete_model2():
    A = DiscreteRootVariable(idx=0, prob_distribution=BinomialDistribution(probability=.25))
    B = DiscreteRootVariable(idx=1, prob_distribution=UniformDiscreteDistribution(num_values=3))
    C = DiscreteVariable(idx=2, num_values=3, parents=[A, B],
                         mapping={
                             (0, 0): 1,
                             (0, 1): 2,
                             (0, 2): 1,
                             (1, 0): 2,
                             (1, 1): 2,
                             (1, 2): 2
                         },
                         noise=DiscreteNoise(
                             prob_distribution=CustomDiscreteDistribution(probs=[.5, .2, .3, .5])))
    variables = [A, B, C]
    return Graph(variables=variables)


def create_simple_continuous_model1():
    A = ContinuousRootVariable(idx=0, prob_distribution=GaussianDistribution(mu=0, sigma=1))
    B = ContinuousVariable(idx=1, parents=[A], func=lambda x: 2 * x,
                           noise=ContinuousNoise(
                               prob_distribution=GaussianDistribution(mu=0, sigma=1)))
    variables = [A, B]
    return Graph(variables=variables)


def create_simple_continuous_model2():
    A = ContinuousRootVariable(idx=0, prob_distribution=GaussianDistribution(mu=0, sigma=1))
    B = ContinuousRootVariable(idx=1, prob_distribution=GaussianDistribution(mu=10, sigma=5))
    C = ContinuousVariable(idx=2, parents=[A, B], func=lambda x1, x2: 2 * x1 + x2,
                           noise=ContinuousNoise(
                               prob_distribution=GaussianDistribution(mu=0, sigma=1)))
    variables = [A, B, C]
    return Graph(variables=variables)


if __name__ == '__main__':
    # num_nodes = 3
    # edge_prob = .7
    # seed = 3
    #
    # G = nx.gnp_random_graph(n=num_nodes, p=edge_prob, seed=seed, directed=True)
    # dag = nx.DiGraph([(u, v, {}) for (u, v) in G.edges() if u < v])
    #
    # assert nx.is_directed_acyclic_graph(dag)
    # print('dag: ', dag.edges())
    #
    # top_sort_idx = list(nx.topological_sort(dag))
    # variables: Dict[int, Variable] = dict()
    #
    # for var_idx in top_sort_idx:
    #     parents = [variables[idx] for idx in sorted(list(dag.predecessors(var_idx)))]
    #
    #     variable = DiscreteVariable(num_values=3, parents=parents)
    #
    #     variables[var_idx] = variable
    #
    # variables_top_sort = [variables[idx] for idx in top_sort_idx]

    def print_observations(graph: Graph):
        df = graph.sample(num_observations=10)
        print(df)


    print_observations(graph=create_simple_discrete_model1())
    print_observations(graph=create_simple_discrete_model2())

    print_observations(graph=create_simple_continuous_model1())
    print_observations(graph=create_simple_continuous_model2())
