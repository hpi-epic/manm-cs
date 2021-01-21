from src.graph.graph import Graph
from src.noise import DiscreteNoise, ContinuousNoise
from src.prob_distributions import BinomialDistribution, UniformDiscreteDistribution, \
    CustomDiscreteDistribution, GaussianDistribution
from src.variables.continuous_variable import ContinuousVariable
from src.variables.discrete_variable import DiscreteVariable


def create_simple_discrete_model1():
    A = DiscreteVariable(idx=0, num_values=2, noise=DiscreteNoise(prob_distribution=BinomialDistribution(probability=.75)))
    B = DiscreteVariable(idx=1, num_values=3, parents=[A], mapping={(0,): 1, (1,): 2},
                         noise=DiscreteNoise(
                             prob_distribution=CustomDiscreteDistribution(probs=[.5, .2, .3])))
    variables = [A, B]
    return Graph(variables=variables)


def create_simple_discrete_model2():
    A = DiscreteVariable(idx=0, num_values=2, noise=DiscreteNoise(prob_distribution=BinomialDistribution(probability=.25)))
    B = DiscreteVariable(idx=1, num_values=3, noise=DiscreteNoise(UniformDiscreteDistribution(num_values=3)))
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
                             prob_distribution=CustomDiscreteDistribution(probs=[.5, .2, .3])))
    variables = [A, B, C]
    return Graph(variables=variables)


def create_simple_continuous_model1():
    A = ContinuousVariable(idx=0, noise=ContinuousNoise(prob_distribution=GaussianDistribution(mu=0, sigma=1)))
    B = ContinuousVariable(idx=1, parents=[A], betas=[2],
                           noise=ContinuousNoise(
                               prob_distribution=GaussianDistribution(mu=0, sigma=1)))
    variables = [A, B]
    return Graph(variables=variables)


def create_simple_continuous_model2():
    A = ContinuousVariable(idx=0, noise=ContinuousNoise(prob_distribution=GaussianDistribution(mu=0, sigma=1)))
    B = ContinuousVariable(idx=1, noise=ContinuousNoise(prob_distribution=GaussianDistribution(mu=10, sigma=5)))
    C = ContinuousVariable(idx=2, parents=[A, B], betas=[.5, 1.1],
                           noise=ContinuousNoise(
                               prob_distribution=GaussianDistribution(mu=0, sigma=1)))
    variables = [A, B, C]
    return Graph(variables=variables)

def create_simple_conditional_model1():
    A = DiscreteVariable(idx=0, num_values=2, noise=DiscreteNoise(prob_distribution=BinomialDistribution(probability=.25)))
    B = ContinuousVariable(idx=1, parents=[A], mapping={(0,): 2, (1,): 10},
                           noise=ContinuousNoise(
                               prob_distribution=GaussianDistribution(mu=0, sigma=1)))
    variables = [A, B]
    return Graph(variables=variables)

def create_simple_conditional_model2():
    A = DiscreteVariable(idx=0, num_values=2, noise=DiscreteNoise(prob_distribution=BinomialDistribution(probability=.25)))
    B = ContinuousVariable(idx=1, noise=ContinuousNoise(prob_distribution=GaussianDistribution(mu=0, sigma=1)))
    C = ContinuousVariable(idx=2, parents=[A, B], betas=[2], mapping={(0,): 2, (1,): 10},
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
        print()


    print('Exemplary discrete models:')
    print_observations(graph=create_simple_discrete_model1())
    print_observations(graph=create_simple_discrete_model2())

    print('\nExemplary continuous models:')
    print_observations(graph=create_simple_continuous_model1())
    print_observations(graph=create_simple_continuous_model2())

    print('\nExemplary conditional models:')
    print_observations(graph=create_simple_conditional_model1())
    print_observations(graph=create_simple_conditional_model2())
