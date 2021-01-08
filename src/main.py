from src.prob_distributions.binomial_distribution import BinomialDistribution
from src.graph.graph import Graph
from src.noise.discrete_noise import DiscreteNoise
from src.variables.discrete_root_variable import DiscreteRootVariable
from src.variables.discrete_variable import DiscreteVariable

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

    A = DiscreteRootVariable(idx=0, prob_distribution=BinomialDistribution(probability=.75))
    B = DiscreteVariable(idx=1, num_values=3, parents=[A],
                         noise=DiscreteNoise(prob_distribution=[.5, .2, .3, .5]))
    variables = [A, B]

    graph = Graph(variables=variables)

    df = graph.sample(num_observations=5)
    print(df)
