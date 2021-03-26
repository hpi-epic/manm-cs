from src.graph import Graph, GraphBuilder
from src.noise import DiscreteNoise, ContinuousNoise
from src.prob_distributions import BinomialDistribution, UniformDiscreteDistribution, \
    CustomDiscreteDistribution, GaussianDistribution
from src.utils import write_single_csv
from src.variables.continuous_variable import ContinuousVariable
from src.variables.discrete_variable import DiscreteVariable


def create_simple_discrete_model1():
    A = DiscreteVariable(idx=0, num_values=2, noise=DiscreteNoise(
        prob_distribution=BinomialDistribution(probability=.75)))
    B = DiscreteVariable(idx=1, num_values=3, parents=[A],
                         noise=DiscreteNoise(
                             prob_distribution=CustomDiscreteDistribution(probs=[.5, .2, .3])))
    variables = [A, B]
    return Graph(variables=variables)


def create_simple_discrete_model2():
    A = DiscreteVariable(idx=0, num_values=2, noise=DiscreteNoise(
        prob_distribution=BinomialDistribution(probability=.25)))
    B = DiscreteVariable(idx=1, num_values=3,
                         noise=DiscreteNoise(UniformDiscreteDistribution(num_values=3)))
    C = DiscreteVariable(idx=2, num_values=3, parents=[A, B],
                         noise=DiscreteNoise(
                             prob_distribution=CustomDiscreteDistribution(probs=[.5, .2, .3])))
    variables = [A, B, C]
    return Graph(variables=variables)


def create_simple_continuous_model1():
    A = ContinuousVariable(idx=0, noise=ContinuousNoise(
        prob_distribution=GaussianDistribution(mu=0, sigma=1)))
    B = ContinuousVariable(idx=1, parents=[A], betas=[2],
                           noise=ContinuousNoise(
                               prob_distribution=GaussianDistribution(mu=0, sigma=1)))
    variables = [A, B]
    return Graph(variables=variables)


def create_simple_continuous_model2():
    A = ContinuousVariable(idx=0, noise=ContinuousNoise(
        prob_distribution=GaussianDistribution(mu=0, sigma=1)))
    B = ContinuousVariable(idx=1, noise=ContinuousNoise(
        prob_distribution=GaussianDistribution(mu=10, sigma=5)))
    C = ContinuousVariable(idx=2, parents=[A, B], betas=[.5, 1.1],
                           noise=ContinuousNoise(
                               prob_distribution=GaussianDistribution(mu=0, sigma=1)))
    variables = [A, B, C]
    return Graph(variables=variables)


def create_simple_conditional_model1():
    A = DiscreteVariable(idx=0, num_values=2, noise=DiscreteNoise(
        prob_distribution=BinomialDistribution(probability=.25)))
    B = ContinuousVariable(idx=1, parents=[A], mapping={(0,): 2, (1,): 10},
                           noise=ContinuousNoise(
                               prob_distribution=GaussianDistribution(mu=0, sigma=1)))
    variables = [A, B]
    return Graph(variables=variables)


def create_simple_conditional_model2():
    A = DiscreteVariable(idx=0, num_values=2, noise=DiscreteNoise(
        prob_distribution=BinomialDistribution(probability=.25)))
    B = ContinuousVariable(idx=1, noise=ContinuousNoise(
        prob_distribution=GaussianDistribution(mu=0, sigma=1)))
    C = ContinuousVariable(idx=2, parents=[A, B], betas=[2], mapping={(0,): 2, (1,): 10},
                           noise=ContinuousNoise(
                               prob_distribution=GaussianDistribution(mu=0, sigma=1)))
    variables = [A, B, C]
    return Graph(variables=variables)


if __name__ == '__main__':
    graph = GraphBuilder() \
        .with_num_nodes(1) \
        .with_edge_density(0.001) \
        .with_discrete_node_ratio(0.0001) \
        .with_discrete_signal_to_noise_ratio(0.9) \
        .with_min_discrete_value_classes(3) \
        .with_max_discrete_value_classes(20) \
        .with_continuous_noise_std(1.0) \
        .with_continuous_beta_mean(0.0001) \
        .with_continuous_beta_std(0.0001) \
        .build(seed=42)
    nx_graph = graph.to_networkx_graph()
    dfs = graph.sample(num_observations=200000)

    write_single_csv(dataframes=dfs, target_path="/home/jonas/Code/mpci-dag/src/mehra_mpci.csv")
    print(nx_graph.nodes)
