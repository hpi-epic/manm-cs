import argparse

from typing import Type, Callable, Optional, Any

from src.graph.graph import Graph
from src.noise import DiscreteNoise
from src.prob_distributions import BinomialDistribution, CustomDiscreteDistribution
from src.variables.discrete_variable import DiscreteVariable


def create_simple_discrete_model1():
    A = DiscreteVariable(idx=0, num_values=2, noise=DiscreteNoise(prob_distribution=BinomialDistribution(probability=.75)))
    B = DiscreteVariable(idx=1, num_values=3, parents=[A], mapping={(0,): 1, (1,): 2},
                         noise=DiscreteNoise(
                             prob_distribution=CustomDiscreteDistribution(probs=[.5, .2, .3])))
    variables = [A, B]
    return Graph(variables=variables)

def type_in_range(type_: Type, lower_bound: Optional[float], upper_bound: Optional[float]) -> Callable[[str], Any]:
    def assert_float_in_range(x: str):
        try:
            x = type_(x)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Cannot convert {x} to type {type_}")

        if lower_bound is not None and x < lower_bound:
            raise argparse.ArgumentTypeError(f"{x} not in range [{lower_bound}, {upper_bound}]")
        if upper_bound is not None and x > upper_bound:
            raise argparse.ArgumentTypeError(f"{x} not in range [{lower_bound}, {upper_bound}]")
        return x
    return assert_float_in_range


def parse_args():
    parser = argparse.ArgumentParser(description='Generate a dataset for benchmarking causal structure learning using '
                                                 'the additive noise model')
    parser.add_argument('--num_nodes', type=type_in_range(int, 1, None), required=True,
                        help='Defines the number of nodes to be in the generated DAG.')
    parser.add_argument('--edge_density', type=type_in_range(float, 0.0, 1.0), required=True,
                        help='Defines the density of edges in the generated DAG.')
    parser.add_argument('--discrete_node_ratio', type=type_in_range(float, 0.0, 1.0), required=True,
                        help='Defines the percentage of nodes that shall be of discrete type. Depending on its value '
                             'the appropriate model (multivariate normal, mixed gaussian, discrete only) is chosen.')
    parser.add_argument('--num_samples', type=type_in_range(int, 1, None), required=True,
                        help='Defines the number of samples that shall be generated from the DAG.')
    parser.add_argument('--discrete_signal_to_noise_ratio', type=type_in_range(float, 0.0, 1.0), required=True,
                        help='Defines the probability that no noise is added within the additive noise model.')
    parser.add_argument('--min_discrete_value_classes', type=type_in_range(int, 2, None), required=True,
                        help='Defines the minimum number of discrete classes a discrete variable shall have.')
    parser.add_argument('--max_discrete_value_classes', type=type_in_range(int, 2, None), required=True,
                        help='Defines the maximum number of discrete classes a discrete variable shall have.')
    parser.add_argument('--continous_noise_standard_deviation', type=type_in_range(float, 0.0, None), required=True,
                        help='Defines the standard deviation of gaussian noise added to continuous variables.')
    parser.add_argument('--api_host', type=str, required=True,
                        help='Endpoint to upload the results')
    args = parser.parse_args()

    assert args.min_discrete_value_classes < args.max_discrete_value_classes, \
        f"Expected min_discrete_value_classes <= max_discrete_value_classes but got min: " \
        f"{args.min_discrete_value_classes}, max: {args.max_discrete_value_classes} "

    return args


if __name__ == '__main__':
    args = parse_args()
    graph = create_simple_discrete_model1()
    df = graph.sample(num_observations=50)
    df.to_csv("anm_result.csv")
