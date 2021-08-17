import argparse
from typing import Type, Callable, Optional, Any

import networkx as nx

from src.graph import Graph, GraphBuilder
from src.utils import write_single_csv

GROUND_TRUTH_FILE = "ground_truth.gml"
SAMPLES_FILE = "samples.csv"


def type_in_range(type_: Type, lower_bound: Optional[float], upper_bound: Optional[float]) -> \
        Callable[[str], Any]:
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
    parser = argparse.ArgumentParser(
        description='Generate a dataset for benchmarking causal structure learning using '
                    'the mixed additive noise model')
    parser.add_argument('--num_nodes', type=type_in_range(int, 1, None), required=True,
                        help='Defines the number of nodes to be in the generated DAG.')
    parser.add_argument('--edge_density', type=type_in_range(float, 0.0, 1.0), required=True,
                        help='Defines the density of edges in the generated DAG.')
    parser.add_argument('--discrete_node_ratio', type=type_in_range(float, 0.0, 1.0), required=True,
                        help='Defines the percentage of nodes that shall be of discrete type. Depending on its value '
                             'the appropriate model (multivariate normal, mixed gaussian, discrete only) is chosen.')
    parser.add_argument('--num_samples', type=type_in_range(int, 1, None), required=True,
                        help='Defines the number of samples that shall be generated from the DAG.')
    parser.add_argument('--discrete_signal_to_noise_ratio', type=type_in_range(float, 0.0, 1.0),
                        required=False, default=0.9,
                        help='Defines the probability that no noise is added within the mixed additive noise model.')
    parser.add_argument('--min_discrete_value_classes', type=type_in_range(int, 2, None), default=3, 
                        required=False,
                        help='Defines the minimum number of discrete classes a discrete variable shall have.')
    parser.add_argument('--max_discrete_value_classes', type=type_in_range(int, 2, None), default=4,
                        required=False,
                        help='Defines the maximum number of discrete classes a discrete variable shall have.')
    parser.add_argument('--continuous_noise_std', type=type_in_range(float, 0.0, None),
                        required=False, default=1.0,
                        help='Defines the standard deviation of gaussian noise added to continuous variables.')
    parser.add_argument('--continuous_beta_mean', type=type_in_range(float, None, None),
                        required=False, default=1.0,
                        help='Defines the mean of the beta values (edge weights) for continuous parent nodes.')
    parser.add_argument('--continuous_beta_std', type=type_in_range(float, 0.0, None),
                        required=False, default=0.0,
                        help='Defines the standard deviation of the beta values (edge weights) for continuous parent '
                             'nodes.')
    parser.add_argument('--num_processes', type=type_in_range(int, 1, None), required=False, default=1,
                        help='Defines the number of processes used to sample data from the created graph.')
    args = parser.parse_args()

    assert args.min_discrete_value_classes <= args.max_discrete_value_classes, \
        f"Expected min_discrete_value_classes <= max_discrete_value_classes but got min: " \
        f"{args.min_discrete_value_classes}, max: {args.max_discrete_value_classes} "

    return args


# python src --num_nodes=5 --edge_density=0.6 --discrete_node_ratio=0.0 --num_samples=1000 --discrete_signal_to_noise_ratio=0.0 --min_discrete_value_classes=2 --max_discrete_value_classes=3 --continuous_noise_std=0.2 --continuous_beta_mean=1.0 --continuous_beta_std=0.0


def graph_from_args(args) -> Graph:
    return GraphBuilder() \
        .with_num_nodes(args.num_nodes) \
        .with_edge_density(args.edge_density) \
        .with_discrete_node_ratio(args.discrete_node_ratio) \
        .with_discrete_signal_to_noise_ratio(args.discrete_signal_to_noise_ratio) \
        .with_min_discrete_value_classes(args.min_discrete_value_classes) \
        .with_max_discrete_value_classes(args.max_discrete_value_classes) \
        .with_continuous_noise_std(args.continuous_noise_std) \
        .with_continuous_beta_mean(args.continuous_beta_mean) \
        .with_continuous_beta_std(args.continuous_beta_std) \
        .build()


if __name__ == '__main__':
    args = parse_args()
    graph = graph_from_args(args)
    dfs = graph.sample(num_observations=args.num_samples, num_processes=args.num_processes)
    write_single_csv(dataframes=dfs, target_path=SAMPLES_FILE)

    nx_graph = graph.to_networkx_graph()
    nx.write_gml(nx_graph, GROUND_TRUTH_FILE)
