import argparse
import sys

import networkx as nx
import requests

from typing import Type, Callable, Optional, Any

from src.graph import Graph, GraphBuilder
from src.noise import DiscreteNoise, ContinuousNoise
from src.prob_distributions import BinomialDistribution, UniformDiscreteDistribution, \
    CustomDiscreteDistribution, GaussianDistribution
from src.variables.continuous_variable import ContinuousVariable
from src.variables.discrete_variable import DiscreteVariable

GROUND_TRUTH_FILE = "ground_truth.gml"
SAMPLES_FILE = "samples.csv"


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
    parser.add_argument('--continuous_noise_std', type=type_in_range(float, 0.0, None), required=True,
                        help='Defines the standard deviation of gaussian noise added to continuous variables.')
    parser.add_argument('--continuous_beta_mean', type=type_in_range(float, None, None), required=True,
                        help='Defines the mean of the beta values (edge weights) for continuous parent nodes.')
    parser.add_argument('--continuous_beta_std', type=type_in_range(float, 0.0, None), required=True,
                        help='Defines the standard deviation of the beta values (edge weights) for continuous parent '
                             'nodes.')
    parser.add_argument('--uploadEndpoint', type=str, required=True,
                        help='Endpoint to upload the dataset')
    parser.add_argument('--apiHost', type=str, required=True,
                        help='Url of backend')
    args = parser.parse_args()

    assert args.min_discrete_value_classes < args.max_discrete_value_classes, \
        f"Expected min_discrete_value_classes <= max_discrete_value_classes but got min: " \
        f"{args.min_discrete_value_classes}, max: {args.max_discrete_value_classes} "

    return args


def upload_results(dataset_upload_url: str, api_host: str):
    samples_files = {"file": open(SAMPLES_FILE, "rb")}
    response = requests.put(url=dataset_upload_url, files=samples_files)
    response.raise_for_status()
    json = response.json()
    print(json)
    assert "id" in json, f"id was not found in json {json}"
    dataset_id = json["id"]

    ground_truth_upload_url = f"http://{api_host}/api/dataset/{dataset_id}/upload"
    ground_truth_files = {"graph_file": open(GROUND_TRUTH_FILE, "rb")}
    response = requests.post(url=ground_truth_upload_url, files=ground_truth_files)
    print(response.json())
    response.raise_for_status()


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
    df = graph.sample(num_observations=args.num_samples)
    df.to_csv(SAMPLES_FILE)
    nx_graph = graph.to_networkx_graph()
    nx.write_gml(nx_graph, GROUND_TRUTH_FILE)
    upload_results(args.uploadEndpoint, args.apiHost)


