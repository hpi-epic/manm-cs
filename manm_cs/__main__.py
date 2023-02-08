import argparse
import math
from typing import Type, Callable, Optional, Any

import networkx as nx

from manm_cs.graph import Graph, GraphBuilder
from manm_cs.utils import write_single_csv

GROUND_TRUTH_FILE = "ground_truth"
SAMPLES_FILE = "samples"


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

def type_in_range_exclusive(type_: Type, lower_bound: Optional[float], upper_bound: Optional[float]) -> \
        Callable[[str], Any]:
    def assert_float_in_range(x: str):
        try:
            x = type_(x)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Cannot convert {x} to type {type_}")

        if lower_bound is not None and x <= lower_bound:
            raise argparse.ArgumentTypeError(f"{x} not in range ({lower_bound}, {upper_bound})")
        if upper_bound is not None and x > upper_bound:
            raise argparse.ArgumentTypeError(f"{x} not in range ({lower_bound}, {upper_bound})")
        return x

    return assert_float_in_range

def to_bool(bool_str: str):
    return bool(int(bool_str))

def lin_func(x):
    return x

def quad_func(x):
    return math.pow(x,2)

def cube_func(x):
    return math.pow(x,3)

FUNCTION_DICTIONARY = {'linear':lin_func,
                       'quadratic':quad_func,
                       'cubic':cube_func,
                       'tanh':math.tanh,
                       'sin':math.sin,
                       'cos':math.cos}

def funcs(function_str: str):
    try:
        val,func = function_str.split(',')
        return(float(val),FUNCTION_DICTIONARY[func])
    except:
        raise argparse.ArgumentTypeError(f"{function_str} has wrong format or not supported")
        
SCALE_DICTIONARY = {'standard': 'standardize_continous_columns',
                    'normal': 'normalize_continous_columns',
                    'rank': 'rank_transform_columns',
                    'uniform': 'uniform_transform_columns'}

def scale(scale_str: str):
    try:
        invert_op = getattr(Graph, SCALE_DICTIONARY[scale_str], None)
        return invert_op
    except:
        raise argparse.ArgumentTypeError(f"{scale_str} has wrong format or not supported")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate a dataset for benchmarking causal structure learning using '
                    'the mixed additive noise model')
    group1 = parser.add_mutually_exclusive_group(required=True)
    group1.add_argument('--num_nodes', type=type_in_range(int, 1, None),
                        help='Defines the number of nodes to be in the generated DAG.')
    arg_nx_file = group1.add_argument('--graph_structure_file', type=str, required=False,
                    help='valid .gml file to load a fixed graph structure to networkx.DiGraph structure.')
    group2 = parser.add_mutually_exclusive_group(required=True)
    group2.add_argument('--edge_density', type=type_in_range(float, 0.0, 1.0),
                        help='Defines the density of edges in the generated DAG.')
    group2._group_actions.append(arg_nx_file)
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
    parser.add_argument('--num_processes', type=type_in_range(int, 1, None), required=False, default=1,
                        help='Defines the number of processes used to sample data from the created graph.')
    parser.add_argument('--functions', type=funcs, required=False, default=[(1.0,lin_func)], nargs='*',
                        help='Defines the probability and functions for relationships between continuous variables '
                             'probabilities have to sum up to 1, supported functions are '
                             'linear, quadratic, cubic, tanh, sin, cos '
                             'format is probabilityF1,F1 probabilityF2,F2 ... .')
    parser.add_argument('--conditional_gaussian', type=to_bool, required=False, default=True,
                        help='Defines if conditional gaussian model is assumed for a mixture of variables. '
                             'possible values are 0 for False and 1 for True.')
    parser.add_argument('--beta_lower_limit', type=type_in_range_exclusive(float, 0.0, None),
                        required=False, default=0.5,
                        help='Defines the lower limit for beta values used for continuous parents. '
                        'Should be greater than 0 and smaller than upper_limit. Note that we sample from the union of [-upper,-lower] and [lower,upper]')
    parser.add_argument('--beta_upper_limit', type=type_in_range_exclusive(float, 0.0, None),
                        required=False, default=1.0,
                        help='Defines the upper limit for beta values used for continuous parents. '
                        'Should be larger than lower_limit. Note that we sample from the union of [-upper,-lower] and [lower,upper]')
    parser.add_argument('--output_ground_truth_file', type=str, required=False, default=GROUND_TRUTH_FILE,
                    help='Output file (path) for the generated ground truth graph (gml). Relative to the directory from which the library is executed. '
                    'Specify without file extension.')
    parser.add_argument('--output_samples_file', type=str, required=False, default=SAMPLES_FILE,
                    help='Output file (path) for the generated samples csv. Relative to the directory from which the library is executed.'
                    'Specify without file extension.')
    parser.add_argument('--variables_scaling', type=scale, required=False, default=None,
                    help='Scale the continuous variables (‘normal’ or ‘standard’) or all variables (‘rank’ or ‘uniform’) in the dataset once all samples are generated.')
    parser.add_argument('--scale_parents', type=to_bool, required=False, default=False,
                    help='Scale the influence of parents on the variables.')
    args = parser.parse_args()

    assert args.min_discrete_value_classes <= args.max_discrete_value_classes, \
        f"Expected min_discrete_value_classes <= max_discrete_value_classes but got min: " \
        f"{args.min_discrete_value_classes}, max: {args.max_discrete_value_classes} "

    return args


# python src --num_nodes=5 --edge_density=0.6 --discrete_node_ratio=0.0 --num_samples=1000 --discrete_signal_to_noise_ratio=0.0 --min_discrete_value_classes=2 --max_discrete_value_classes=3 --continuous_noise_std=0.2 --functions 0.7,linear 0.3,quadratic --conditional_gaussian 0


def graph_from_args(args) -> Graph:
    if args.graph_structure_file:
        dag = nx.read_gml(args.graph_structure_file)
        assert nx.is_directed_acyclic_graph(dag)
        return GraphBuilder() \
            .with_networkx_DiGraph(dag) \
            .with_discrete_node_ratio(args.discrete_node_ratio) \
            .with_discrete_signal_to_noise_ratio(args.discrete_signal_to_noise_ratio) \
            .with_min_discrete_value_classes(args.min_discrete_value_classes) \
            .with_max_discrete_value_classes(args.max_discrete_value_classes) \
            .with_continuous_noise_std(args.continuous_noise_std) \
            .with_functions(args.functions) \
            .with_conditional_gaussian(args.conditional_gaussian) \
            .with_betas(args.beta_lower_limit, args.beta_upper_limit) \
            .with_scaled_parent_influence(args.scale_parents) \
            .build()
    else:
        return GraphBuilder() \
            .with_num_nodes(args.num_nodes) \
            .with_edge_density(args.edge_density) \
            .with_discrete_node_ratio(args.discrete_node_ratio) \
            .with_discrete_signal_to_noise_ratio(args.discrete_signal_to_noise_ratio) \
            .with_min_discrete_value_classes(args.min_discrete_value_classes) \
            .with_max_discrete_value_classes(args.max_discrete_value_classes) \
            .with_continuous_noise_std(args.continuous_noise_std) \
            .with_functions(args.functions) \
            .with_conditional_gaussian(args.conditional_gaussian) \
            .with_betas(args.beta_lower_limit, args.beta_upper_limit) \
            .with_scaled_parent_influence(args.scale_parents) \
            .build()


if __name__ == '__main__':
    args = parse_args()
    graph = graph_from_args(args)

    dfs = graph.sample(num_observations=args.num_samples, num_processes=args.num_processes)
    if (args.variables_scaling):
        dfs = args.variables_scaling(graph, dataframes=dfs)
        
    write_single_csv(dataframes=dfs, target_path=f"{args.output_samples_file}.csv")

    nx_graph = graph.to_networkx_graph()
    nx.write_gml(nx_graph, f"{args.output_ground_truth_file}.gml")
