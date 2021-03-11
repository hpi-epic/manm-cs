import hashlib
import json
import logging
import os
import time
from datetime import timedelta
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from src.graph import Graph, GraphBuilder
from src.utils import write_single_csv

FOLDER_PATH = "../datasets/"
DATA_EXTENSION = ".csv"
GROUND_TRUTH_EXTENSION = ".gml"

PARAMS = {}
PARAMS['num_nodes'] = [10, 100, 1000, 10000]
PARAMS['edge_density'] = [0.5]
PARAMS['discrete_node_ratio'] = [0.5]
PARAMS['discrete_signal_to_noise_ratio'] = [0.95]
PARAMS['min_discrete_value_classes'] = [3]
PARAMS['max_discrete_value_classes'] = [4]
PARAMS['continuous_noise_std'] = [1.0]
PARAMS['continuous_beta_mean'] = [0.0]
PARAMS['continuous_beta_std'] = [0.5]
PARAMS['num_samples'] = [10000, 100000, 1000000]
PARAMS['num_processes'] = [16, 32, 64]

logging.getLogger().setLevel(logging.INFO)


def graph_from_args(args) -> Graph:
    return GraphBuilder() \
        .with_num_nodes(args['num_nodes']) \
        .with_edge_density(args['edge_density']) \
        .with_discrete_node_ratio(args['discrete_node_ratio']) \
        .with_discrete_signal_to_noise_ratio(args['discrete_signal_to_noise_ratio']) \
        .with_min_discrete_value_classes(args['min_discrete_value_classes']) \
        .with_max_discrete_value_classes(args['max_discrete_value_classes']) \
        .with_continuous_noise_std(args['continuous_noise_std']) \
        .with_continuous_beta_mean(args['continuous_beta_mean']) \
        .with_continuous_beta_std(args['continuous_beta_std']) \
        .build()


def execute_benchmark(args):
    current_measurement = args
    file_name = hashlib.sha224(json.dumps(args, sort_keys=True).encode('utf-8')).hexdigest()
    file_path = f"{FOLDER_PATH}{file_name}"

    # start experiment
    start = time.time()

    try:
        # graph generation
        graph = graph_from_args(args)
        dfs = graph.sample(num_observations=args['num_samples'], num_processes=args['num_processes'])
        end_generation = time.time()

        # write csv file
        data_file_path = file_path + DATA_EXTENSION
        write_single_csv(dataframes=dfs, target_path=data_file_path)
        end_csv = time.time()

        nx_graph = graph.to_networkx_graph()
        nx.write_gml(nx_graph, file_path + GROUND_TRUTH_EXTENSION)

        # end measurement
        end_gt = time.time()

        current_measurement['time_generation'] = timedelta(
            seconds=end_generation - start) / timedelta(milliseconds=1)
        current_measurement['time_csv'] = timedelta(seconds=end_csv - start) / timedelta(
            milliseconds=1)
        current_measurement['time_gt'] = timedelta(seconds=end_gt - start) / timedelta(
            milliseconds=1)
        current_measurement['path_dataset'] = data_file_path
        current_measurement['path_ground_truth'] = file_name + GROUND_TRUTH_EXTENSION
        current_measurement['success'] = True

        if not args['keep_data']:
            os.remove(data_file_path)

    except Exception as e:
        logging.error(e)
        logging.error(current_measurement)

        current_measurement['time_generation'] = None
        current_measurement['time_csv'] = None
        current_measurement['time_gt'] = None
        current_measurement['path_dataset'] = None
        current_measurement['path_ground_truth'] = None
        current_measurement['success'] = False

    return current_measurement


def is_valid(args):
    if args['min_discrete_value_classes'] > args['max_discrete_value_classes']:
        return False

    return True


if __name__ == '__main__':
    # Create folder if needed
    Path(FOLDER_PATH).mkdir(parents=True, exist_ok=True)

    # Calculate number of benchmark iterations
    number_iterations = np.prod([len(PARAMS[key]) for key in PARAMS])
    current_number = 0

    measurements = []
    start_time_global = time.time()

    #for num_nodes, edge_density, discrete_node_ratio, discrete_signal_to_noise_ratio, min_discrete_value_classes, max_discrete_value_classes, continuous_noise_std, continuous_beta_mean, continuous_beta_std, num_samples

    for num_nodes in PARAMS['num_nodes']:
        for edge_density in PARAMS['edge_density']:
            for discrete_node_ratio in PARAMS['discrete_node_ratio']:
                for discrete_signal_to_noise_ratio in PARAMS['discrete_signal_to_noise_ratio']:
                    for min_discrete_value_classes in PARAMS['min_discrete_value_classes']:
                        for max_discrete_value_classes in PARAMS['max_discrete_value_classes']:
                            for continuous_noise_std in PARAMS['continuous_noise_std']:
                                for continuous_beta_mean in PARAMS['continuous_beta_mean']:
                                    for continuous_beta_std in PARAMS['continuous_beta_std']:
                                        for num_samples in PARAMS['num_samples']:
                                            for num_processes in PARAMS['num_processes']:
                                                args = dict()
                                                args['num_nodes'] = num_nodes
                                                args['edge_density'] = edge_density
                                                args['discrete_node_ratio'] = discrete_node_ratio
                                                args[
                                                    'discrete_signal_to_noise_ratio'] = discrete_signal_to_noise_ratio
                                                args[
                                                    'min_discrete_value_classes'] = min_discrete_value_classes
                                                args[
                                                    'max_discrete_value_classes'] = max_discrete_value_classes
                                                args['continuous_noise_std'] = continuous_noise_std
                                                args['continuous_beta_mean'] = continuous_beta_mean
                                                args['continuous_beta_std'] = continuous_beta_std
                                                args['num_samples'] = num_samples
                                                args['num_processes'] = num_processes
                                                args['keep_data'] = False

                                                if is_valid(args):
                                                    measurement = execute_benchmark(args)
                                                    measurements.append(measurement)

                                                    df_measurements = pd.DataFrame(measurements)
                                                    df_measurements.to_csv(FOLDER_PATH + "metadata.csv")
                                                    current_number += 1
                                                    current_time_delta_global = timedelta(
                                                        seconds=time.time() - start_time_global) / timedelta(
                                                        milliseconds=1) / 1000

                                                    avg_time_per_iteration = current_time_delta_global / current_number
                                                    logging.info(
                                                        f" {current_number} / {number_iterations}"
                                                        f" Avg: {round(avg_time_per_iteration, 3)}s"
                                                        f" Est: {round((number_iterations - current_number) * avg_time_per_iteration, 3)}s")

    logging.info("\nFinished all experiments.\n")