import contextlib
import hashlib
import logging
import itertools
import os
import time
from typing import Dict
from typing import Tuple
from uuid import uuid4
from odo import odo

import networkx as nx
import pandas as pd
import psycopg2
import requests
from sqlalchemy import create_engine

from src.graph.graph_builder import GraphBuilder
from src.utils import write_single_csv

logging.getLogger().setLevel(logging.INFO)
engine = create_engine('postgresql+psycopg2://admin:admin@localhost:5433/postgres', echo=False)

API_HOST = "http://vm-mpws2018-proj.eaalab.hpi.uni-potsdam.de"
API_EXPERIMENTS = f"{API_HOST}/api/experiments"
API_EXPERIMENT_START = lambda id: f"{API_HOST}/api/experiment/{id}/start"
API_EXPERIMENT_JOBS = lambda id: f"{API_HOST}/api/experiment/{id}/jobs"
API_RESULT_GTCOMPARE = lambda id: f"{API_HOST}/api/result/{id}/gtcompare"

ALPHA_VALUES = [0.01, 0.05]
NUM_JOBS = 1


def generate_experiment_settings(dataset_id: int, max_discrete_value_classes: int, cores: int,
                                 alpha: float, discrete_node_ratio: float) -> Dict:
    if discrete_node_ratio == 1:
        return {
            "algorithm_id": 1,
            "dataset_id": dataset_id,
            'description': f"{alpha}",
            "name": "pcalg DISCRETE",
            "parameters": {
                "alpha": alpha,
                "cores": cores,
                "independence_test": "disCI",
                "skeleton_method": "stable.fast",
                "subset_size": -1,
                "verbose": 0
            }
        }
    elif discrete_node_ratio == 0:
        return {
            "algorithm_id": 1,
            "dataset_id": dataset_id,
            "description": f"{alpha}",
            "name": "PC GAUSS",
            "parameters": {
                "alpha": alpha,
                "cores": cores,
                "independence_test": "gaussCI",
                "skeleton_method": "stable.fast",
                "subset_size": -1,
                "verbose": 0
            }
        }
    else:
        return {
            'algorithm_id': 3,
            'dataset_id': dataset_id,
            'description': f"{max_discrete_value_classes} {alpha}",
            'name': "BNLEARN MI-CG",
            'parameters': {
                'alpha': alpha,
                'cores': cores,
                'discrete_limit': max_discrete_value_classes,
                'independence_test': "mi-cg",
                'subset_size': -1,
                'verbose': 0
            }
        }


def generate_job_config(node: str, runs: int, enforce_cpus: int):
    return {
        "node": node,
        "runs": runs,
        "parallel": True,
        "enforce_cpus": enforce_cpus
    }


@contextlib.contextmanager
def execute_with_connection():
    conn = None
    try:
        conn = psycopg2.connect(
            host="localhost",
            port="5433",
            user="admin",
            password="admin",
            dbname="postgres"
        )
        yield conn
    except (Exception, psycopg2.DatabaseError) as error:
        logging.error(f'Error occurred while trying to connect to the database. {error}')
    finally:
        if conn is not None:
            conn.close()
            logging.info(f'Database connection closed.')


def generate_data(benchmark_id: str, config: dict) -> Tuple[str, str]:
    os.makedirs('data', exist_ok=True)
    data_path = f'data/benchmarking-experiment-{benchmark_id}-data.csv'

    logging.info('Starting graph builder...')
    graph = GraphBuilder() \
        .with_num_nodes(config['num_nodes']) \
        .with_edge_density(config['edge_density']) \
        .with_discrete_node_ratio(config['discrete_node_ratio']) \
        .with_discrete_signal_to_noise_ratio(config['discrete_signal_to_noise_ratio']) \
        .with_min_discrete_value_classes(config['min_discrete_value_classes']) \
        .with_max_discrete_value_classes(config['max_discrete_value_classes']) \
        .with_continuous_noise_std(config['continuous_noise_std']) \
        .with_continuous_beta_mean(config['continuous_beta_mean']) \
        .with_continuous_beta_std(config['continuous_beta_std']) \
        .build()
    logging.info('Starting graph sampling...')
    dfs = graph.sample(num_observations=config['num_samples'])

    logging.info('Writing samples...')
    write_single_csv(dataframes=dfs, target_path=data_path)
    logging.info(f'Successfully written samples to {data_path}')

    logging.info('Writing graph...')
    os.makedirs('graph', exist_ok=True)
    graph_path = f'graph/benchmarking-experiment-{benchmark_id}-graph.gml'
    nx.write_gml(graph.to_networkx_graph(), graph_path)
    logging.info(f'successfully written graph to {graph_path}')

    return data_path, graph_path


def upload_data_and_create_dataset(benchmark_id: str, data_path: str,
                                   graph_path: str) -> Tuple[int, str]:
    data_table_name = f'benchmarking_experiment_{benchmark_id}_data'
    with engine.begin() as connection:
        # df = pd.read_csv(data_path)
        logging.info('Uploading data to database...')
        with execute_with_connection() as conn:
            start = time.time()
            data_file = open(data_path, 'r')
            cur = conn.cursor()
            cur.copy_from(data_file, sep=',')
            data_file.close()
            # Error with pandas: odo(data_path, f'postgresql://admin:admin@localhost:5433/postgres:{data_table_name}')
            # Too slow: df.to_sql(data_table_name, con=connection, method='multi', index=False, if_exists='replace')
            end = time.time()
        logging.info(f'Successfully uploaded data to table {data_table_name} in {end - start}s')

    json_data = {
        'name': f'benchmarking-experiment-{benchmark_id}',
        'description': f'Dataset containing the data for '
                       f'the benchmarking-experiment with id {benchmark_id}',
        'load_query': f'SELECT * FROM {data_table_name}',
        'data_source': 'postgres'
    }
    logging.info('Creating dataset...')
    res = requests.post(url=f'{API_HOST}/api/datasets', json=json_data)
    res.raise_for_status()
    res = res.json()
    dataset_id = res['id']
    logging.info('Successfully created dataset')

    logging.info('Uploading ground truth...')
    files = {"graph_file": open(graph_path, "rb")}
    res = requests.post(url=f'{API_HOST}/api/dataset/{dataset_id}/upload', files=files)
    res.raise_for_status()
    logging.info('Successfully uploaded ground truth')

    return dataset_id, data_table_name


def add_experiment(dataset_id: int, max_discrete_value_classes: int, cores: int,
                   discrete_node_ratio: float):
    experiments = []
    for alpha in ALPHA_VALUES:
        experiments += [generate_experiment_settings(
            dataset_id=dataset_id,
            max_discrete_value_classes=max_discrete_value_classes,
            cores=cores,
            alpha=alpha,
            discrete_node_ratio=discrete_node_ratio
        )]

    responses = []
    for experiment_payload in experiments:
        response = requests.post(API_EXPERIMENTS, json=experiment_payload)

        if response.status_code != 200:
            error_msg = f"API Request to {API_EXPERIMENTS} with payload={experiment_payload} failed."
            logging.error(error_msg)
            raise Exception(error_msg)

        responses.append(response.json())

    experiment_ids = [response['id'] for response in responses]
    return experiment_ids


def run_experiment(experiment_id: int, node: str, runs: int, enforce_cpus: int):
    job_config = generate_job_config(node, runs, enforce_cpus)
    response = requests.post(API_EXPERIMENT_START(experiment_id), json=job_config)

    if response.status_code != 200:
        error_msg = f"API Request to {API_EXPERIMENT_START(experiment_id)} failed."
        logging.error(error_msg)
        raise Exception(error_msg)


def download_results(experiment_id: int):
    # Get jobs of experiment
    response_jobs = requests.get(API_EXPERIMENT_JOBS(experiment_id))

    if response_jobs.status_code != 200:
        error_msg = f"API Request to {API_EXPERIMENT_JOBS(experiment_id)} failed."
        logging.error(error_msg)
        raise Exception(error_msg)

    jobs = response_jobs.json()
    result_ids = [job["result"]["id"] for job in jobs if job["result"]]

    gtcompares = []
    for id in result_ids:
        response = requests.get(API_RESULT_GTCOMPARE(id))

        if response.status_code != 200:
            error_msg = f"API Request to {API_RESULT_GTCOMPARE(id)} failed."
            logging.error(error_msg)
            raise Exception(error_msg)

        gtcompares.append(response.json())

    # Get result id's of experiment jobs

    pass


def delete_dataset_with_data(table_name: str, dataset_id: str, api_host: id):
    with execute_with_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(f"DROP TABLE {table_name};")
            conn.commit()
    requests.delete(f"{api_host}/api/dataset/{dataset_id}")


def run_with_config(config: dict):
    benchmark_id = hashlib.md5(uuid4().__str__().encode()).hexdigest()
    data_path, graph_path = generate_data(benchmark_id=benchmark_id, config=config)
    dataset_id, data_table_name = upload_data_and_create_dataset(benchmark_id=benchmark_id,
                                                                 data_path=data_path,
                                                                 graph_path=graph_path)

    logging.info('Adding experiment...')
    experiment_ids = add_experiment(
        dataset_id=dataset_id,
        max_discrete_value_classes=config['max_discrete_value_classes'],
        discrete_node_ratio=config['discrete_node_ratio'],
        cores=config["cores"]
    )
    logging.info('Successfully added experiment')

    logging.info('Starting all experiments...')
    for id in experiment_ids:
        run_experiment(
            experiment_id=id,
            node=config["node"],
            runs=NUM_JOBS,
            enforce_cpus=0
        )
    logging.info('Successfully started all experiments')

    # delete_dataset_with_data(table_name=data_table_name, dataset_id=dataset_id, api_host=API_HOST)


if __name__ == '__main__':
    num_nodes_list = [20] # [20, 50, 100]
    edge_density_list = [.5] # [0.2, 0.5, 0.8]
    discrete_node_ratio = [0.0] # , 0.4, 0.6, 1.0]
    variable_params = [num_nodes_list, edge_density_list, discrete_node_ratio]

    for num_nodes, edge_density, discrete_node_ratio in list(itertools.product(*variable_params)):
        config = dict()
        config['num_nodes'] = num_nodes
        config['edge_density'] = edge_density
        config['discrete_node_ratio'] = discrete_node_ratio
        config['discrete_signal_to_noise_ratio'] = 0.8
        config['min_discrete_value_classes'] = 3
        config['max_discrete_value_classes'] = 5
        config['continuous_noise_std'] = 2.0
        config['continuous_beta_mean'] = 3.0
        config['continuous_beta_std'] = 1.0
        config['num_samples'] = 100000
        config['cores'] = 120
        config['node'] = "galileo"

        run_with_config(config=config)
