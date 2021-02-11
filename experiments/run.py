import contextlib
import hashlib
import logging
import itertools
import os
import time
from typing import Dict
from typing import Tuple
from uuid import uuid4
import socketio
from flatten_dict import flatten
import networkx as nx
import pandas as pd
import psycopg2
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from src.graph.graph_builder import GraphBuilder
from src.utils import write_single_csv
import queue

logging.getLogger().setLevel(logging.INFO)

POSTGRES_HOST = "localhost"
POSTGRES_PORT = "5433"
POSTGRES_USER = "admin"
POSTGRES_PASSWORD = "admin"
POSTGRES_DBNAME = "postgres"

API_HOST = "http://vm-mpws2018-proj.eaalab.hpi.uni-potsdam.de"
API_EXPERIMENTS = f"{API_HOST}/api/experiments"
API_EXPERIMENT_START = lambda id: f"{API_HOST}/api/experiment/{id}/start"
API_EXPERIMENT_JOBS = lambda id: f"{API_HOST}/api/experiment/{id}/jobs"
API_RESULT_GTCOMPARE = lambda id: f"{API_HOST}/api/result/{id}/gtcompare"
API_JOB = lambda id: f"{API_HOST}/api/job/{id}"

CSV_RESULT_OUTPUT = "job_results.csv"


# TODO
ALPHA_VALUES = [0.01, 0.05]
NUM_JOBS = 1

RUNNING_JOBS = []

CONFIG_QUEUE = queue.Queue()
MEASUREMENTS_CONFIGS = {}
MEASUREMENTS = [] # {"config", "experiment_config", "result"}

# Setup retry strategy
retry_strategy = Retry(
    total=3,
    status_forcelist=[429, 500, 502, 503, 504],
    method_whitelist=["GET", "PUT", "DELETE", "OPTIONS"],
    backoff_factor=10
)
adapter = HTTPAdapter(max_retries=retry_strategy)
http = requests.Session()
http.mount("http://", adapter)

# Setup job listener
sio = socketio.Client()
sio.connect(API_HOST)

@sio.on('job')
def on_job_update(job):

    job_id = job["id"]
    if job_id in RUNNING_JOBS:
        job = get_job(job_id)

        if job["status"] == "done" or job["status"] == "error":
            logging.info(f'Completed job {job_id} with status {job["status"]}')
            row_properties = MEASUREMENTS_CONFIGS[job_id]

            if job["status"] == "done":
                job_result = job["result"]

                gd_compare = get_gtcompare(job_result["id"])
                row_properties["gd_compare"] = gd_compare

            row_properties["result"] = job_result

            if job_id in RUNNING_JOBS:
                MEASUREMENTS.append(flatten(row_properties, reducer='path'))
                RUNNING_JOBS.remove(job_id)

                pd.DataFrame(MEASUREMENTS).to_csv(CSV_RESULT_OUTPUT)

                #Start next
                run_with_config(config=CONFIG_QUEUE.get())

            if CONFIG_QUEUE.empty():
                logging.info("No more jobs are running")
                sio.disconnect()


def get_job(job_id: int):
    response_job = http.get(API_JOB(job_id))

    if response_job.status_code != 200:
        error_msg = f"API Request to {API_JOB(job_id)} failed."
        logging.error(error_msg)
        raise Exception(error_msg)

    return response_job.json()

def get_gtcompare(result_id: int):
    gt_compare = http.get(API_RESULT_GTCOMPARE(result_id))

    if gt_compare.status_code != 200:
        error_msg = f"API Request to {API_RESULT_GTCOMPARE(result_id)} failed."
        logging.error(error_msg)
        raise Exception(error_msg)

    return gt_compare.json()


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
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            dbname=POSTGRES_DBNAME
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


def sql_column_type_string(column_name:str, dtype: str) -> str:
    if dtype == "int64":
        return f'"{column_name}" INT'
    if dtype == "float64":
        return f'"{column_name}" FLOAT'
    raise AttributeError(f"dtype {dtype} unknown")


def upload_data_and_create_dataset(benchmark_id: str, data_path: str,
                                   graph_path: str) -> Tuple[int, str]:
    data_table_name = f'benchmarking_experiment_{benchmark_id}_data'
    df = pd.read_csv(data_path)
    sql_columns = ', '.join([sql_column_type_string(name, df.dtypes[name]) for name in df.columns])
    create_table_query = f'CREATE TABLE {data_table_name} ({sql_columns})'

    logging.info('Uploading data to database...')
    with execute_with_connection() as conn, open(data_path, 'r') as data_file:
        start = time.time()
        cur = conn.cursor()
        cur.execute(create_table_query)
        next(data_file) # Skip the header row.
        cur.copy_from(data_file, data_table_name, sep=',')
        conn.commit()
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
    res = http.post(url=f'{API_HOST}/api/datasets', json=json_data)
    res.raise_for_status()
    res = res.json()
    dataset_id = res['id']
    logging.info('Successfully created dataset')

    logging.info('Uploading ground truth...')
    files = {"graph_file": open(graph_path, "rb")}
    res = http.post(url=f'{API_HOST}/api/dataset/{dataset_id}/upload', files=files)
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
        response = http.post(API_EXPERIMENTS, json=experiment_payload)

        if response.status_code != 200:
            error_msg = f"API Request to {API_EXPERIMENTS} with payload={experiment_payload} failed with response {response.content}."
            logging.error(error_msg)
            raise Exception(error_msg)

        responses.append(response.json())

    return responses


def run_experiment(experiment_id: int, node: str, runs: int, enforce_cpus: int):
    job_config = generate_job_config(node, runs, enforce_cpus)
    response = http.post(API_EXPERIMENT_START(experiment_id), json=job_config)

    if response.status_code != 200:
        error_msg = f"API Request to {API_EXPERIMENT_START(experiment_id)} failed with response {response.content}."
        logging.error(error_msg)
        raise Exception(error_msg)


def get_jobs(experiment_id: int):
    response_jobs = http.get(API_EXPERIMENT_JOBS(experiment_id))

    if response_jobs.status_code != 200:
        error_msg = f"API Request to {API_EXPERIMENT_JOBS(experiment_id)} failed."
        logging.error(error_msg)
        raise Exception(error_msg)

    return response_jobs.json()


def delete_dataset_with_data(table_name: str, dataset_id: str, api_host: id):
    with execute_with_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(f"DROP TABLE {table_name};")
            conn.commit()
    http.delete(f"{api_host}/api/dataset/{dataset_id}")


def run_with_config(config: dict):
    benchmark_id = hashlib.md5(uuid4().__str__().encode()).hexdigest()
    data_path, graph_path = generate_data(benchmark_id=benchmark_id, config=config)
    dataset_id, data_table_name = upload_data_and_create_dataset(benchmark_id=benchmark_id,
                                                                 data_path=data_path,
                                                                 graph_path=graph_path)

    logging.info('Adding experiment...')
    experiments = add_experiment(
        dataset_id=dataset_id,
        max_discrete_value_classes=config['max_discrete_value_classes'],
        discrete_node_ratio=config['discrete_node_ratio'],
        cores=config["cores"]
    )
    logging.info('Successfully added experiment')

    logging.info('Starting all experiments...')
    experiment_ids = [experiment["id"] for experiment in experiments]

    for index, experiment_id in enumerate(experiment_ids):
        run_experiment(
            experiment_id=experiment_id,
            node=config["node"],
            runs=NUM_JOBS,
            enforce_cpus=0
        )
        logging.info(f'Getting jobs for experiment {experiment_id}')

        jobs = get_jobs(experiment_id)

        for job in jobs:
            job_id = job["id"]
            RUNNING_JOBS.append(job_id)
            MEASUREMENTS_CONFIGS[job_id] = {
                "config": config,
                "experiment_config": experiments[index]
            }
    logging.info('Successfully started all experiments')

    # delete_dataset_with_data(table_name=data_table_name, dataset_id=dataset_id, api_host=API_HOST)

def should_continue(num_nodes: int, edge_density: float, discrete_node_ratio: float, num_samples: int):
    if num_nodes < 50:
        return True
    elif num_nodes == 50:
        if edge_density < 0.6:
            return True
        elif edge_density == 0.6:
            if discrete_node_ratio < 0.4:
                return True
            elif discrete_node_ratio == 0.4:
                if num_samples <= 7500:
                    return True

    return False


if __name__ == '__main__':
    num_nodes_list = [100, 200]
    edge_density_list = [0.2, 0.4, 0.6]
    discrete_node_ratio_list = [0.0, 0.4, 0.6, 1.0]
    num_samples_list = [1000, 2500, 5000, 7500, 10000]
    variable_params = [num_nodes_list, edge_density_list, discrete_node_ratio_list, num_samples_list]

    for num_nodes, edge_density, discrete_node_ratio, num_samples in list(itertools.product(*variable_params)):

        config = dict()
        config['num_nodes'] = num_nodes
        config['edge_density'] = edge_density
        config['discrete_node_ratio'] = discrete_node_ratio
        config['discrete_signal_to_noise_ratio'] = 0.9
        config['min_discrete_value_classes'] = 2
        config['max_discrete_value_classes'] = 3
        config['continuous_noise_std'] = 0.2
        config['continuous_beta_mean'] = 1.0
        config['continuous_beta_std'] = 0.0
        config['num_samples'] = num_samples
        config['cores'] = 120
        config['node'] = "galileo"

        CONFIG_QUEUE.put(config)

    # start first config:
    run_with_config(config=CONFIG_QUEUE.get())

