import requests, psycopg2, contextlib
import logging
import json
from typing import Dict

from src.graph.graph_builder import GraphBuilder
from src.utils import write_single_csv

API_HOST = "http://vm-mpws2018-proj.eaalab.hpi.uni-potsdam.de"
API_EXPERIMENTS = f"{API_HOST}/api/experiments"
API_EXPERIMENT_START = lambda id: f"{API_HOST}/api/experiment/{id}/start"
API_EXPERIMENT_JOBS = lambda id: f"{API_HOST}/api/experiment/{id}/jobs"
API_RESULT_GTCOMPARE = lambda id :f"{API_HOST}/api/result/{id}/gtcompare"

ALPHA_VALUES = [0.01, 0.05]
NUM_JOBS = 1


def generate_experiment_settings(data_set_id: int, discrete_limit: int, cores: int, alpha: float, discrete_node_ratio: float) -> Dict:
    if discrete_node_ratio == 1:
        return {
            "algorithm_id": 1,
            "dataset_id": data_set_id,
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
            "dataset_id": data_set_id,
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
            'dataset_id': data_set_id,
            'description': f"{discrete_limit} {alpha}",
            'name': "BNLEARN MI-CG",
            'parameters': {
                'alpha': alpha,
                'cores': cores,
                'discrete_limit': discrete_limit,
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
            host="galileo.eaalab.hpi.uni-potsdam.de:5433",
            user="admin",
            password="admin"
        )
        yield conn
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')

def generate_data(num_nodes: int, edge_density: float, discrete_node_ratio: float, \
                  discrete_signal_to_noise_ratio: float, min_discrete_value_classes: int, \
                  max_discrete_value_classes: int, continuous_noise_std: float, \
                  continuous_beta_mean: float, continuous_beta_std: float, num_samples: int) -> str:
    data_path = 'samples.csv'
    graph = GraphBuilder() \
        .with_num_nodes(num_nodes) \
        .with_edge_density(edge_density) \
        .with_discrete_node_ratio(discrete_node_ratio) \
        .with_discrete_signal_to_noise_ratio(discrete_signal_to_noise_ratio) \
        .with_min_discrete_value_classes(min_discrete_value_classes) \
        .with_max_discrete_value_classes(max_discrete_value_classes) \
        .with_continuous_noise_std(continuous_noise_std) \
        .with_continuous_beta_mean(continuous_beta_mean) \
        .with_continuous_beta_std(continuous_beta_std) \
        .build()
    dfs = graph.sample(num_observations=num_samples)
    write_single_csv(dataframes=dfs, target_path=data_path)
    return data_path
    

def upload_data_and_create_dataset(data_path: str):
    with execute_with_connection() as conn:
        print('asdf')


def add_experiment(data_set_id: int, discrete_limit: int, cores: int):
    experiments = []
    for alpha in ALPHA_VALUES:
        experiments += [generate_experiment_settings(
            data_set_id=data_set_id,
            discrete_limit=discrete_limit,
            cores=cores,
            alpha=alpha
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


def delete_dataset_with_data():
    pass


if __name__ == '__main__':
    # data_path = generate_data()
    #upload_data_and_create_dataset(data_path='data_path')
    dataset_id = 27
    discrete_limit = 15
    cores = 16
    node = "galileo"

    experiment_ids = add_experiment(dataset_id, discrete_limit, cores)
    for id in experiment_ids:
        run_experiment(id, node, NUM_JOBS, 0)



