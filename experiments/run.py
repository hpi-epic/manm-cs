import contextlib
import hashlib
import logging
import itertools
import os
import threading
import time
from typing import Dict, Tuple, List
from uuid import uuid4
import socketio
from flatten_dict import flatten
import networkx as nx
import pandas as pd
import psycopg2
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import subprocess
import copy

from src.graph.graph_builder import GraphBuilder, Graph
from src.utils import write_single_csv

logging.getLogger().setLevel(logging.INFO)

POSTGRES_HOST = "localhost"
POSTGRES_PORT = "5432"
POSTGRES_USER = "admin"
POSTGRES_PASSWORD = "admin"
POSTGRES_DBNAME = "postgres"

API_HOST = "http://localhost:5000"
API_EXPERIMENTS = f"{API_HOST}/api/experiments"
API_EXPERIMENT_START = lambda id: f"{API_HOST}/api/experiment/{id}/start"
API_EXPERIMENT_JOBS = lambda id: f"{API_HOST}/api/experiment/{id}/jobs"
API_RESULT_GTCOMPARE = lambda id: f"{API_HOST}/api/result/{id}/gtcompare"
API_JOB = lambda id: f"{API_HOST}/api/job/{id}"

CSV__RESULT_OUTPUT = "job_results.csv"

# TODO
ALPHA_VALUES = [0.05]
NUM_JOBS = 10
DOCKER_PROCESS_TIMEOUT_SEC = 30 * 60

ALL_EXPERIMENTS_STARTED = False
RUNNING_JOBS = []
write_file_lock = threading.Lock()

MEASUREMENTS_CONFIGS = {}
MEASUREMENTS = []  # {"config", "experiment_config", "result"}

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

docker_run_logs_dir = "docker_run_logs"
os.makedirs(docker_run_logs_dir, exist_ok=True)

# Setup job listener
sio = socketio.Client()
sio.connect(API_HOST)


def check_job_for_completion(job_id):
    if job_id in RUNNING_JOBS:
        job = get_job(job_id)

        if job["status"] == "done" or job["status"] == "error":
            logging.info(f'Completed job {job_id} with status {job["status"]}')
            row_properties = MEASUREMENTS_CONFIGS[job_id]

            job_result = job["result"]

            if job["status"] != "error":  # or if job_result:
                gd_compare = get_gtcompare(job_result["id"])
                row_properties["result"] = job_result
                row_properties["gd_compare"] = gd_compare
            with write_file_lock:
                if job_id in RUNNING_JOBS:
                    MEASUREMENTS.append(flatten(row_properties, reducer='path'))
                    RUNNING_JOBS.remove(job_id)

                    pd.DataFrame(MEASUREMENTS).to_csv(CSV__RESULT_OUTPUT)

            if ALL_EXPERIMENTS_STARTED and len(RUNNING_JOBS) == 0:
                logging.info("No more jobs are running")
                sio.disconnect()
        else:
            logging.info(f"Job with id {job_id} is not finished, its status is {job['status']}")


def check_all_jobs_for_completion():
    # sometimes the job message from socketio comes in when the job status is not yet done, so we need to recheck this.
    logging.info("Checking all jobs for completion")
    for job_id in RUNNING_JOBS:
        check_job_for_completion(job_id)


@sio.on('job')
def on_job_update(job):
    job_id = job["id"]
    check_job_for_completion(job_id)


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
                                 alpha: float, discrete_node_ratio: float, sampling_factor: float) -> Dict:
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
                "verbose": 0,
                "sampling_factor": sampling_factor
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
                "verbose": 0,
                "sampling_factor": sampling_factor
            }
        }
    else:
        return {
            'algorithm_id': 2,
            'dataset_id': dataset_id,
            'description': f"{max_discrete_value_classes} {alpha}",
            'name': "PCALG with MI-CG",
            "parameters": {
                "alpha": alpha,
                "cores": cores,
                "independence_test": "micg",
                "skeleton_method": "stable.fast",
                'discrete_node_limit': max_discrete_value_classes,
                "subset_size": -1,
                "verbose": 0,
                "sampling_factor": sampling_factor
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


def generate_graph_with_at_least_one_edge(config: dict, seed: int) -> Graph:
    max_retries = 100
    for retry_id in range(seed, seed + max_retries):
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
            .build(seed=retry_id)
        nx_graph = graph.to_networkx_graph()
        if nx_graph.edges:
            return graph
    raise Exception(f"Retried {max_retries} times to generate a graph but no edge was generated Config: {config}")


def generate_data(benchmark_id: str, config: dict, seed: int) -> Tuple[str, str]:
    os.makedirs('data', exist_ok=True)
    data_path = f'data/benchmarking-experiment-{benchmark_id}-data.csv'

    graph = generate_graph_with_at_least_one_edge(config, seed)

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


def sql_column_type_string(column_name: str, dtype: str) -> str:
    if dtype == "int64":
        return f'"{column_name}" INT'
    if dtype == "float64":
        return f'"{column_name}" FLOAT'
    raise AttributeError(f"dtype {dtype} unknown")


def upload_data(benchmark_id: str, data_path: str) -> Tuple[int, str]:
    data_table_name = f'benchmarking_experiment_{benchmark_id}_data'
    df = pd.read_csv(data_path)
    sql_columns = ', '.join([sql_column_type_string(name, df.dtypes[name]) for name in df.columns])
    create_table_query = f'CREATE TABLE {data_table_name} ({sql_columns})'

    logging.info('Uploading data to database...')
    with execute_with_connection() as conn, open(data_path, 'r') as data_file:
        start = time.time()
        cur = conn.cursor()
        cur.execute(create_table_query)
        next(data_file)  # Skip the header row.
        cur.copy_from(data_file, data_table_name, sep=',')
        conn.commit()
        end = time.time()
    logging.info(f'Successfully uploaded data to table {data_table_name} in {end - start}s')

    return data_table_name


def add_experiment(dataset_id: int, max_discrete_value_classes: int, cores: int,
                   discrete_node_ratio: float, sampling_factor: float):
    experiments = []
    for alpha in ALPHA_VALUES:
        experiments += [generate_experiment_settings(
            dataset_id=dataset_id,
            max_discrete_value_classes=max_discrete_value_classes,
            cores=cores,
            alpha=alpha,
            discrete_node_ratio=discrete_node_ratio,
            sampling_factor=sampling_factor
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


def create_dataset(benchmark_id: int, data_table_name: str, graph_path: str) -> int:
    # Create dataset metadata object
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
    res = http.post(url=f'{API_HOST}/api/dataset/{dataset_id}/ground-truth', files=files)
    res.raise_for_status()
    logging.info('Successfully uploaded ground truth')

    return dataset_id


def run_job_in_docker(job: dict, experiment: dict) -> subprocess.Popen:
    log_file = os.path.join(docker_run_logs_dir, f"{job['id']}.log")
    command = ["docker", "run", "--net=host", "mpci/mpci_execution_r", experiment['algorithm']['script_filename'], "-j",
               str(job['id']), "-d", str(experiment['dataset_id']), "--api_host", "localhost:5000", "--send_sepsets",
               "0"]
    for k, v in experiment['parameters'].items():
        command.append('--' + k)
        command.append(str(v))
    with open(log_file, 'w') as log_file_handle:
        log_file_handle.write(" ".join(command))
        log_file_handle.write("\n")
        ls_output = subprocess.Popen(command, stdout=log_file_handle, stderr=log_file_handle)
    return ls_output


def set_error_code_for_job(job_id: int):
    with execute_with_connection() as conn:
        update_job_status_error = f"UPDATE job SET status = 'error', error_code = 'UNKNOWN' WHERE id={job_id}"
        cur = conn.cursor()
        cur.execute(update_job_status_error)
        conn.commit()
    failed_jobs_file = os.path.join(docker_run_logs_dir, "failed_jobs.log")
    with open(failed_jobs_file, 'a') as failed_jobs_file_handle:
        failed_jobs_file_handle.write(f"{job_id} \n")


def rename_file_with_dataset_id(path: str, dataset_id: int) -> None:
    extension = os.path.splitext(path)[1]
    folder, _ = os.path.split(path)
    file = f"{dataset_id}{extension}"
    new_path = os.path.join(folder, file)
    os.rename(path, new_path)


def run_experiments_for_config(config: dict, dataset_id: int, num_samples_list: List[int], dataset_num_samples: int):
    for num_samples in num_samples_list:
        logging.info('Adding experiment...')
        experiments = add_experiment(
            dataset_id=dataset_id,
            max_discrete_value_classes=config['max_discrete_value_classes'],
            discrete_node_ratio=config['discrete_node_ratio'],
            cores=config["cores"],
            sampling_factor=num_samples / dataset_num_samples
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

            # we start NUM_JOBS jobs in parallel, then wait for them to be finished
            docker_subprocess_handles = []
            for job in jobs:
                job_id = job["id"]
                RUNNING_JOBS.append(job_id)
                MEASUREMENTS_CONFIGS[job_id] = {
                    "config": config,
                    "experiment_config": experiments[index]
                }
                subprocess_handle = run_job_in_docker(job, experiments[index])
                docker_subprocess_handles.append((job_id, subprocess_handle))

            for job_id, process in docker_subprocess_handles:
                try:
                    process.communicate(timeout=DOCKER_PROCESS_TIMEOUT_SEC)
                    return_code = process.returncode
                    if return_code != 0:
                        set_error_code_for_job(job_id)

                except subprocess.TimeoutExpired:
                    process.kill()
                    logging.info("The process was killed commandline is {}".format(process.args))
                    set_error_code_for_job(job_id)
            # for job in jobs:
            #     job_id = job["id"]
            #     check_job_for_completion(job_id)
        logging.info('Successfully started all experiments')
        check_all_jobs_for_completion()


def run_with_config(config: dict, num_samples_list: List[int], dataset_num_samples: int, num_graphs_per_config: int):
    assert dataset_num_samples >= max(
        num_samples_list), f"dataset_num_samples {dataset_num_samples} should be >= max {max(num_samples_list)}"

    config["num_samples"] = dataset_num_samples

    seeds = [i * 1000 for i in range(num_graphs_per_config)]

    for seed in seeds:
        benchmark_id = hashlib.md5(uuid4().__str__().encode()).hexdigest()
        data_path, graph_path = generate_data(benchmark_id=benchmark_id, config=config, seed=seed)
        data_table_name = upload_data(benchmark_id=benchmark_id,
                                      data_path=data_path)
        dataset_id = create_dataset(benchmark_id=benchmark_id, data_table_name=data_table_name,
                                    graph_path=graph_path)
        mpci_config = copy.deepcopy(config)
        mpci_config["seed"] = seed
        mpci_config["generator"] = "mpci-dag"
        run_experiments_for_config(mpci_config, dataset_id, num_samples_list, dataset_num_samples)

        # cleanup
        rename_file_with_dataset_id(data_path, dataset_id)
        rename_file_with_dataset_id(graph_path, dataset_id)


def run():
    num_nodes_list = [5, 10, 15]
    edge_density_list = [0.2]
    discrete_node_ratio_list = [0.25, 0.5, 0.75]
    continuous_noise_std_list = [1.0]
    num_samples_list = [100, 1000, 10000, 100000]
    discrete_signal_to_noise_ratio_list = [0.9]
    discrete_value_classes_list = [(3, 4)]
    dataset_num_samples = 200000
    num_graphs_per_config = 5

    variable_params = [
        num_nodes_list,
        edge_density_list,
        discrete_node_ratio_list,
        continuous_noise_std_list,
        discrete_signal_to_noise_ratio_list,
        discrete_value_classes_list
    ]

    for num_nodes, \
            edge_density, \
            discrete_node_ratio, \
            continuous_noise_std, \
            discrete_signal_to_noise_ratio, \
            discrete_value_classes in list(itertools.product(*variable_params)):
        min_discrete_value_classes, max_discrete_value_classes = discrete_value_classes
        config = dict()
        config['num_nodes'] = num_nodes
        config['edge_density'] = edge_density
        config['discrete_node_ratio'] = discrete_node_ratio
        config['discrete_signal_to_noise_ratio'] = discrete_signal_to_noise_ratio
        config['min_discrete_value_classes'] = min_discrete_value_classes
        config['max_discrete_value_classes'] = max_discrete_value_classes
        config['continuous_noise_std'] = continuous_noise_std
        config['continuous_beta_mean'] = 1.0
        config['continuous_beta_std'] = 0.0
        config['cores'] = 1
        config['node'] = "galileo"

        run_with_config(config=config, num_samples_list=num_samples_list, dataset_num_samples=dataset_num_samples,
                        num_graphs_per_config=num_graphs_per_config)


if __name__ == '__main__':
    run()
    ALL_EXPERIMENTS_STARTED = True

    while RUNNING_JOBS:
        logging.info(f"Still waiting for jobs {RUNNING_JOBS}")

        time.sleep(60)
    logging.info(f"All jobs completed")
    sio.disconnect()
