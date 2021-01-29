import contextlib
import hashlib
import json
import logging
import os
from typing import Tuple
from uuid import uuid4

import pandas as pd
import psycopg2
import requests
from sqlalchemy import create_engine

from src.graph.graph_builder import GraphBuilder
from src.utils import write_single_csv

API_HOST = 'http://vm-mpws2018-proj.eaalab.hpi.uni-potsdam.de'

logging.getLogger().setLevel(logging.INFO)

engine = create_engine('postgresql+psycopg2://admin:admin@localhost:5431/postgres', echo=False)


@contextlib.contextmanager
def execute_with_connection():
    conn = None
    try:
        conn = psycopg2.connect(
            host="localhost",
            port="5431",
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


def generate_data(benchmark_id: str, config: dict) -> str:
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

    logging.info(f'Writing samples...')
    write_single_csv(dataframes=dfs, target_path=data_path)
    logging.info(f'Successfully written samples to {data_path}')

    return data_path


def upload_data_and_create_dataset(benchmark_id: str, data_path: str) -> Tuple[str, str]:
    data_table_name = f'benchmarking_experiment_{benchmark_id}_data'
    with engine.begin() as connection:
        df = pd.read_csv(data_path)
        logging.info('Uploading data to database...')
        df.to_sql(data_table_name, con=connection, index=False, if_exists="fail")
        logging.info(f'Successfully uploaded data to table {data_table_name}')

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
    logging.info('Successfully created dataset')

    return res['id'], data_table_name


def add_experiment():
    pass


def run_experiment():
    pass


def download_results():
    pass


def delete_dataset_with_data(table_name: str, dataset_id: str, api_host: id):
    with execute_with_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(f"DROP TABLE {table_name};")
            conn.commit()
    requests.delete(f"{api_host}/api/dataset/{dataset_id}")


def run_with_config(config: dict):
    benchmark_id = hashlib.md5(uuid4().__str__().encode()).hexdigest()
    data_path = generate_data(benchmark_id=benchmark_id, config=config)
    dataset_id, data_table_name = upload_data_and_create_dataset(benchmark_id=benchmark_id,
                                                                 data_path=data_path)

    # delete_dataset_with_data(table_name=data_table_name, dataset_id=dataset_id, api_host=API_HOST)


if __name__ == '__main__':
    config = dict()
    config['num_nodes'] = 20
    config['edge_density'] = 0.8
    config['discrete_node_ratio'] = 1.0 # 0.4
    config['discrete_signal_to_noise_ratio'] = 0.5
    config['min_discrete_value_classes'] = 5
    config['max_discrete_value_classes'] = 10
    config['continuous_noise_std'] = 2.0
    config['continuous_beta_mean'] = 3.0
    config['continuous_beta_std'] = 1.0
    config['num_samples'] = 10000

    run_with_config(config=config)
