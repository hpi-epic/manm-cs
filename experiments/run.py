import contextlib
import psycopg2

from src.graph.graph_builder import GraphBuilder
from src.utils import write_single_csv


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


def add_experiment():
    pass


def run_experiment():
    pass


def download_results():
    pass


def delete_dataset_with_data():
    pass


if __name__ == '__main__':
    # data_path = generate_data()
    upload_data_and_create_dataset(data_path='data_path')
