import itertools
from anytree import Node, RenderTree
from pathos.multiprocessing import ProcessingPool

from experiments.src.tree.dataset_node import DatasetConfig, DatasetNode
from experiments.src.tree.experiment_node import ExperimentConfig
from experiments.src.tree.root_node import RootNode

if __name__ == '__main__':

    # Graph settings
    num_nodes_list: [int] = [5, 10, 20, 50]
    edge_density_list: [float] = [0.2, 0.4, 0.6]
    discrete_node_ratio_list: [float] = [0.0, 0.4, 0.6, 1.0]
    discrete_class_ranges: [(int, int)] = [(2, 3)]
    max_samples: int = 200

    # Experiment Settings
    num_samples_list: [int] = [100]
    alphas: [float] = [0.05]
    num_jobs: int = 5
    cores: int = 16

    dataset_variables = [num_nodes_list, edge_density_list, discrete_node_ratio_list, discrete_class_ranges]
    experiment_variables = [num_samples_list, alphas]

    root_node = RootNode()
    #### TODO ADD function list
    for num_nodes, edge_density, discrete_node_ratio, discrete_class_range in list(itertools.product(*dataset_variables)):
        dataset_node = DatasetNode(DatasetConfig(
            num_nodes=num_nodes,
            edge_density=edge_density,
            discrete_node_ratio=discrete_node_ratio,
            min_discrete_value_classes=discrete_class_range[0],
            max_discrete_value_classes=discrete_class_range[1],
            discrete_signal_to_noise_ratio=0.95,
            continuous_noise_std=0.2,
            max_samples=max_samples
        ), parent=root_node)

        for num_samples, alpha in list(itertools.product(*experiment_variables)):
            experiment_node = dataset_node.create_experiment(ExperimentConfig(
                sampling_factor=num_samples / max_samples,
                alpha=alpha,
                cores=cores
            ))
            experiment_node.create_jobs(num_jobs)

    for pre, fill, node in RenderTree(root_node):
        print("%s%s" % (pre, node.name))

    from multiprocessing import Pool, Queue

    q = Queue()
    def worker(node):
        try:
            node.resolve()
        except:
            pass
        print("worker closed")

    pool = ProcessingPool()
    pool.map(worker, root_node.leaves)  # devides id_list between 2 processes, defined above

