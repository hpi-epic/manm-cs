import hashlib
import logging
import os
from abc import ABC
from dataclasses import dataclass
from uuid import uuid4
from typing import List, Tuple, Callable

import networkx as nx

import experiments.src.adapters.backend_adapter as BackendAdapter
from experiments.src.adapters.database_adapter import DatabaseAdapter
from experiments.src.tree.base_node import BaseNode
from experiments.src.tree.experiment_node import ExperimentConfig, ExperimentNode
from experiments.src.tree.node_type import NodeType

from src.graph import Graph, GraphBuilder
import pandas as pd

from src.utils import write_single_csv


@dataclass
class DatasetConfig:
    num_nodes: int
    edge_density: float
    discrete_node_ratio: float
    min_discrete_value_classes: int
    max_discrete_value_classes: int
    discrete_signal_to_noise_ratio: float
    max_samples: int
    continuous_noise_std: float
    functions: List[Tuple[float, Callable[...,float]]]

@dataclass
class ResolvedDataset:
    dataset_id: int


class DatasetNode(BaseNode):
    type = NodeType.DATASET
    config: DatasetConfig
    
    benchmark_id: str
    data_path: str
    graph_path: str
    table_name: str

    def __init__(self, config: DatasetConfig, parent):
        super(DatasetNode, self).__init__()
        self.parent = parent

        self.config = config
        self.benchmark_id = hashlib.md5(uuid4().__str__().encode()).hexdigest()
        self.data_path = f'data/benchmarking-experiment-{self.benchmark_id}-data.csv'
        self.graph_path = f'graph/benchmarking-experiment-{self.benchmark_id}-graph.gml'
        self.table_name = f'benchmarking_experiment_{self.benchmark_id}_data'

        self.name = f"Dataset {self.benchmark_id}"

    def _generate_graph_with_at_least_one_edge(self) -> Graph:
        max_retries = 100
        for retry_id in range(max_retries):
            graph = GraphBuilder() \
                .with_num_nodes(self.config.num_nodes) \
                .with_edge_density(self.config.edge_density) \
                .with_discrete_node_ratio(self.config.discrete_node_ratio) \
                .with_discrete_signal_to_noise_ratio(self.config.discrete_signal_to_noise_ratio) \
                .with_min_discrete_value_classes(self.config.min_discrete_value_classes) \
                .with_max_discrete_value_classes(self.config.max_discrete_value_classes) \
                .with_continuous_noise_std(self.config.continuous_noise_std) \
                .with_functions(self.config.functions) \
                .build(seed=retry_id)
            nx_graph = graph.to_networkx_graph()
            if nx_graph.edges:
                return graph
        raise Exception(f"Retried {max_retries} times to generate a graph but no edge was generated.")

    def resolve_impl(self):
        # Create folder for dataset and ground truth
        os.makedirs('data', exist_ok=True)
        os.makedirs('graph', exist_ok=True)

        # Create graph
        graph = self._generate_graph_with_at_least_one_edge()
        nx.write_gml(graph.to_networkx_graph(), self.graph_path)

        # Sample from graph
        dfs = graph.sample(num_observations=self.config.max_samples, num_processes=1)
        write_single_csv(dataframes=dfs, target_path=self.data_path)

        # Upload data to database
        DatabaseAdapter.instance().upload_dataset(table_name=self.table_name, data_path=self.data_path)

        dataset = BackendAdapter.BackendAdapter.instance().create_dataset(table_name=self.table_name)
        dataset_id = dataset["id"]

        BackendAdapter.BackendAdapter.instance().upload_ground_truth(dataset_id, self.graph_path)
        logging.error("dataset created")
        return ResolvedDataset(dataset_id=dataset_id)

    def create_experiment(self, experiment_config):
         return ExperimentNode(experiment_config, parent=self)
