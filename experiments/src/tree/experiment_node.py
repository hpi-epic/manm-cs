from dataclasses import dataclass
from typing import Dict

import experiments.src.adapters.backend_adapter as BackendAdapter
from experiments.src.tree.base_node import BaseNode
from experiments.src.tree.job_node import JobNode
from experiments.src.tree.node_type import NodeType


@dataclass
class ExperimentConfig:
    cores: int
    alpha: float
    sampling_factor: float


@dataclass
class ResolvedExperiment:
    experiment_id: int
    response: Dict


class ExperimentNode(BaseNode):
    config: ExperimentConfig
    type = NodeType.EXPERIMENT

    # Information from previous dataset node
    dataset_node = None

    def __init__(self, config: ExperimentConfig, parent):
        super(ExperimentNode, self).__init__()
        self.parent = parent
        self.name = f"Experiment"

        self.config = config

        self.dataset_node = self.get_parent_with_type(NodeType.DATASET)

    def _generate_settings(self):
        discrete_node_ratio = self.dataset_node.config.discrete_node_ratio

        if discrete_node_ratio == 1:
            return {
                "algorithm_id": 1,
                "dataset_id": self.dataset_node.resolved_data.dataset_id,
                'description': f"{self.config.alpha}",
                "name": "pcalg DISCRETE",
                "parameters": {
                    "alpha": self.config.alpha,
                    "cores": self.config.cores,
                    "independence_test": "disCI",
                    "skeleton_method": "stable.fast",
                    "subset_size": -1,
                    "verbose": 0,
                    "sampling_factor": self.config.sampling_factor
                }
            }
        elif discrete_node_ratio == 0:
            return {
                "algorithm_id": 1,
                "dataset_id": self.dataset_node.resolved_data.dataset_id,
                "description": f"{self.config.alpha}",
                "name": "PC GAUSS",
                "parameters": {
                    "alpha": self.config.alpha,
                    "cores": self.config.cores,
                    "independence_test": "gaussCI",
                    "skeleton_method": "stable.fast",
                    "subset_size": -1,
                    "verbose": 0,
                    "sampling_factor": self.config.sampling_factor
                }
            }
        else:
            return {
                'algorithm_id': 3,
                'dataset_id': self.dataset_id,
                'description': f"{self.max_discrete_value_classes} {self.config.alpha}",
                'name': "BNLEARN MI-CG",
                'parameters': {
                    'alpha': self.config.alpha,
                    'cores': self.config.cores,
                    'discrete_limit': self.dataset_node.config.max_discrete_value_classes,
                    'independence_test': "mi-cg",
                    'subset_size': -1,
                    'verbose': 0,
                    "sampling_factor": self.config.sampling_factor
                }
            }

    def resolve_impl(self):
        experiment_settings = self._generate_settings()

        experiment_response = BackendAdapter.BackendAdapter.instance().create_experiment(experiment_settings)
        experiment_id = experiment_response["id"]
        return ResolvedExperiment(experiment_id=experiment_id, response=experiment_response)

    def create_jobs(self, num_jobs: int):
        jobs = []
        for _ in range(num_jobs):
            jobs.append(JobNode(parent=self))
