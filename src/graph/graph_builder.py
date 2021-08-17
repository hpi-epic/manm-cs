from typing import Dict, Optional

import networkx as nx
import numpy as np
from validation import validate_int, validate_float

from src.graph import Graph
from src.noise import GaussianNoiseBuilder, DiscreteNoiseBuilder
from src.prob_distributions.continuous.bimodal_distribution import BimodalDistribution
from src.prob_distributions.continuous.gaussian_distribution import GaussianDistribution
from src.variables.continuous_variable import ContinuousVariable
from src.variables.discrete_variable import DiscreteVariable
from src.variables.variable import Variable
from src.variables.variable_type import VariableType


class GraphBuilder:
    num_nodes: int
    edge_density: float
    discrete_node_ratio: float
    num_samples: int

    discrete_signal_to_noise_ratio: float
    min_discrete_value_classes: Optional[int] = None
    max_discrete_value_classes: Optional[int] = None
    continuous_noise_std: float
    continuous_beta_mean: float
    continuous_beta_std: float

    def with_num_nodes(self, num_nodes: int) -> 'GraphBuilder':
        validate_int(num_nodes, min_value=1)
        self.num_nodes = num_nodes
        return self

    def with_edge_density(self, edge_density: float) -> 'GraphBuilder':
        validate_float(edge_density, min_value=0.0, max_value=1.0)
        self.edge_density = edge_density
        return self

    def with_discrete_node_ratio(self, discrete_node_ratio: float) -> 'GraphBuilder':
        validate_float(discrete_node_ratio, min_value=0.0, max_value=1.0)
        self.discrete_node_ratio = discrete_node_ratio
        return self

    def with_num_samples(self, num_samples: int) -> 'GraphBuilder':
        validate_int(num_samples, min_value=1)
        self.num_samples = num_samples
        return self

    def with_discrete_signal_to_noise_ratio(self,
                                            discrete_signal_to_noise_ratio: float) -> 'GraphBuilder':
        validate_float(discrete_signal_to_noise_ratio, min_value=0.0, max_value=1.0)
        self.discrete_signal_to_noise_ratio = discrete_signal_to_noise_ratio
        return self

    def with_min_discrete_value_classes(self, min_discrete_value_classes: int) -> 'GraphBuilder':
        validate_int(min_discrete_value_classes, min_value=1)
        if self.max_discrete_value_classes is not None \
                and min_discrete_value_classes > self.max_discrete_value_classes:
            raise ValueError(f'Expected min_discrete_value_classes to be smaller ' +
                             'or equal to max_discrete_value_classes, but are '
                             f'{min_discrete_value_classes} and {self.max_discrete_value_classes}')
        self.min_discrete_value_classes = min_discrete_value_classes
        return self

    def with_max_discrete_value_classes(self, max_discrete_value_classes: int) -> 'GraphBuilder':
        validate_int(max_discrete_value_classes, min_value=1)
        if self.min_discrete_value_classes is not None \
                and self.min_discrete_value_classes > max_discrete_value_classes:
            raise ValueError(f'Expected max_discrete_value_classes to be greater ' +
                             'or equal to min_discrete_value_classes, but are ' +
                             f'{max_discrete_value_classes} and {self.min_discrete_value_classes}')
        self.max_discrete_value_classes = max_discrete_value_classes
        return self

    def with_continuous_noise_std(self, continuous_noise_std: float) -> 'GraphBuilder':
        validate_float(continuous_noise_std, min_value=0.0)
        self.continuous_noise_std = continuous_noise_std
        return self

    def with_continuous_beta_mean(self, continuous_beta_mean: float) -> 'GraphBuilder':
        validate_float(continuous_beta_mean)
        self.continuous_beta_mean = continuous_beta_mean
        return self

    def with_continuous_beta_std(self, continuous_beta_std: float) -> 'GraphBuilder':
        validate_float(continuous_beta_std, min_value=0.0)
        self.continuous_beta_std = continuous_beta_std
        return self

    def build(self, seed: int = 0) -> Graph:
        # Generate graph using networkx package
        G = nx.gnp_random_graph(n=self.num_nodes, p=self.edge_density, seed=seed, directed=True)
        # Convert generated graph to DAG
        dag = nx.DiGraph()
        dag.add_nodes_from(G)
        dag.add_edges_from([(u, v, {}) for (u, v) in G.edges() if u < v])
        assert nx.is_directed_acyclic_graph(dag)

        # Create list of topologically sorted nodes
        # Note, nodes are ordered already, sorting step may become relevant, if graph generation above is changed
        top_sort_idx = list(nx.topological_sort(dag))
        num_discrete_nodes = int(self.discrete_node_ratio * self.num_nodes)

        variables_by_idx: Dict[int, Variable] = {}
        for i, node_idx in enumerate(top_sort_idx):
            parents = [variables_by_idx[idx] for idx in sorted(list(dag.predecessors(node_idx)))]

            if i < num_discrete_nodes:
                # Consider the first num_discrete_nodes nodes to be of discrete type
                num_values = np.random.randint(
                    low=self.min_discrete_value_classes,
                    high=self.max_discrete_value_classes + 1,
                    size=1)[0]
                noise = DiscreteNoiseBuilder() \
                    .with_signal_to_noise_ratio(
                    signal_to_noise_ratio=self.discrete_signal_to_noise_ratio) \
                    .with_num_discrete_values(num_discrete_values=num_values) \
                    .build()
                variable = DiscreteVariable(idx=node_idx, num_values=num_values,
                                            parents=parents, noise=noise)
            else:
                noise = GaussianNoiseBuilder() \
                    .with_sigma(sigma=self.continuous_noise_std) \
                    .build()
                num_continuous_parents = sum(
                    [1 for p in parents if p.type == VariableType.CONTINUOUS])

                # Note: For experiments betas have been fixed to 1, uncomment and comment the following
                # betas = list(np.ones(num_continuous_parents))
                betas_dist1 = GaussianDistribution(mu=self.continuous_beta_mean,
                                                   sigma=self.continuous_beta_std)
                betas_dist2 = GaussianDistribution(mu=-1 * self.continuous_beta_mean,
                                                   sigma=self.continuous_beta_std)
                betas = BimodalDistribution(prob_dist1=betas_dist1, prob_dist2=betas_dist2) \
                    .sample(num_observations=num_continuous_parents)

                variable = ContinuousVariable(idx=node_idx, parents=parents, betas=betas,
                                              noise=noise)

            variables_by_idx[node_idx] = variable

        variables = [variables_by_idx[idx] for idx in top_sort_idx]
        return Graph(variables=variables)
