from typing import Dict, Optional, Callable, List, Tuple

import networkx as nx
import numpy as np
import random
from validation import validate_int, validate_float, validate_bool, validate_text

from manm_cs.graph import Graph
from manm_cs.noise import GaussianNoiseBuilder, DiscreteNoiseBuilder
from manm_cs.prob_distributions.continuous.bimodal_distribution import BimodalDistribution
from manm_cs.prob_distributions.continuous.gaussian_distribution import GaussianDistribution
from manm_cs.variables.continuous_variable import ContinuousVariable
from manm_cs.variables.discrete_variable import DiscreteVariable
from manm_cs.variables.variable import Variable
from manm_cs.variables.variable_type import VariableType


class GraphBuilder:
    num_nodes: int
    edge_density: float
    discrete_node_ratio: float
    conditional_gaussian: bool
    num_samples: int

    discrete_signal_to_noise_ratio: float
    min_discrete_value_classes: Optional[int] = None
    max_discrete_value_classes: Optional[int] = None
    continuous_noise_std: float
    scale_parents: Optional[bool] = False

    beta_lower_limit: float
    beta_upper_limit: float

    functions: List[Tuple[float, Callable[...,float]]]

    dag: Optional[nx.DiGraph] = None

    def with_graph_structure_file(self, file_name: str) -> 'GraphBuilder':
        validate_text(file_name, pattern='.*\.gml')
        self.graph_structure_file_name = file_name
        return self

    def with_networkx_DiGraph(self, graph: 'DiGraph') -> 'GraphBuilder':
        assert nx.is_directed_acyclic_graph(graph)
        self.with_num_nodes(nx.number_of_nodes(graph))
        self.with_edge_density(nx.density(graph))
        self.dag = graph
        return self

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

    def with_conditional_gaussian(self, conditional_gaussian: bool) -> 'GraphBuilder':
        validate_bool(conditional_gaussian)
        self.conditional_gaussian = conditional_gaussian
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

    def with_scaled_parent_influence(self, scale_parents: bool) -> 'GraphBuilder':
        validate_bool(scale_parents)
        self.scale_parents = scale_parents
        return self

    def with_functions(self, function_tuples: List[Tuple[float, Callable[...,float]]]) -> 'GraphBuilder':
        ### Transform input in form of
        # [(0.5 F1), (0.3 F2), (0.2 F3)]
        # into, to allow drawing later on using random number between 0 and 1
        # [(0.5 F1), (0.8 F2), (1.0 F3)]
        #TODO validation
        self.functions = []
        cur_range = 0.0
        for function_tuple in function_tuples:
            cur_range += function_tuple[0]
            if cur_range > 1.0:
                raise ValueError(f'Ranges of functions should not exceeed 1.0')
            self.functions.append((cur_range,function_tuple[1]))
        if cur_range < 1.0:
            raise ValueError(f'Ranges of functions should sum up to 1.0 but are ' +
                             f'{cur_range}')
        return self

    def with_betas(self, lower_limit: float, upper_limit: float) -> 'GraphBuilder':
        # Switch limits if lower is larger than upper
        if lower_limit > upper_limit:
            tmp = upper_limit
            upper_limit = lower_limit
            lower_limit = tmp
        validate_float(lower_limit, min_value=0.0)
        if lower_limit <= 0.0:
            raise ValueError(f'Range of beta has to be within (0, Inf), ' +
                             f'but is {lower_limit} and {upper_limit}')
        validate_float(upper_limit, min_value=lower_limit)
        self.beta_lower_limit = lower_limit
        self.beta_upper_limit = upper_limit
        return self

    def chose_function(self):
        rand_val = random.random()
        for function_tuple in sorted(self.functions):
            if rand_val <= function_tuple[0]:
                return function_tuple[1]
        ### use last entry as default
        def identical(value):
            return value
        return identical

    def sample_beta(self) -> List[float]:
        if np.random.randint(0,2) == 0:
            # we sample beta from [-upper, -lower]
            return random.uniform(-self.beta_upper_limit, -self.beta_lower_limit)
        else:
            # we sample beta from [lower, upper]
            return random.uniform(self.beta_lower_limit, self.beta_upper_limit)

    def generate_discrete_variable(self, parents, node_idx) -> 'DiscreteVariable':
        num_values = np.random.randint(
            low=self.min_discrete_value_classes,
            high=self.max_discrete_value_classes + 1,
            size=1)[0]
        noise = DiscreteNoiseBuilder() \
            .with_signal_to_noise_ratio(
            signal_to_noise_ratio=self.discrete_signal_to_noise_ratio) \
            .with_num_discrete_values(num_discrete_values=num_values) \
            .build()
        return DiscreteVariable(idx=node_idx, num_values=num_values,
                                parents=parents, noise=noise, scale_parents=self.scale_parents)

    def generate_continuous_variable(self, parents, node_idx) -> 'ContinuousVariable':
        noise = GaussianNoiseBuilder() \
            .with_sigma(sigma=self.continuous_noise_std) \
            .build()
        num_continuous_parents = sum(
            [1 for p in parents if p.type == VariableType.CONTINUOUS])

        functions = [self.chose_function() for p in range(num_continuous_parents)]
        betas = [self.sample_beta() for p in range(num_continuous_parents)]
        return ContinuousVariable(idx=node_idx, parents=parents, functions=functions,
                                  noise=noise, betas=betas, scale_parents=self.scale_parents)

    def generate_dag(self, seed: int) -> 'DiGraph':
        # Generate graph using networkx package
        G = nx.gnp_random_graph(n=self.num_nodes, p=self.edge_density, seed=seed, directed=False)
        # Convert generated graph to DAG
        dag = nx.DiGraph()
        dag.add_nodes_from(G)
        dag.add_edges_from([(u, v, {}) for (u, v) in G.edges() if u < v])
        assert nx.is_directed_acyclic_graph(dag)
        return dag

    def build(self, seed: int = 0) -> Graph:
        self.dag = self.generate_dag(seed) if self.dag is None else self.dag

        # Create list of topologically sorted nodes
        # Note, nodes are ordered already, sorting step may become relevant, if graph generation above is changed
        top_sort_idx = list(nx.topological_sort(self.dag))
        num_discrete_nodes = int(self.discrete_node_ratio * self.num_nodes)

        variables_by_idx: Dict[int, Variable] = {}
        for i, node_idx in enumerate(top_sort_idx):
            parents = [variables_by_idx[idx] for idx in sorted(list(self.dag.predecessors(node_idx)))]

            # Conditional Gaussian:
            if self.conditional_gaussian == True:
                if i < num_discrete_nodes:
                    # Consider the first num_discrete_nodes nodes to be of discrete type
                    variable = self.generate_discrete_variable(parents, node_idx)
                else:
                    variable = self.generate_continuous_variable(parents, node_idx)
            # Mixed:
            else:
                # For each node: decide for discrete or conintuous given probability of self.discrete_node_ratio
                # Discrete:
                if random.random() < self.discrete_node_ratio:
                    variable = self.generate_discrete_variable(parents, node_idx)
                # Continuous:
                else:
                    variable = self.generate_continuous_variable(parents, node_idx)

            variables_by_idx[node_idx] = variable

        variables = [variables_by_idx[idx] for idx in top_sort_idx]
        return Graph(variables=variables)
