import networkx as nx
import numpy as np

from typing import Dict, Optional
from networkx.algorithms.dag import topological_sort
from validation import validate_int, validate_float
from src.variables.variable import Variable
from src.variables.discrete_variable import DiscreteVariable
from src.noise import DiscreteNoise
from src.graph import Graph

class GraphBuilder:
    num_nodes: int
    edge_density: float
    discrete_node_ratio: float
    num_samples: int

    discrete_signal_to_noise_ratio: float
    min_discrete_value_classes: Optional[int] = None
    max_discrete_value_classes: Optional[int] = None
    continous_noise_standard_deviation: float

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

    def with_discrete_signal_to_noise_ratio(self, discrete_signal_to_noise_ratio: float) -> 'GraphBuilder':
        validate_float(discrete_signal_to_noise_ratio, min_value=0.0, max_value=1.0)
        self.discrete_signal_to_noise_ratio = discrete_signal_to_noise_ratio
        return self
    
    def with_min_discrete_value_classes(self, min_discrete_value_classes: int) -> 'GraphBuilder':
        validate_int(min_discrete_value_classes, min_value=1)
        if self.max_discrete_value_classes is not None \
            and min_discrete_value_classes > self.max_discrete_value_classes:
            raise ValueError(f'Expected min_discrete_value_classes to be smaller ' +
                              'or equal to max_discrete_value_classes, ' +
                              'but are {min_discrete_value_classes} and {self.max_discrete_value_classes}')
        self.min_discrete_value_classes = min_discrete_value_classes
        return self
    
    def with_max_discrete_value_classes(self, max_discrete_value_classes: int) -> 'GraphBuilder':
        validate_int(max_discrete_value_classes, min_value=1)
        if self.min_discrete_value_classes is not None \
            and self.min_discrete_value_classes > max_discrete_value_classes:
            raise ValueError(f'Expected max_discrete_value_classes to be greater ' +
                              'or equal to min_discrete_value_classes, ' +
                              'but are {max_discrete_value_classes} and {self.min_discrete_value_classes}')
        self.max_discrete_value_classes = max_discrete_value_classes
        return self
    
    def with_continous_noise_standard_deviation(self, continous_noise_standard_deviation: float) -> 'GraphBuilder':
        validate_float(continous_noise_standard_deviation, min_value=0.0)
        self.continous_noise_standard_deviation = continous_noise_standard_deviation
        return self

    def build(self, seed: int = 0) -> Graph:
        # Generate graph using networkx package
        G = nx.gnp_random_graph(n=self.num_nodes, p=self.edge_density, seed=seed, directed=True)
        # Convert generated graph to DAG
        dag = nx.DiGraph([(u, v, {}) for (u, v) in G.edges() if u < v])
        assert nx.is_directed_acyclic_graph(dag)

        print('dag: ', dag.edges())

        # Create list of topologically sorted nodes 
        top_sort_idx = list(nx.topological_sort(dag))
        num_discrete_nodes = int(self.discrete_node_ratio * self.num_nodes)

        variables_by_idx: Dict[int, Variable] = {}
        for i, node_idx in enumerate(top_sort_idx):
            parents = [variables_by_idx[idx] for idx in sorted(list(dag.predecessors(node_idx)))]

            if i < num_discrete_nodes:
                # Consider the first num_discrete_nodes nodes to be of discrete type
                num_values = np.random.randint(low=self.min_discrete_value_classes, 
                                               high=self.max_discrete_value_classes, size=1)[0]
                noise = DiscreteNoise.builder() \
                    .with_signal_to_noise_ratio(signal_to_noise_ratio=self.discrete_signal_to_noise_ratio) \
                    .with_num_discrete_values(num_discrete_values=num_values) \
                    .build()
                variable = DiscreteVariable(idx=node_idx, num_values=num_values, parents=parents, noise=noise)
            else:
                raise Exception('Not yet supported')
                # variable = ContinuousVariable(idx=node_idx)

            variables_by_idx[node_idx] = variable

        variables = [variables_by_idx[idx] for idx in top_sort_idx]
        return Graph(variables=variables)
        