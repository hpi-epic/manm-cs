from validation import validate_int, validate_float

class GraphBuilder:
    num_nodes: int
    edge_density: float
    discrete_node_ratio: float
    num_samples: int

    discrete_signal_to_noise_ratio: float
    min_discrete_value_classes: int
    max_discrete_value_classes: int
    continous_noise_standard_deviation: float

    def with_num_nodes(self, num_nodes: int) -> 'GraphBuilder':
        validate_int(num_nodes, min_value=1)
        self.num_nodes = num_nodes
        return self

    def with_edge_density(self, edge_density: int) -> 'GraphBuilder':
        validate_float(edge_density, min_value=0, max_value=1)
        self.edge_density = edge_density
        return self

    def with_discrete_node_ratio(self, discrete_node_ratio: float) -> 'GraphBuilder':
        validate_float(discrete_node_ratio, min_value=0, max_value=1)
        self.discrete_node_ratio = discrete_node_ratio
        return self

    def with_num_samples(self, num_samples: int) -> 'GraphBuilder':
        validate_int(num_samples, min_value=1)
        self.num_samples = num_samples
        return self

    def with_discrete_signal_to_noise_ratio(self, discrete_signal_to_noise_ratio: float) -> 'GraphBuilder':
        validate_float(discrete_signal_to_noise_ratio, min_value=0, max_value=1)
        self.discrete_signal_to_noise_ratio = discrete_signal_to_noise_ratio
        return self
    
    def with_min_discrete_value_classes(self, min_discrete_value_classes: int) -> 'GraphBuilder':
        validate_int(min_discrete_value_classes, min_value=1)
        if self.max_discrete_value_classes is not None 
            and min_discrete_value_classes > self.max_discrete_value_classes:
            raise ValueError(f'Expected min_discrete_value_classes to be smaller ' +
                              'or equal to max_discrete_value_classes, ' +
                              'but are {min_discrete_value_classes} and {self.max_discrete_value_classes}')
        self.min_discrete_value_classes = min_discrete_value_classes
        return self
    
    def with_min_discrete_value_classes(self, max_discrete_value_classes: int) -> 'GraphBuilder':
        validate_int(max_discrete_value_classes, min_value=1)
        if self.min_discrete_value_classes is not None 
            and self.min_discrete_value_classes > max_discrete_value_classes:
            raise ValueError(f'Expected max_discrete_value_classes to be greater ' +
                              'or equal to min_discrete_value_classes, ' +
                              'but are {max_discrete_value_classes} and {self.min_discrete_value_classes}')
        self.max_discrete_value_classes = max_discrete_value_classes
        return self
    
    def with_continous_noise_standard_deviation(self, continous_noise_standard_deviation: float) -> 'GraphBuilder':
        validate_float(continous_noise_standard_deviation, min_value=0)
        self.continous_noise_standard_deviation = continous_noise_standard_deviation
        return self

    def build(self):
        # TODO: Implement graph generation
        pass