import multiprocessing
from typing import List, Optional

import networkx as nx
import pandas as pd
import numpy as np
from pathos.multiprocessing import ProcessingPool

from manm_cs.variables.variable import Variable


class Graph:
    variables: List[Variable]

    def __init__(self, variables: List[Variable]):
        self.variables = variables

    def __get_top_sort_variables(self):
        # TODO: implement function
        return self.variables

    def sample(self, num_observations: int,
               num_processes: Optional[int] = None) -> List[pd.DataFrame]:
        """Return dataframe of size (num_observations, len(variables))

        """

        def fn(chunk_size: int):
            np.random.seed() # reseed thread to ensure independence

            df = pd.DataFrame()
            for variable in self.variables:
                df[variable.idx] = variable.sample(df=df, num_observations=chunk_size)
            return df

        if num_processes is None:
            max_num_processes = max(int(num_observations / 10000), 1)
            num_processes = min(multiprocessing.cpu_count(), max_num_processes)

        pool = ProcessingPool()
        chunk_sizes = [int(num_observations / num_processes) for _ in range(num_processes)]
        chunk_sizes[-1] = num_observations - sum(chunk_sizes[:-1])

        return pool.map(fn, chunk_sizes)

    def to_networkx_graph(self) -> nx.DiGraph:
        nx_graph = nx.DiGraph()
        for var in sorted(self.variables, key=lambda v: v.idx):
            nx_graph.add_node(var.idx)
            for parent in var.parents:
                nx_graph.add_edge(parent.idx, var.idx)
        return nx_graph
