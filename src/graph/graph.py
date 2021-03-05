import multiprocessing
from typing import List, Optional

import networkx as nx
import pandas as pd
from pathos.multiprocessing import ProcessingPool

from src.variables.variable import Variable


class Graph:
    variables: List[Variable]

    def __init__(self, variables: List[Variable]):
        self.variables = variables

    def __get_top_sort_variables(self):
        # TODO: implement function
        return self.variables

    def sample(self, num_observations: int) -> List[pd.DataFrame]:
        """Return dataframe of size (num_observations, len(variables))

        """
        df = pd.DataFrame()
        for variable in self.variables:
            df[variable.idx] = variable.sample(df=df, num_observations=num_observations)

        return [df]

    def to_networkx_graph(self) -> nx.DiGraph:
        nx_graph = nx.DiGraph()
        for var in sorted(self.variables, key=lambda v: v.idx):
            nx_graph.add_node(var.idx)
            for parent in var.parents:
                nx_graph.add_edge(parent.idx, var.idx)
        return nx_graph
