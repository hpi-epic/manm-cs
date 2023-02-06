import multiprocessing
from typing import List, Optional

import networkx as nx
import pandas as pd
import numpy as np
from scipy.stats import rankdata
from pathos.multiprocessing import ProcessingPool


from manm_cs.variables.variable import Variable, VariableType


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

    def standardize_continous_columns(self, dataframes: List[pd.DataFrame]) -> List[pd.DataFrame]:
        # Standardize continuous variables to have mean=0, and variance=1, i.e, X = (X - X.mean)/(X.std)

        # merge df in dataframes
        merged_df = pd.concat(dataframes, axis=1)
        for variable in self.variables:
            if variable.type == VariableType.CONTINUOUS:
                merged_df[variable.idx] =  (merged_df[variable.idx] - merged_df[variable.idx].mean())/ merged_df[variable.idx].std()

        return [merged_df]
    
    def normalize_continous_columns(self, dataframes: List[pd.DataFrame]) -> List[pd.DataFrame]:
        # Normalize continuous variables to take valus in [0,1], i.e, X = (X - X.min)/(X.max - X.min)
        
        # merge df in dataframes
        merged_df = pd.concat(dataframes, axis=1)
        for variable in self.variables:
            if variable.type == VariableType.CONTINUOUS:
                merged_df[variable.idx] =  (merged_df[variable.idx] - merged_df[variable.idx].min())/ (merged_df[variable.idx].max() - merged_df[variable.idx].min())

        return [merged_df]
    
    def rank_transform_columns(self, dataframes: List[pd.DataFrame]) -> List[pd.DataFrame]:
        # Rank transform all variables while preserving ties, i.e., for x_i return rank(x_i) (note: rank(x_i)=rank(x_j) if x_i=x_j such that discrete points are preserved.)

        # merge df in dataframes
        merged_df = pd.concat(dataframes, axis=1)
        for variable in self.variables:
            merged_df[variable.idx] = rankdata(merged_df[variable.idx], method='dense', axis=0).astype(np.float32)

        return [merged_df]
    
    def uniform_transform_columns(self, dataframes: List[pd.DataFrame]) -> List[pd.DataFrame]:
        # Unfiform all variables to take values in [0,1] with equal marginal distances while preserving ties for discrete variables, i.e., first step rank transform, second step normalize.

        # merge df in dataframes
        merged_df = pd.concat(dataframes, axis=1)
        for variable in self.variables:
            ranked_idx = rankdata(merged_df[variable.idx], method='dense', axis=0).astype(np.float32)
            merged_df[variable.idx] =  (ranked_idx - ranked_idx.max())/ (ranked_idx.max() - ranked_idx.min())

        return [merged_df]


