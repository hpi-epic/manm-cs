from typing import List

import pandas as pd

from src.variables.variable import Variable


class Graph:
    variables: List[Variable]

    def __init__(self, variables: List[Variable]):
        self.variables = variables

    def __get_top_sort_variables(self):
        # TODO: implement function
        return self.variables

    def sample(self, num_observations: int) -> pd.DataFrame:
        """Return dataframe of size (num_observations, len(variables))

        """
        df = pd.DataFrame()
        for variable in self.variables:
            df[variable.idx] = variable.sample(df=df, num_observations=num_observations)
        return df
