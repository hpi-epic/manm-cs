import pandas as pd

from typing import List

def write_single_csv(dataframes: List[pd.DataFrame], target_path: str):
    dataframes[0].to_csv(target_path)
    for df in dataframes[1:]:
        df.to_csv(target_path, mode='a', header=False)

def write_single_hdf(dataframes: List[pd.DataFrame], target_path: str):
    dataframes[0].to_hdf(target_path, key='generated_dataset', format='table')
    for df in dataframes[1:]:
        df.to_hdf(target_path, "generated_dataset", format='table', append=True)
