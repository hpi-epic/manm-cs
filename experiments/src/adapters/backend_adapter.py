import json
import logging

import requests
from requests.adapters import HTTPAdapter, Retry

from experiments.src.templates.singleton import Singleton

@Singleton
class BackendAdapter:
    def __init__(self):
        self.API_HOST = "http://localhost:5000" #todo env
        self.API_EXPERIMENTS = f"{self.API_HOST}/api/experiments"
        self.API_EXPERIMENT_START = lambda id: f"{self.API_HOST}/api/experiment/{id}/start"
        self.API_EXPERIMENT_JOBS = lambda id: f"{self.API_HOST}/api/experiment/{id}/jobs"
        self.API_RESULT_GTCOMPARE = lambda id: f"{self.API_HOST}/api/result/{id}/gtcompare"
        self.API_JOB = lambda id: f"{self.API_HOST}/api/job/{id}"
        self.API_DATASET = lambda id: f"{self.API_HOST}/api/dataset/{id}"
        self.API_DATASETS = f"{self.API_HOST}/api/datasets"
        self.API_DATASET_GROUND_TRUTH = lambda id: f'{self.API_HOST}/api/dataset/{id}/ground-truth'

        self.node = "minikube" #todo

        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["GET", "PUT", "DELETE", "OPTIONS"],
            backoff_factor=10
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.http = requests.Session()
        self.http.mount("http://", adapter)

    def __generic_get(self, url: str):
        response = self.http.get(url)

        if response.status_code != 200:
            error_msg = f"API Request to {url} failed."
            logging.error(error_msg)
            raise Exception(error_msg)

        return response.json()

    def __generic_post(self, url: str, payload=None, files=None):
        response = self.http.post(url, json=payload, files=files)

        if response.status_code != 200:
            error_msg = f"API Request to {url} failed with response {response.content}."
            logging.error(error_msg)
            raise Exception(error_msg)

        return response.json()

    def __generic_delete(self, url: str):
        response = self.http.delete(url)

        if response.status_code != 200:
            error_msg = f"API Request to {url} failed with response {response.content}."
            logging.error(error_msg)
            raise Exception(error_msg)

    def get_ground_truth_compare(self, result_id: int):
        return self.__generic_get(self.API_RESULT_GTCOMPARE(result_id))

    def get_job(self, job_id: int):
        return self.__generic_get(self.API_JOB(job_id))

    def get_jobs(self, experiment_id: int):
        return self.__generic_get(self.API_EXPERIMENT_JOBS(experiment_id))

    def get_experiment_jobs(self, experiment_id: int):
        return self.__generic_get(self.API_EXPERIMENT_JOBS(experiment_id))

    def start_job(self, experiment_id: int):
        json = {
            'experiment_id': f'{experiment_id}',
            'node': f'{self.node}',
            'runs': 1,
            'enforce_cpus': 0
        }
        return self.__generic_post(self.API_EXPERIMENT_START(experiment_id), payload=json)

    def create_experiment(self, config):
        return self.__generic_post(self.API_EXPERIMENTS, payload=config)

    def delete_dataset(self, dataset_id: int):
        return self.__generic_delete(self.API_DATASET(dataset_id))

    def create_dataset(self, table_name: str):
        json_data = {
            'name': f'{table_name}',
            'description': f'Dataset containing the data for '
                           f'{table_name}',
            'load_query': f'SELECT * FROM {table_name}',
            'data_source': 'postgres'
        }
        return self.__generic_post(self.API_DATASETS, payload=json_data)

    def upload_ground_truth(self, dataset_id: int, graph_path: str):
        files = {"graph_file": open(graph_path, "rb")}
        return self.__generic_post(self.API_DATASET_GROUND_TRUTH(dataset_id), files=files)
