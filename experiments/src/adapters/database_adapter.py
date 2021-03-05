import logging
from dataclasses import dataclass

import psycopg2

from experiments.src.templates.singleton import Singleton
import pandas as pd

@dataclass
class DatabaseConfig:
    host: str
    port: int
    user: str
    password: str
    dbname: str


@Singleton
class DatabaseAdapter:

    def __init__(self):
        self.connection = psycopg2.connect(
            host="localhost", #todo enviroment variables
            port="5432",
            user="admin",
            password="admin",
            dbname="postgres"
        )

    def __del__(self):
        if self.connection:
            self.connection.close()

    def _sql_column_type_string(self, column_name: str, dtype: str) -> str:
        if dtype == "int64":
            return f'"{column_name}" INT'
        if dtype == "float64":
            return f'"{column_name}" FLOAT'
        raise AttributeError(f"dtype {dtype} unknown")

    def upload_dataset(self, table_name: str, data_path: str):
        df = pd.read_csv(data_path)
        sql_columns = ', '.join([self._sql_column_type_string(name, df.dtypes[name]) for name in df.columns])
        create_table_query = f'CREATE TABLE {table_name} ({sql_columns})'

        logging.info('Uploading data to database...')
        with open(data_path, 'r') as data_file:
            cur = self.connection.cursor()
            cur.execute(create_table_query)
            next(data_file)  # Skip the header row.
            cur.copy_from(data_file, table_name, sep=',')
            self.connection.commit()

    def set_error_code_for_job(self, job_id: int):
        update_job_status_error = f"UPDATE job SET status = 'error', error_code = 'UNKNOWN' WHERE id={job_id}"
        cur = self.connection.cursor()
        cur.execute(update_job_status_error)
        self.connection. conn.commit()

        #failed_jobs_file = os.path.join(docker_run_logs_dir, "failed_jobs.log")
        #with open(failed_jobs_file, 'a') as failed_jobs_file_handle:
        #    failed_jobs_file_handle.write(f"{job_id} \n")
