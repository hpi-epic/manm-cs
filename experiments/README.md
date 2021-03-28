# Experiment Run
## Dependencies

- Docker
- Running PostgresSQL instance

If you do not have a running PostgreSQL database, you can start one by executing the following docker-compose.yml with ```docker-compose up````

```
version: '3'
services:
  postgres-db:
    image: postgres:12
    environment:
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: admin
    ports:
      - "5432:5432"
```
## Experiment Setup

Before running experiments, we have to setup parts of the core pipeline. We use the backend and the database of MPCI to measure the quality of defined CSL algorithms on generated data.

Execute following commands in the mpci repository to start the backend and the database of MPCI. 

```
cd services/python-images

# Add algorithms to postgresSQL
sh setup_algorithms.py

# Start backend
sh src/run_dev_server.sh
```

# Experiment adjustment

The experiment is defined in ```run.py```. You can adjust which data should be generated for the algorithms in the run method. Each parameter of MPCI-dag is represented as list.

```
num_nodes_list = [5, 10, 15]
edge_density_list = [0.6]  # [0.2, 0.4, 0.6]
discrete_node_ratio_list = [0.0]
continuous_noise_std_list = [0.2]
num_samples_list = [1000, 10000, 10000]
discrete_signal_to_noise_ratio_list = [0.9]
discrete_value_classes_list = [(2, 3)]


dataset_num_samples = 200000 # total samples
num_graphs_per_config = 1
```
The experiment script executes following steps for each permutation of the settings:
- Generate dataset if needed
- Upload dataset and ground truth to database if needed
- Create experiments in database
- Execute experiments
- Collect results from each experiment / job

Results are written to ```job_results.csv```
``` 