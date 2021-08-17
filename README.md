# MANM-CS
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
Data generation module for benchmarking methods for causal structure learning (CSL) from heterogeneous observational data based upon the mixed additive noise model (MANM).


## Getting started

### Get the code
Start by cloning this repository and switching to the correct branch.
```
git clone git@github.com:hpi-epic/manm-cs.git
cd manm-cs 
git checkout feature/data-generation-anm
```
### Install requirements within venv

Please make sure you have Python 3 installed. We tested the execution of our data generation with Python 3.9.
We recommend installing the requirements defined in [requirements.txt](requirements.txt) using [virtualenv](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

MacOS / Linux

```
# Install virtualenv 
python3 -m pip install --user virtualenv

# Create a new virtual environment
python3 -m venv env

# Activate the virtual environment
source env/bin/activate
```

Windows

```
# Install virtualenv 
py -m pip install --user virtualenv

# Create a new virtual environment
py -m venv env

# Activate the virtual environment
.\env\Scripts\activate
```

After the creation of a new virtual enviroment, we can install the project dependencies defined in [requirements.txt](requirements.txt) for both platforms.

```
python3 -m pip install -r requirements.txt 
```

### Execute data generation

You can start the data generation with following command. The generated graph and the dataset are saved as ground_truth.gml and samples.csv in the current working directory. Available parameters for data generation can be seen with ```python3 -m src --help```.

```
python3 -m src \
    --num_nodes 10 \
    --edge_density 0.5 \
    --num_samples 10000 \
    --discrete_node_ratio 0.5
```

## Parameters

| name                           | Value Range | Default | Description |
| ------------------------------ | ----------- |----|  --- |
| num_nodes                      | \[1, Inf)   | None | Defines the number of nodes to be in the generated DAG. |
| edge_density                   | \[0, 1\]    | None | Defines the density of edges in the generated DAG.  |
| discrete_node_ratio            | \[0, 1\]    | None | Defines the percentage of nodes that shall be of discrete type. Depending on its value the appropriate model (multivariate normal, mixed gaussian, discrete only) is chosen. |
| num_samples                    | \[1, Inf)   | None | Defines the number of samples that shall be generated from the DAG. |
| discrete_signal_to_noise_ratio | \[0, 1\]    | 0.9 | Defines the probability that no noise is added within the mixed additive noise model. |
| min_discrete_value_classes     | \[2, Inf)  | 3 | Defines the minimum number of discrete classes a discrete variable shall have. |
| max_discrete_value_classes     | \[2, Inf)  | 4 | Defines the maximum number of discrete classes a discrete variable shall have. |
| continuous_beta_mean            | (-Inf, Inf) | 1.0 | Defines the mean of the beta values (edge weights) for continuous parent nodes. |
| continuous_beta_std             | \[0, Inf)   | 0.0 | Defines the standard deviation of the beta values (edge weights) for continuous parent nodes. |
| continuous_noise_std            | \[0, Inf)   | 1.0 | Defines the standard deviation of gaussian noise added to continuous variables. |
| num_processes | [1, Inf) | 1 | Number of processes used for data sampling |

## License

MIT
