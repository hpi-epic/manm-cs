# mpci-dag
Data generation module for MPCI

## Getting started

### Get the code
Start by cloning this repository and switching to the correct branch.
```
git clone git@github.com:hpi-epic/mpci-dag.git
cd mpci-dag
git checkout feature/data-generation-anm
```
### Install requirements within venv

https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/

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

You can start the data generation with following command. The generated graph and the dataset are saved as ground_truth.gml and samples.csv in the current working directory. All available parameters for data generation can be seen with ```python3 -m src --help```

```
python3 -m src \
    --num_nodes 10 \
    --edge_density 0.5 \
    --num_samples 10000 \
    --discrete_node_ratio 0.5 \
    --discrete_signal_to_noise_ratio 0.5 \
    --min_discrete_value_classes 3 \
    --max_discrete_value_classes 4 \
    --continuous_noise_std 1 \
    --continuous_beta_mean 1 \
    --continuous_beta_std 0
```

## Parameters

| name                           | Value Range | Description |
| ------------------------------ | ----------- |  --- |
| num_nodes                      | \[1, Inf)   | Defines the number of nodes to be in the generated DAG. |
| edge_density                   | \[0, 1\]    | Defines the density of edges in the generated DAG.  |
| discrete_node_ratio            | \[0, 1\]    | Defines the percentage of nodes that shall be of discrete type. Depending on its value the appropriate model (multivariate normal, mixed gaussian, discrete only) is chosen. |
| num_samples                    | \[1, Inf)   | Defines the number of samples that shall be generated from the DAG. |
| discrete_signal_to_noise_ratio | \[0, 1\]    | Defines the probability that no noise is added within the additive noise model. |
| min_discrete_value_classes     | \[2, Inf)  | Defines the minimum number of discrete classes a discrete variable shall have. |
| max_discrete_value_classes     | \[2, Inf)  | Defines the maximum number of discrete classes a discrete variable shall have. |
| continuous_beta_mean            | (-Inf, Inf) | Defines the mean of the beta values (edge weights) for continuous parent nodes. |
| continuous_beta_std             | \[0, Inf)   | Defines the standard deviation of the beta values (edge weights) for continuous parent nodes. |
| continuous_noise_std            | \[0, Inf)   | Defines the standard deviation of gaussian noise added to continuous variables. |

