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
Please make sure you have Python 3 installed.
We recommend installing the requirements defined in [requirements.txt](requirements.txt) using [venv](https://docs.python.org/3/library/venv.html).
```
# Create a virtual environment
python -m venv venv

# Activate your virtual environment
source venv/bin/activate

# Install all requirements
python -m pip install -r requirements.txt
```

### Start data generation
We defined several exemplary models. You can generate observations from them by executing the following command
```
python -m src
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

