# MANM-CS
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
Data generation module for benchmarking methods for causal structure learning (CSL) from mixed discrete-continuous and nonlinear observational data based upon the mixed additive noise model (MANM).
The related paper ["MANM-CS: Data Generation for Benchmarking Causal Structure Learning from Mixed Discrete-Continuous and Nonlinear Data"](https://why21.causalai.net/papers/WHY21_16.pdf) was published at the NeurIPS-21 Workshop ["Causal Inference & Machine Learning: Why now?"](https://why21.causalai.net/index.html).

## Getting started



### Installation / Usage via pip
```
python3 -m pip install manm-cs
python3 -m manm_cs --num_nodes 10 --edge_density 0.5 --num_samples 10000 --discrete_node_ratio 0.5
```

### Installation from Source

#### Get the code
Start by cloning this repository.
```
git clone git@github.com:hpi-epic/manm-cs.git
cd manm-cs
```
#### Install requirements within venv

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

After the creation of a new virtual environment, we can install the project dependencies defined in [setup.cfg](setup.cfg) for both platforms.

```
python3 -m pip install .
```

#### Execute data generation

You can start the data generation with following command. The generated graph and the dataset are saved as ground_truth.gml and samples.csv in the current working directory. Available parameters for data generation can be seen with ```python3 -m manm_cs --help```.

```
python3 -m manm_cs \
    --num_nodes 10 \
    --edge_density 0.5 \
    --num_samples 10000 \
    --discrete_node_ratio 0.5
```

### Build Release
```
python3 -m pip install --upgrade build twine
python3 -m build

# Upload to testPyPi
# use __token__ as username and the pypi token as password
python3 -m twine upload --repository testpypi dist/*

# Upload to PyPi
python3 -m twine upload dist/*
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
| continuous_noise_std           | \[0, Inf)   | 1.0 | Defines the standard deviation of gaussian noise added to continuous variables. |
| functions           			 | (\[0, 1\], func)   | id | A list of probabilities and mathmatical functions for relationships between two continuous nodes. Note, the input are tuples (probability, function), where the sum of all probabilities has to equal 1. Command line supported functions are: [linear, quadratic, cubic, tanh, sin, cos] |
| num_processes | [1, Inf) | 1 | Number of processes used for data sampling |
| conditional_gaussian | 0 or 1 | 1 | '1' Defines that conditional gaussian model is assumed for a mixture of variables. Otherwise '0', discrete variables can have continuous parents. |
| beta_lower_limit | (0, Inf) | 0.5 | Lower limit for beta values for influence of continuous parents. Betas are sampled uniform from the union of [-upper,-lower] and [lower,upper]. Upper limit see below. |
| beta_upper_limit | (0, Inf) | 1 | Upper limit for beta values for influence of continuous parents. Betas are sampled uniform from the union of [-upper,-lower] and [lower,upper]. Lower limit see above. |
| graph_structure_file |  | None | Defines a path to a .gml file for a fixed DAG structure (ignoring node and edge characteristics) used during manm_cs graph building. Note graph_structure_file is mutually exclusive to num_nodes and edge_density. |
| variables_scaling | {'normal', 'standard', 'rank', 'rank', 'uniform'} | None | 'Scale the continuous variables ('normal' or standard') or all variables ('rank' or 'uniform’) in the dataset once all samples are generated. ||
| scale_parents | 0 or 1 | 0 | Defines if the influence of the parents on a child node is scaled, e.g., the sum of values of the parents is divided by the number of parents for a continuous child node. |


## License

MIT
