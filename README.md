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
We recommend installing the requirements defined in `requirements.txt` using `venv`.
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