# HemTracer

A Python package providing convenient CFD post-processing functions for Lagrangian hemolysis estimation.

## Installation

The project has been tested on macOS for `Python 3.12.0` with the packages listed in `requirements.txt`. 

The recommended way to install is to create a virtual environment from the home folder of this repository
```shell
python3.12 -m venv env
source venv/bin/activate
python3.12 -m pip install .
```
This creates a folder `env` that will contain all required packages, ensuring consistent conditions. You can exit the virtual environment by running
```shell
deactivate
```

## Usage
Whenever you want to use `hemtracer`, you need to activate the virtual environment first by running
```shell
source /path/to/hemtracer/venv/bin/activate
```
Then your Python will use the correct environment and you can run
```shell
python your_hemo_analysis_script.py
```

## Code
Check the [code documentation](https://nicodirkes.github.io/HemTracer/) for more information.
