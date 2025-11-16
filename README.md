# Measuring the Extinction Cross Section of Helmholtz Resonators via the Optical Theorem

This repository contains the code accompanying the paper "Measuring the Extinction Cross Section of Helmholtz Resonators via the Optical Theorem."


# Contents

This repository provides examples and code related to the paper, as well as additional demonstrations.
The main workflows are organized as Jupyter notebooks:
1. `1_prepation.ipynb`
    
    Contains tests and examples for Mie scattering calculations.

1. `2_optical_theorem.ipynb`
    
    Demonstrates the use of the optical theorem and analyzes its convergence with respect to measurement distance and source position for both plane and spherical waves.

1. `3_data_processing.ipynb`
    
    Shows examples of solving optimization problems and using optimization routines.
    Includes a two-step approach for Mie scattering and for scattering on Helmholtz resonators.

1. `4_experiment.ipynb`

    The main notebook.
    Applies the two-step approach to experimentally measured fields.


# COMSOL Models
To ensure all steps are reproducible, COMSOL models are provided as Python scripts so users can generate their own models.
The scripts are based on the [MPh](https://github.com/MPh-py/MPh) package and require prior experience working with it.

This repository includes three models:
- `sphere_scattering`
- `helmholtz_pw_scattering`
- `helmholtz_sw_scattering`

Each model has been tested with COMSOL versions `6.0` and `6.2`.

Example notebooks demonstrating the models are located in the `comsol` folder under the corresponding model names.


# Data

All data generated using COMSOL or obtained experimentally is stored in the `data` folder, along with optimized model weights used in the two-step approach:
- `data/helmholtz_experiment`: contains experimental data for the scattering of a spherical wave on a Helmholtz resonator and model weights (used in `4_experiment.ipynb`);
- `data/helmholtz_pw_comsol`: extinction and scattering cross-sections calculated for the scattering of a plane wave on a Helmholtz resonator, along with a `script.py` file for generating the data using COMSOL (used in `4_experiment.ipynb` and `3_data_processing.ipynb`);
- `data/helmholtz_sw_comsol`: contains incident and scattered pressure fields calculated for the scattering of a spherical wave on a Helmholtz resonator, along with a `script.py` file for generating the data using COMSOL (used in `3_data_processing.ipynb`);
- `data/sphere_comsol`: extinction and scattering cross-sections calculated for the scattering of a plane wave on a sphere, along with a `script.py` file for generating them using COMSOL (used in `1_prepation.ipynb`);
- `data/sphere_theory`: contains model weights used for analytically calculated data processing (used in `3_data_processing.ipynb`).

# Installation

Clone the repository:
```
git clone https://github.com/djiboshin/paper-optical-theorem.git
```
Enter the project directory:
```
cd paper-optical-theorem
```

The required Python version is **3.12**.
This repository uses [uv](https://github.com/astral-sh/uv) for dependency management.
Following command creates virtual enviroment with all required packages installed:
```
uv sync
```
If you want to use the [MPh](https://github.com/MPh-py/MPh) package to work with COMSOL examples, install the additional dependencies using:
```
uv sync --extra comsol
```

After synchronization, you can use the created virtual environment as usual.
The last command also installs the local editable packages `comsol`, `mie_utils`, and `processing` located in the `src` folder.
These packages can then be imported in your code, for example:
```
import comsol
import mie_utils
import processing
```


# Testing

To verify that your installation is working correctly or to run all tests, use:
```
uv run pytest
```

# Contributing

To install all development-related dependencies, use:
```
uv sync --extra dev
```
or, if you also want to include COMSOL-related packages:
```
uv sync --extra dev --extra comsol
```

The repo uses `black` for formatting.
`flake8` is recommended for code linting.
Before committing, format all files, run tests, and clear Jupyter notebook metadata:
```
uv run black .
uv run pytest --cov src --cov-report html
uv run jupyter nbconvert --ClearMetadataPreprocessor.enabled=True --to notebook --inplace **/*.ipynb
```
