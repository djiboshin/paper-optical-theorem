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
The scripts are based on the[MPh](https://github.com/MPh-py/MPh) package and require prior experience working with it.
This repository includes two models:
- `sphere_scattering`
- `helmholtz_scattering`

Example notebooks demonstrating both models are located in the `comsol` folder.


# Data

All data generated using COMSOL or obtained experimentally is stored in the `data` folder.


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
This repository uses [uv](https://github.com/astral-sh/uv) for dependency management:
```
uv sync
```
If you want to use the [MPh](https://github.com/MPh-py/MPh) package to work with COMSOL examples, install the additional dependencies using:
```
uv sync --extra comsol
```

After synchronization, you can use the created virtual environment as usual.

