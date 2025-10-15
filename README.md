# Optimality-Based Control Space Reduction for Infinite-Dimensional Control Spaces

```
# ~~~
# This file is part of the paper:
#   
#           "Optimality-Based Control Space Reduction for Infinite-Dimensional Control Spaces"
# by Michael Kartmann and Stefan Volkwein
# Preprint: TODO
#
# Copyright 2025 all developers. All rights reserved.
# License: Licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
# Authors: Michael Kartmann
# 
# ~~~
```

In this repository, we provide the code for the numerical experiments of the paper "Optimality-Based Control Space Reduction for Infinite-Dimensional Control Spaces" by Michael Kartmann and Stefan Volkwein. A preprint is available [here](todo).

## Setup 

To run the code you need to install the python package FEniCS 2019 in your (local) environment together with SciPy, Numpy and Matplotlib. This can be done using `conda` via
```
conda create -n control_reduction -c conda-forge python=3.9 fenics=2019.1.0 numpy scipy matplotlib

```
or using the provided `environment.yml`-file via
```
conda env create -f environment.yml
```
After installing all the packages, run one of the experiments, e.g. by
```
conda activate control_reduction
python main_rom_opti.py
```


## Organization of the repository

The code consists of the main files

* `main_rom_opti.py`: the main file for experiment in Section 5.3,
* `main_adaptive_opti.py`: the main file for experiment in Section 5.4.

The modeling and discretization of the problem is realized in the following files:

* `discretizer.py`: discretizes the problem to obtain a full-order model (FOM),
* `model.py`: contains the implementation of the full-order or reduced-order model (ROM),
* `reductor.py`: reduces the full-order model to obtain a reduced-order model,

Moreover, the following files contain the code for the adaptive optimization algorithm

* `adaptive_opt.py:` contains the implementation of adaptive POD optimization method (Algorithm 1).

Additionally there are some helper files.

## Contact

If there are questions of any kind, don't hesitate to get in touch with us at <michael.kartmann@uni-konstanz.de>.
