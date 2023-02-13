# Applefy: Robust Detection Limits for High-Contrast Imaging
![Python 3.8 | 3.9](https://img.shields.io/badge/python-3.8_|_3.9-blue)
[![Documentation Status](https://readthedocs.org/projects/applefy/badge/?version=latest)](https://applefy.readthedocs.io/en/latest/?badge=latest)
---

This is the documentation of ``applefy``, a Python package for calculating 
detection limits for exoplanet high contrast imaging (HCI) datasets. 
``applefy`` provides a number of features and functionalities to improve the accuracy
and robustness of contrast curve calculations. It implements the classical 
approach based on the t-test (compare 
[Mawet et al. 2014](https://arxiv.org/abs/1407.2247>)) as well as the parametric
boostrap test for non-Gaussian residual noise (Bonse et al. subm.).

Applefy has two main feature:

1. Compute contrast curves (see [documentation](https://applefy.readthedocs.io/en/latest/02_user_documentation/01_contrast_curves.html))
2. Compute contrast grids (see [documentation](https://applefy.readthedocs.io/en/latest/02_user_documentation/02_contrast_grid.html))

Further, this repository contains the code needed to reproduce the results of 
our paper:
> COMING SOON

---

## Documentation
A full documentation of the package, including several examples and tutorials 
can be found [on ReadTheDocs](https://applefy.readthedocs.io).

This short guide will walk you through the required steps to set up and install
`applefy`.

The code was written for **Python 3.8 and above**
 

## Installation

The code of `applefy` is available on the [PyPI repository](https://pypi.org/project/applefy/)
as well as on 
[GitHub](https://github.com/markusbonse/applefy). We *strongly* recommend you 
to use a [virtual environment](https://virtualenv.pypa.io/en/latest/) to install
the package.

Applefy can only be used together with a data post-processing libary
for high-contrast imaging data! The following packages are currently 
supported:

1. [PynPoint](https://pynpoint.readthedocs.io/en/latest/installation.html)

2. [VIP](https://vip.readthedocs.io/en/latest/Installation-and-dependencies.html)

### Installation from PyPI

Just run:
```bash
pip install applefy
```

### Installation from GitHub

Start by cloning the repository and install `applefy` as a Python package:

```bash
git clone git@github.com:markusbonse/applefy.git ;
cd applefy ;
pip install .
```

In case you intend to modify the package you can install the package in 
"edit mode" by using the `-e` flag:

```bash
pip install -e .
```

### Additional Options

Depending on the use case `applefy` can be installed with additional options. 
If you install `applefy` from GitHub you can add them by:

```bash
pip install -e ".[option1,option2,...]"
```

If you install `applefy` from PiPy you can add them by:

```bash
pip install "applefy[option1,option2,...]"
```

The following options are available:
1. `dev`: Adds all dependencies needed to build the documentation page with
[sphinx](https://www.sphinx-doc.org/en/master/).
2. `fast_sort`: Installs the library 
[parallel_sort](https://pypi.org/project/parallel-sort/) which can speed up the
calculation of 
[bootstrap experiments](02_user_documentation/03_bootstrapping.ipynb). Since,
parallel_sort is a wrapper around the GNU library it is only supported on Linux.
3. `plotting`: Installs the libraries [seaborn](https://seaborn.pydata.org), 
[matplotlib](https://matplotlib.org) and 
[bokeh](https://docs.bokeh.org/en/latest/)
which we use in our plots. 
4. `vip`: Installs applefy with [VIP](https://vip.readthedocs.io/en/latest/Installation-and-dependencies.html).
Note, this option is conflicting with the 
option `pynpoint`.
5. `pynpoint`: Installs applefy with PynPoint using the [PynPoint](https://pynpoint.readthedocs.io/en/latest/installation.html)
version available on GitHub. Note, this option is conflicting with the option `vip`.

## Demonstration dataset
The tutorials in the 
[user documentation](https://applefy.readthedocs.io) are based 
on a demonstration dataset (NACO at the VLT). The data is publicly available
at [Zenodo](https://zenodo.org/record/7630239#.Y-auZy2cZQI). The repository 
contains three files:

1. `30_data`: This is the NACO L' Beta Pic dataset as a hdf5 already. 
The data was pre-processed with [PynPoint](https://pynpoint.readthedocs.io/en/latest/).
2. `70_results`: Contains results created by the tutorials of the user 
documentation. They are only needed if you don't want to compute your own PCA
residuals.
3. `laplace_lookup_tables.csv`: Are the lookup tables for the 
LaplaceBootstrapTest.

## Reproduce our results
Check out the [plot gallery](https://applefy.readthedocs.io/en/latest/04_apples_with_apples/01_general.html)
in the ``applefy`` documentation.

## Authors
All code was written by Markus J. Bonse, with additional contributions from 
Timothy Gebhard.