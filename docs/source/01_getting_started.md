# Getting Started
This short guide will walk you through the required steps to set up and install
`applefy`.
```{attention} 
The code was written for **Python 3.8 and above**
``` 

## Installation

The code of `applefy` is available on the **PyPI repository** as well as on 
[GitHub](https://github.com/markusbonse/applefy). We *strongly* recommend you 
to use a [virtual environment](https://virtualenv.pypa.io/en/latest/) to install
tha package.

```{attention} 
Applefy can only be used together with a data post-processing libary
for high-contrast imaging data! The follwing packages are currently 
supported:

1. [PynPoint](https://pynpoint.readthedocs.io/en/latest/installation.html)

2. [VIP](https://vip.readthedocs.io/en/latest/Installation-and-dependencies.html)
``` 

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
1. `dev`: Add all dependencies needed to build this documentation page with
[sphinx](https://www.sphinx-doc.org/en/master/).
2. `fast_sort`: Installs the library 
[parallel_sort](https://pypi.org/project/parallel-sort/) which can speed up the
calculation of 
[bootstrap experiments](02_user_documentation/03_bootstrapping.ipynb).
3. `plotting`: Installs the libraries [seaborn](https://seaborn.pydata.org), 
[matplotlib](https://matplotlib.org) and 
[bokeh](https://docs.bokeh.org/en/latest/)
which we use in our plots. 

## Demonstration dataset
The tutorials in the 
[user documentation](02_user_documentation/01_contrast_curves.ipynb) are based 
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
