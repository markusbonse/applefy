"""
Setup script to install package.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from setuptools import setup


# -----------------------------------------------------------------------------
# RUN setup() FUNCTION
# -----------------------------------------------------------------------------

setup(
    name="applefy",
    version="0.1",
    description=(
        "applefy: A library to compute detection limits for high contrast"
        " imaging of exoplanets"
    ),
    author="Markus Bonse, Timothy Gebhard",
    license="MIT License",
    url="https://github.com/markusbonse/applefy.git",
    install_requires=[
        "astropy",
        "numpy",
        "pandas",
        "photutils",
        "numba",
        "scikit-learn",
        "scipy>=1.7",
        "tqdm",
        "h5py",
        "ipywidgets"],
    extras_require={
        "dev": ["furo",
                "sphinx>=2.1",
                "myst-parser",
                "nbsphinx",
                "sphinx-copybutton",
                "sphinx-gallery<=0.10",
                # needed for syntax highlighting in jupyter notebooks
                "IPython",
                # spell checking in jupyter notebooks
                "jupyter_contrib_nbextensions"],
        "fast_sort": ["parallel_sort==0.0.3", ],
        "plotting": ["seaborn", "matplotlib"],
    },
    packages=["applefy"],
    zip_safe=False,
)
