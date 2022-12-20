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
        "h5py"],
    extras_require={
        "dev": ["furo", "sphinx"],
        "fast_sort": ["parallel_sort", ],
    },
    packages=["applefy"],
    zip_safe=False,
)
