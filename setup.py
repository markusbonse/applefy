"""
Setup script to install package.
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()


# -----------------------------------------------------------------------------
# RUN setup() FUNCTION
# -----------------------------------------------------------------------------

setup(
    name="applefy",
    version="0.0.2",
    description=(
        "applefy: A library to compute detection limits for high contrast"
        " imaging of exoplanets"
    ),
    author="Markus Bonse, Timothy Gebhard",
    author_email="mbonse@phys.ethz.ch",
    license="MIT License",
    url="https://github.com/markusbonse/applefy.git",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy~=1.20",
        "astropy~=4.3.1",
        "pandas~=1.5.2",
        "photutils~=1.3.0",
        "numba~=0.54.1",
        "scikit-learn~=1.0.2",
        "scipy~=1.7.3",
        "tqdm~=4.62.3",
        "h5py~=3.6.0"],
    extras_require={
        "dev": ["vip_hci>=1.3",
                #'pynpoint @ git+https://github.com/markusbonse/PynPoint.git#egg=pynpoint-0.10.1',
                "pynpoint==0.10.0",
                "furo",
                "sphinx_rtd_theme==1.1.1",
                "sphinx>=2.1",
                "myst-parser",
                "nbsphinx",
                "sphinx-copybutton",
                "sphinx-gallery<=0.10",
                "twine",
                # needed for syntax highlighting in jupyter notebooks
                "IPython",
                # spell checking in jupyter notebooks
                "jupyter_contrib_nbextensions",
                "sphinx-autodoc-typehints"],
        "fast_sort": ["parallel_sort==0.0.3", ],
        "plotting": ["seaborn", "matplotlib", "bokeh>=3.0.3"],
    },
    packages=find_packages(include=['applefy',
                                    'applefy.*']),
    zip_safe=False,
)
