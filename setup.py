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
