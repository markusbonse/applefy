# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------

project = 'applefy'
copyright = '2023, Markus Johannes Bonse'
author = 'Markus Johannes Bonse'

# The full version, including alpha/beta/rc tags
release = '0.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.napoleon',
              'sphinx.ext.autodoc',
              'nbsphinx',
              'sphinx_copybutton',
              'sphinx_gallery.load_style',
              'myst_parser',
              'sphinx_autodoc_typehints',
              'sphinx.ext.intersphinx']

# Gallery style
sphinx_gallery_conf = {
    'thumbnail_size': (50, 50)
}

napoleon_use_param = True
autodoc_member_order = 'bysource'

autodoc_default_options = {
    "members": True, "undoc-members": True, "show-inheritance": True}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "packaging": ("https://packaging.pypa.io/en/latest", None),
    "numpy": ('https://numpy.org/doc/stable/', None),
    "pandas": ('https://pandas.pydata.org/docs/', None),
    "sklearn": ('https://scikit-learn.org/stable/', None)
}

simplify_optional_unions = False

# The master toctree document.
master_doc = 'index'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static',]

# Setups for furo theme
html_theme_options = {
    #"announcement": "<em>Important</em> announcement!",
    "light_logo": "applefy_logo.pdf",
    "dark_logo": "applefy_logo.pdf",
    "sidebar_hide_name": True,
    "light_css_variables": {
        "color-brand-primary": "#4e4c4d",
        "color-sidebar-caption-text": "FFFFFF",
        "color-brand-content": "#7C4DFF",
    },
}
