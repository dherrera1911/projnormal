# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import glob
import inspect
import os
import pathlib
import sys

import torch

import projnormal

sys.path.insert(0, os.path.abspath("../../src/"))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
project = 'projnormal'
copyright = '2025, Daniel Herrera-Esposito'
author = 'Daniel Herrera-Esposito'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    #"numpydoc",
]

intersphinx_mapping = {
    "torch": ("https://pytorch.org/docs/stable/", None)
}

add_module_names = False

autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'private-members': False,
    'special-members': False,
    'inherited-members': False,
    'show-inheritance': True,
    'module-first': True,
}


myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True

autosummary_generate = True
exclude_patterns = [
    'projnormal/classes/projected_normal.py',
    'projnormal/classes/const.py',
    'projnormal/classes/ellipse.py',
    'const.py',
    'ellipse_const.py',
    'ellipse.py',
]

#numpydoc_class_members_toctree = False




# -- Execution Configuration ---------------------------------------------------

# Remove execution time limit
nb_execution_timeout = -1  # Use -1 or None to disable the timeout

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
#html_theme_options = {}


# MYST

myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "amsmath",
]
