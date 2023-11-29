# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
from curses import version
import sys
import os
sys.path.insert(0, os.path.abspath('..'))
from version import _version

project = 'HemTracer'
copyright = '2023, Nico Dirkes'
author = 'Nico Dirkes'
version = _version
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.mathjax', 'sphinxcontrib.bibtex']
bibtex_bibfiles = ['refs.bib']
bibtex_default_style = 'plain'

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '*.old.*']

autodoc_type_aliases = {
    'Iterable': 'Iterable',
    'ArrayLike': 'ArrayLike',
    'NDArray': 'NDArray',
}

toc_object_entries_show_parents = 'hide'
autodoc_typehints = "description"
autoclass_content = "both"
autodoc_member_order = 'alphabetical' # 'bysource'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
