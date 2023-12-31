# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))
print("HERE", sys.path)


project = 'OpenProtein-Python'
copyright = '2023, NE47.bio'
author = 'NE47.bio'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

#extensions = [
#    #'recommonmark',
#    'sphinx_markdown_tables',
#    'myst_parser',
#    "sphinx_markdown_builder",
#]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'sphinx_rtd_theme'
html_theme = 'sphinx_rtd_theme' #sphinx_rtd_theme or  agogo

html_static_path = ['_static']
html_css_files = [
    'custom.css',
]

html_sidebars = {
    '**': ['globaltoc.html', 'sourcelink.html', 'searchbox.html'],
}


# add the extensions to the list of extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'recommonmark',
    'sphinx_markdown_tables',
    'sphinx_markdown_builder',
    'nbsphinx'
    #'myst_parser'
]

nbsphinx_execute = 'never'

# ... the rest of your configuration

# at the bottom of the file, add the following
from recommonmark.transform import AutoStructify
def setup(app):
    app.add_transform(AutoStructify)
