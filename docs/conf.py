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
import os
import sys

sys.path.insert(0, os.path.abspath('../'))


# -- Project information -----------------------------------------------------

project = 'csoundengine'
copyright = '2021, Eduardo Moguillansky'
author = 'Eduardo Moguillansky'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'autodocsumm',
    'sphinx_automodapi.automodapi',
    'sphinx_automodapi.smart_resolver',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
    'sphinx.ext.viewcode',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.graphviz',
    # "sphinx.ext.intersphinx",
    # "sphinx.ext.extlinks",
    "sphinx_design",
    # "sphinx-book-theme"
    # "sphinxawesome_theme.highlighting",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'piccolo_theme'
html_theme = 'sphinx_book_theme'
# html_theme = 'sphinxawesome_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "friendly"


typehints_fully_qualified = False
typehints_document_rtype = True

# Autodoc
autodoc_member_order = 'bysource'

set_type_checking_flag = False

# numpydoc_show_class_members = False

autodoc_mock_imports = ["libcsound", "xxhash"]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
# napoleon_include_private_with_doc = True
# napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

html_theme_options = {
    # 'navigation_depth': 3,
    'use_fullscreen_button': False,
    'use_download_button': False,
    "show_toc_level": 2
    # 'show_navbar_depth': 2
}

html_css_files = [
    'custom.css',
]


