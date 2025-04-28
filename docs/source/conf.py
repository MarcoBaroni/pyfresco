# Configuration file for the Sphinx documentation builder.

import os
import sys

# -- Path setup --------------------------------------------------------------

# Add the project root directory to sys.path, so Python modules can be found
sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------

project = 'FRESCO'
copyright = '2025, Marco Baroni, Beatrice Baschetti, Alessandro Pisello, '
copyright += 'Maurizio Petrelli, Massimiliano Porreca, Matteo Massironi'
author = 'Marco Baroni, Beatrice Baschetti, Alessandro Pisello, Maurizio Petrelli, Massimiliano Porreca, Matteo Massironi'

release = '0.1.1'  # Full version (e.g., "0.1.1")

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',        # Auto-document docstrings
    'sphinx.ext.napoleon',       # Support for Google/NumPy-style docstrings
    'sphinx.ext.intersphinx',    # Link to other project's documentation
    'sphinx.ext.viewcode',       # Add links to highlighted source code
    'sphinx.ext.mathjax',        # LaTeX math rendering
    'sphinx.ext.autosummary',    # Generate function/class summary tables
]

# Automatically generate summary tables
autosummary_generate = True

# Mock heavy libraries (torch, rasterio, etc.)
autodoc_mock_imports = ["torch", "torchvision", "rasterio", "spectral"]

# Improve autodoc output
autodoc_preserve_defaults = True
autodoc_typehints = "description"

# Paths
templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']

# -- Intersphinx Mapping -----------------------------------------------------
# Example to link to Python Standard Library
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
}


                    
