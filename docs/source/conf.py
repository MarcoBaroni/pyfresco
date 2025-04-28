# Include the root directory for autodoc to find the package
import os
import sys
sys.path.insert(0, os.path.abspath('../../pyfresco'))

project = 'FRESCO'
copyright = '2025, Marco Baroni, Beatrice Baschetti, Alessandro Pisello, Maurizio Petrelli, Massimiliano Porreca, Matteo Massironi'
author = 'Marco Baroni, Beatrice Baschetti, Alessandro Pisello, Maurizio Petrelli, Massimiliano Porreca, Matteo Massironi'
release = '0.1.1'

extensions = [
    'sphinx.ext.autodoc',       # Automatically generate docs from docstrings
    'sphinx.ext.napoleon',      # Support for Google-style docstrings
    'sphinx.ext.intersphinx',   # Link to other Sphinx documentations
    'sphinx.ext.viewcode',      # Add links to highlighted source code
    'sphinx.ext.mathjax',  # Enables LaTeX-style math rendering
    'sphinx.ext.autosummary'
    ]

templates_path = ['_templates']
exclude_patterns = []

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "style_nav_header_background": "#007bff",  # Blue
}
html_static_path = ['_static']

autodoc_mock_imports = ["torch", "torchvision", "rasterio", "spectral"]
autodoc_preserve_defaults = True
autodoc_typehints = "description"

                    
