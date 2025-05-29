.. FRESCO documentation master file, created by
   sphinx-quickstart on Tue Mar 11 12:36:10 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to FRESCO documentation!
====================

.. figure:: _static/fresco.png
   :alt: Logo by https://www.ilariareed.it
   :width: 325px
   :align: center

FRESCO is a Free and Open Source Python module made to 
open, extract, preprocess and analyze spectral data gathered from the 
Compact Reconnaissance Imaging Spectromoter for Mars (CRISM).
This Python package features 4 different modules:
 - RGB maps: possibility to create RGB maps from spectral paramters datacube. It is made up of a Python function that permits the user to open the CRISM MTRDR reflectance and spectral parameters datacubes, and a class that permits the creation and the interactive adjustement of RGB maps;
 - Spectra Extraction: three different means of spectra extraction from the CRISM datacube. It is made up by a single class with which it possible to either draw on a given RGB map a polygon, a square or a point from which to select the spectra from the corresponding pixel position in the refelctance datacube;
 - Spectra Normalization: various methods to normalize and pre process the target spectrum. It is made up of a single Python class in which 5 different methods for spectra continuum removal are implemented. Moreover two methods for smoothing are conceived inside the class and two methods for final normalization are possible;
 - Spectra Analysis: possibility to either make analogue comparisons and/or spectral deconvolution. It is made up by 2 Python classes, one for direct spectral comparison with the spectra from the MICA files (Viviano-Beck et al., 2015) and for Machine Learning driven Modified Gaussian Models (Sunshine et al., 1990). The other class permits the analysis of mafic-related spectra with a pipeline adapted from Horgan et al., 2014.

.. toctree::
   :maxdepth: 2

   modules

Logo by https://www.ilariareed.it

