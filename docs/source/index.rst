.. FRESCO documentation master file, created by
   sphinx-quickstart on Tue Mar 11 12:36:10 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to FRESCO documentation!
====================

FRESCO is a Free and Open Source Python module made to 
open, extract, preprocess and analyze spectral data gathered from the 
Compact Reconnaissance Imaging Spectromoter for Mars (CRISM).
This Python package features 4 different modules:
 - RGB maps: possibility to create RGB maps from spectral paramters datacube
 - Spectra Extraction: three different means of spectra extraction from the CRISM datacube
 - Spectra Normalization: various methods to normalize and pre process the target spectrum
 - Spectra Analysis: Possibility to either make analogue comparisons and/or spectral deconvolution. Also an option for mafic analysis is present.

FRESCO is installable by pip:

      pip install pyfresco

or from github:

      pip install git+https://github.com/MarcoBaroni/pyfresco.git

.. figure:: _static/fresco.png
   :alt: Logo by https://www.ilariareed.it
   :width: 325px
   :align: left

.. toctree::
   :maxdepth: 3

   modules

Logo by https://www.ilariareed.it

