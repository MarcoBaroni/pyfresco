__author__ = 'Marco Baroni, Beatrice Baschetti, Alessandro Pisello, Matteo Massironi, Maurizio Petrelli'


# pyFRESCO/__init__.py
from .RGBmap import open_raw, RGBImageManipulator
from .SpectraExtract import SpectraExtract
from .SpectraNorm import SpectraNorm
from .SpectraAnalysis import SpectraAnalysis

__all__ = ["open_raw","RGBImageManipulator", "SpectraExtract", "SpectraNorm", "SpectraAnalysis"]
