# setup.py

from setuptools import setup, find_packages

setup(
    name="pyFRESCO",  
    version="0.1.0",  
    description="A package for CRISM MTRDR datacubes RGB image generation, spectra exctraction and spectra analysis.",
    author="Marco Baroni, Beatrice Baschetti, Alessandro Pisello, Matteo Massironi, Maurizio Petrelli",
    author_email="mbaroni96@gmail.com",
    url="...", 
    packages=find_packages(exclude=["tests"]),  
    include_package_data = True,
    install_requires=[
	"kneed>=0.8.5",
	"matplotlib>=3.7.5",
	"numpy>=1.24.4",
	"pandas>=2.0.3",
	"rasterio>=1.3.10",
	"scipy>=1.10.1",
	"spectral>=0.23.1",
	"torch>=2.4.0",
	"torchvision>=0.19.0"
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "flake8>=3.8"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
