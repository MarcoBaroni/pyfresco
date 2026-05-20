Installation
============
We recommend using a virtual environment to keep your dependencies isolated. We strongly suggest installing pyFRESCO in a `Conda Environment <https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html>`_. 

Step 1: Installing Anaconda
---------------------------
First, install `Anaconda <https://www.anaconda.com/download/success>`_ for your OS.

Step 2: Installing pyFRESCO
-----------------------------
To create a new environment in the anaconda prompt write and run, row by row:

.. code-block:: bash

  conda create -n pyfresco python=3.10
  conda activate pyfresco

Now, there are two recommended ways to install *pyFRESCO*: via the Python Package Index (PyPI) or directly from the source repository on GitHub.

Step 2.1: Installing from PyPi
------------------------------
The easiest way to install the latest stable release of pyfresco is the following. In the anaconda or command prompt write and run:

.. code-block:: bash

  pip install pyfresco

This will also install all required dependencies (NumPy, SciPy, pandas, matplotlib, etc.).

Step 2.2: Installing from source
--------------------------------

If you want the latest development version, in the anaconda or command prompt write and run:

.. code-block:: bash

  pip install git+https://github.com/MarcoBaroni/pyfresco.git

Alternatively, clone the repository manually and install in editable mode. In the anaconda or command prompt write and run, row by row:

.. code-block:: bash

  git clone https://github.com/MarcoBaroni/pyfresco.git
  cd pyfresco
  pip install -e .

This setup allows you to track changes to the source code without reinstalling.

Step 2.3: Where to run
--------------------------------

PyFRESCO is a PYthon tool that can work on any Python editor, but we recommend to run pyFRESCO on the Jupyter Notebook/Lab, since it was the editor in which it was tested more.

To install Jupyter Notebook run:

.. code-block:: bash

  pip install notebook

Then to open it run:

.. code-block:: bash

  jupyter notebook

To install Jupyter Lab run:

.. code-block:: bash

  pip install jupyterlab

Then to open it run:

.. code-block:: bash

  jupyter lab

Troubleshooting
---------------

Upgrade pip, setuptools, and wheel to avoid most build issues:

.. code-block:: bash

  pip install --upgrade pip setuptools wheel
