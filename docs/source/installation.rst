Installation
============

Clone PySEQM
------------------------------

This is the recommended way to install PySEQM

.. code-block:: bash

      git clone https://github.com/lanl/PYSEQM.git


.. note::

   You must install PyTorch and Numpy first before installing PySEQM




.. Upgrade PySEQM
.. --------------

.. If you already have PySEQM installed, you can upgrade it using:

.. .. code-block:: bash

..    pip install --upgrade git+https://github.com/lanl/pyseqm.git


Test the Installation
---------------------

To verify the installation, run:

.. code-block:: bash

    python -c "import torch; print('PyTorch version:', torch.__version__)"
    python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
    python -c "import torch; print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"


.. You can set up an environment to run PySEQM if you have Conda or Miniconda installed

.. .. code-block:: bash

..     module load miniconda3
..     conda create -n pyseqm-env python=3.12
..     conda activate pyseqm-env
..     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
..     pip install git+https://github.com/lanl/pyseqm.git

