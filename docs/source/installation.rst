Installation
============

Clone PySEQM
------------------------------

This is the recommended way to install PySEQM

.. code-block:: bash

      git clone https://github.com/lanl/PYSEQM.git
      cd PYSEQM
      pip install .


.. note::

   You must install PyTorch and Numpy first before installing PySEQM



Test the Installation
---------------------

To verify the installation, run:

.. code-block:: bash

    python -c "import seqm; print('PySEQM imported successfully')"
    python -c "import torch; print(f'PyTorch installed | CUDA available: {torch.cuda.is_available()}')"


