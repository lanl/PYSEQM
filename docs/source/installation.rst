Installation
============

Clone and Install from Source
------------------------------

This is the recommended way to install PYSEQM

.. code-block:: bash

      git clone https://github.com/lanl/PYSEQM.git
      cd PYSEQM
      pip install .




Verify the Installation
-------------------------

After installation, confirm that the package imports correctly:

.. code-block:: bash

   python -c "import seqm; print('✔ PYSEQM imported successfully')"

Check GPU Support (Optional)
----------------------------
If you have NVIDIA CUDA–enabled hardware and installed a compatible PyTorch build, verify GPU availability:

.. code-block:: bash

    python -c "import torch; print(f'PyTorch installed | CUDA available: {torch.cuda.is_available()}')"


