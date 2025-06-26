.. _quick_start:

Quickstart
==========

This quickstart is a high-level introduction on how to get started with using PYSEQM. 

For full examples, please check out the `examples directory <https://github.com/lanl/PYSEQM/tree/master/examples>`_ in our `GitHub repository <https://github.com/lanl/pyseqm>`_.

1. Install PYSEQM
------------------
Refer to :doc:`installation` instructions.

2. Write Your First Input File
-------------------------------
Create a file called `run_quickstart.py` with the following contents:

.. code-block:: python

   #!/usr/bin/env python
   import torch
   from seqm.seqm_functions.constants import Constants
   from seqm.Molecule import Molecule
   from seqm.ElectronicStructure import Electronic_Structure

   # 1. Precision & device
   torch.set_default_dtype(torch.float64)
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

   # 2. Define a simple molecule (H₂O)
   species = torch.as_tensor([[8, 1, 1]], dtype=torch.int64, device=device)
   coordinates = torch.tensor(
       [[[0.00, 0.00, 0.00],
         [0.96, 0.00, 0.00],
         [-0.24, 0.93, 0.00]]],
       dtype=torch.float64, device=device
   )
   const    = Constants().to(device)

   # 3. Set up SCF parameters
   seqm_parameters = {
       'method':        'AM1',
       'scf_eps':       1e-8,
       'scf_converger': [0, 0.1],
   }

   molecule = Molecule(const, seqm_parameters, coordinates, species).to(device)

   # 4. Run single‐point SCF
   driver = Electronic_Structure(seqm_parameters).to(device)
   driver.(molecule)

   # 5. Print results
   print(f"Total Energy (eV):   {molecule.Etot.item():.6f}")
   print(f"Heat of Formation:   {molecule.Hf.item():.6f}")
   print(f"Force on Atoms:\n      {molecule.force}")

3. Execute the Script
---------------------
Run your quickstart script:

.. code-block:: bash

   python run_quickstart.py

You should see printed:

- Total energy (eV)  
- Heat of formation (eV)  
- Forces on each atom (eV/Å)

Next Steps
----------
- Consult :doc:`Initialization` for batching, GPU, and input formats
- Explore :doc:`single_point_scf` for more SCF options  
- Try :doc:`excited_states` to compute excited states  
- See :doc:`bomd` for running MD trajectories  
