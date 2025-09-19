.. _excited_states:

CIS & RPA Excited State Calculations
====================================

PYSEQM supports two methods for excited-state calculations: Configuration Interaction Singles (CIS) and Time-Dependent Hartree-Fock (TD-HF), also known as the Random Phase Approximation (RPA).

CIS constructs excited states by promoting electrons from occupied to virtual orbitals starting from the ground-state Hartree-Fock reference. These singly excited configurations form the basis for excited-state wavefunctions. CIS can be spin-adapted to target specific spin states (e.g., singlets) efficiently.

RPA includes both excitation and de-excitation operators, and accounts for more electronic correlation than CIS. 

Both methods are calculated with `Electronic_Structure` driver.

Configuring Excited States
--------------------------

In order to prompt an excited state calculation, simply add an `excited_states` key/value pair to your `seqm_parameters` dictionary:

.. code-block:: python

    seqm_parameters = {
        ...
        'excited_states': {
            'n_states': 10,
            'method': 'rpa',
            'cis_tolerance': 1e-5
        }
    }

The `excited_states` key takes a dictionary as its value. 
This value dictionary should contain the following key/value pairs: 

- **n_states**  (`int`): Number of excited states to compute.

- **method**  (`str`): Method used for excited state calculations. Available options:

  - ``'rpa'``
  - ``'cis'``

By default it is set to ``'cis'``

- **cis_tolerance** (`float`): Convergence criterion for CIS/RPA excited states. By default it is set to ``1e-6``.

See :ref:`seqm-parameters` for other settings for the calculation.


Running an Excited-State Calculation
------------------------------------

.. code-block:: python

   import torch
   from seqm.seqm_functions.constants import Constants
   from seqm.Molecule import Molecule
   from seqm.ElectronicStructure import Electronic_Structure

   torch.set_default_dtype(torch.float64)
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

   species = torch.as_tensor([[8,6,1,1],
                              [8,6,1,1],],
                           dtype=torch.int64, device=device)

   coordinates = torch.tensor([
                                 [
                                 [0.00,    0.00,    0.00],
                                 [1.22,    0.00,    0.00],
                                 [1.82,    0.94,    0.00],
                                 [1.82,   -0.94,    0.00]
                                 ],
                                 [
                                 [0.00,    0.00,    0.00],
                                 [1.22,    0.00,    0.00],
                                 [1.81,    0.94,   -0.20],
                                 [1.82,   -0.94,    0.20]
                                 ],
                              ], device=device)

   const = Constants().to(device)

   seqm_parameters = {
      'method': 'AM1',
      'scf_eps': 1.0e-8,
      'scf_converger': [1],
      'excited_states': {'n_states': 10},
   }

   molecule = Molecule(const, seqm_parameters, coordinates, species).to(device)
   esdriver = Electronic_Structure(seqm_parameters).to(device)
   esdriver(molecule)

After ``esdriver(molecule)`` has run, inspect the ``molecule`` attributes:

- ``molecule.cis_energies`` Tensor of shape `(batch, n_states)` containing excitation energies (eV) above the ground state.

In addition, excitation energies and a few other useful quantities (transition dipole moments, oscillator strengths) are printed to console.

Batch Homogeneity Requirement
-----------------------------
.. warning::
   Excited-state calculations (**CIS** or **RPA**) require a **homogeneous** batch: every molecule must have the **same atomic composition and ordering**. Only their **coordinates** may differ. Heterogeneous batches (different species across entries) are not supported for excited states. Use separate runs for mixed batches.
