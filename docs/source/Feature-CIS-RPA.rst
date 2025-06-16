CIS Excited State Calculations
------------------------------

Estimates electronically excited-state energies using the Configuration Interaction Singles (CIS) method, which extends SCF results to higher energy eigenstates of the electronic Hamiltonian. The CIS wavefunction approximates excited states by considering single-electron excitations from the ground-state configuration. While less computationally intensive than correlated methods like the Random Phase Approximation (RPA), CIS provides a stable and widely used framework for modeling vertical excitation energies.

This implementation allows users to compute multiple excited states efficiently, making it suitable for large-scale or semiempirical simulations.


Must be added to run Excited States

For excited state calculations, add the `excited_states` key to your `seqm_parameters` dictionary:

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
n_states  
Number of excited states to compute.

**method**  
Method used for excited state calculations. Available options:

- `'rpa'`
- `'cis'`
By default it is set to `cis`

**cis_tolerance**

is optional set can be left blank 
Convergence criterion for CIS/RPA excited states. By default it is set to 1e-6.



See :ref:`seqm-parameters` for more personalized parameter settings for the calculation.




.. code-block:: python

   import torch
   from seqm.seqm_functions.constants import Constants
   from seqm.Molecule import Molecule
   from seqm.ElectronicStructure import Electronic_Structure
   from seqm.seqm_functions.read_xyz import read_xyz

   torch.set_default_dtype(torch.float64)
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

   species = torch.as_tensor([[8,6,1,1],
                              [8,6,1,1],
                              [8,8,6,0]],
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
                                 [1.82,    0.94,    0.00],
                                 [1.82,   -0.94,    0.00]
                                 ],
                                 [
                                 [0.00,    0.00,    0.00],
                                 [1.23,    0.00,    0.00],
                                 [1.82,    0.94,    0.00],
                                 [0.0,     0.0,     0.0]
                                 ]
                              ], device=device)

   species = torch.as_tensor(species, dtype=torch.int64, device=device)[:]
   coordinates = torch.tensor(coordinates, device=device)[:]
   const = Constants().to(device)

   elements = [0] + sorted(set(species.reshape(-1).tolist()))

   seqm_parameters = {
      'method': 'AM1',
      'scf_eps': 1.0e-8,
      'scf_converger': [2, 0.0],
      'sp2': [False, 1.0e-5],
      'learned': [],
      'pair_outer_cutoff': 1.0e8,
      'eig': True,
      'excited_states': {'n_states': 10},
   }

   molecules = Molecule(const, seqm_parameters, coordinates, species).to(device)
   esdriver = Electronic_Structure(seqm_parameters).to(device)
   esdriver(molecules)
