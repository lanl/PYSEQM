Born Oppenheimer Molecular Dynamics
--------------------------------------------------------

Simulates nuclear motion on the electronic ground-state potential energy surface by recalculating the electronic structure at each time step. Forces are derived from fully converged self-consistent field (SCF) calculations, ensuring accurate energy conservation. Suitable for studying thermal processes, structural fluctuations, and vibrational dynamics in molecules. Computationally intensive due to repeated SCF optimizations at every step.



Must be added to run Molecular Dynamics 


The `output` dictionary controls the output settings for Molecular Dynamics (MD) runs:

.. code-block:: python

    output = {
        'molid': [0],
        'thermo': 1,
        'dump': 1,
        'prefix': '../../Outputs_location'
    }

:molid:  
    List of molecule IDs to simulate.  
    Example: ``[0]`` runs molecule 0.

:thermo:  
    Frequency (in time steps) to print XYZ position information to the screen.  
    For example, ``1`` = every timestep, ``2`` = every other timestep, etc.

:dump:  
    Frequency (in time steps) to write XYZ electronic structure trajectory to output files.
    Follows the same format as ``thermo``.

:prefix:  
    File path prefix for all output files.  
    Example: ``'../../Outputs_location'`` saves outputs to the specified directory.



See :ref:`seqm-parameters` for more personalized parameter settings for the calculation.






.. code-block:: python

   import torch
   from seqm.seqm_functions.constants import Constants
   from seqm.Molecule import Molecule
   from seqm.MolecularDynamics import Molecular_Dynamics_Basic
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
      'scf_eps': 1.0e-6,
      'scf_converger': [2, 0.0],
      'sp2': [False, 1.0e-5],
      'learned': [],
      'pair_outer_cutoff': 1.0e10,
   }


   output = {
   'molid': [0], 
   'thermo': 1, 
   'dump': 1, 
   'prefix': '../../Outputs_location'
   }

   molecule = Molecule(const, seqm_parameters, coordinates, species).to(device)
   md = Molecular_Dynamics_Basic(seqm_parameters=seqm_parameters, Temp=300.0, timestep=0.4, output=output).to(device)
   md.initialize_velocity(molecule)
   _ = md.run(molecule, 10, remove_com=[True, 1], Info_log=True)