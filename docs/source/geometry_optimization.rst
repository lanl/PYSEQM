Geometry Optimization
=====================

Overview
--------
PYSEQM provides two approaches for finding minimum-energy structures:

- **geomeTRIC** integration 
  
  Robust, advanced algorithms using an external package called `geomeTRIC <https://geometric.readthedocs.io/en/latest/>`_; single-molecule only.
- **Steepest-Descent (SD)** via `Geometry_Optimization_SD` driver  

  Batch-capable but may converge extremely slowly or fail on complex surfaces.  

geomeTRIC Integration
---------------------
PYSEQM can run robust optimization on a single molecule using `geomeTRIC <https://geometric.readthedocs.io/en/latest/>`_ as the optimization driver.
First install the geomeTRIC interface:

.. code-block:: bash

   cd PYSEQM/
   pip install '.[geomopt]'

Then in your script:

.. code-block:: python

   from seqm.geometryOptimization import geomeTRIC_optimization

   # `molecules` must be a single-entry batch:
   geomeTRIC_optimization(molecules)


Example optimization with geomeTRIC
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import torch
    from seqm.seqm_functions.constants import Constants
    from seqm.Molecule import Molecule
    from seqm.geometryOptimization import geomeTRIC_optimization
    
    torch.set_default_dtype(torch.float64)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    species = torch.as_tensor([[6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1]] ,                           
                              dtype=torch.int64, device=device)
    
    coordinates = torch.tensor([
                                 [
                                   [ -35.894866 ,   9.288626 ,  10.054420], 
                                   [ -36.369609 ,   8.136872 ,   9.487425],
                                   [ -34.536331 ,   9.569570 ,   9.955502],
                                   [ -35.492308 ,   7.159723 ,   8.983842],
                                   [ -33.712643 ,   8.697458 ,   9.304412],
                                   [ -34.183340 ,   7.494548 ,   8.762036],
                                   [ -36.585465 ,   9.939934 ,  10.537961],
                                   [ -37.450824 ,   8.064607 ,   9.436016],                            
                                   [ -34.070057 ,  10.502494 ,  10.500065],                            
                                   [ -35.950748 ,   6.201235 ,   8.765212],                            
                                   [ -33.438373 ,   6.826374 ,   8.246956],                            
                                   [ -32.654911 ,   8.872607 ,   9.123019]
                                  ],
                               ], device=device)
    
    const = Constants().to(device)
    
    seqm_parameters = {
                       'method' : 'AM1', 
                       'scf_eps' : 1.0e-8,
                       'scf_converger' : [0,0.1],
                       }
    
    molecules = Molecule(const, seqm_parameters, coordinates, species).to(device)

    geomeTRIC_optimization(molecules)



Built-in Steepest-Descent
-------------------------
Use `Geometry_Optimization_SD` driver for simple, batched optimizations:

.. code-block:: python

   from seqm.MolecularDynamics import Geometry_Optimization_SD

   opt = Geometry_Optimization_SD(
       seqm_parameters,
       alpha=0.008,          # scaling factor for step size
       force_tol=1e-2,       # converge when max force < 0.01 eV/Å
       max_evl=40            # max steps (energy+force evaluations)
   ).to(device)

   max_force, dE = opt.run(molecule)

- **alpha** (`float`): scaling factor for scaling step size of coordinate updates  
- **force_tol** (`float`): convergence threshold on maximum force  
- **max_evl** (`int`): maximum number of evaluations  

.. warning::
   Steepest-descent can be very inefficient and may fail for rough potential-energy surfaces.
