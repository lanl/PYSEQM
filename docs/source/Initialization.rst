Initialization
==============

Input Formats
-------------

PySEQM supports input data from PyTorch tensors or `.xyz` files. 


Using PyTorch Tensors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each row represents a molecule. 


- species: a tensor of atomic numbers (Z) for the atoms in each molecule.
- coordinates: a tensor of 3D Cartesian coordinates (x, y, z) for each atom.

.. code-block:: python

    species = torch.as_tensor([
        [8, 6, 1, 0, 0],           # Molecule 1: O, C, H (padded with 2 zeros)
        [1, 1, 0, 0, 0],           # Molecule 2: H2 (padded with 3 zeros)
        [6, 1, 1, 1, 1],           # Molecule 3: CH4 (no padding needed)
    ], dtype=torch.int64, device=device)

    coordinates = torch.tensor([
        [  # Molecule 1
            [0.00, 0.00, 0.00],
            [1.22, 0.00, 0.00],
            [1.82, 0.94, 0.00],
            [0.00, 0.00, 0.00],  # padding
            [0.00, 0.00, 0.00],  # padding
        ],
        [  # Molecule 2
            [0.00, 0.00, 0.00],
            [0.74, 0.00, 0.00],
            [0.00, 0.00, 0.00],  # padding
            [0.00, 0.00, 0.00],  # padding
            [0.00, 0.00, 0.00],  # padding
        ],
        [  # Molecule 3
            [0.00, 0.00, 0.00],
            [0.63, 0.63, 0.63],
            [-0.63, -0.63, 0.63],
            [-0.63, 0.63, -0.63],
            [0.63, -0.63, -0.63],
        ],
    ], dtype=torch.float32, device=device)

Reading from `.xyz` Files
~~~~~~~~~~~~~~~~~~~~~~~~~

PySEQM can also read molecular data directly from `.xyz` files.

An `.xyz` file includes:

1. Number of atoms (first line)
2. Comment or title (second line)
3. Atom symbol and coordinates per line (x, y, z)

**Example `water.xyz` file**

.. code-block:: text

    3
    Water molecule
    O    0.000    0.000    0.000
    H    0.757    0.586    0.000
    H   -0.757    0.586    0.000

To read and pad multiple `.xyz` files:

.. code-block:: python

    species, coordinates = read_xyz([
        '../../data_one.xyz',
        '../../data_two.xyz',
        '../../data_three.xyz'
    ])

Padding for Variable Length Molecules
-------------------------------------

To batch molecules of varying sizes, **zero-padding** ensures all tensors have consistent shapes:

All molecules are padded to match the maximum number of atoms in the batch.



.. code-block:: python

    species = torch.as_tensor([
        [8, 1, 0, 0],     # 2 real atoms, 2 padded
        [8, 6, 1, 1],     # 4 real atoms
        [8, 6, 1, 0]      # 3 real atoms, 1 padded
    ], dtype=torch.int64, device=device)

    coordinates = torch.tensor([
        [
            [0.00,  0.00,  0.00],
            [1.22,  0.00,  0.00],
            [0.00,  0.00,  0.00],  # padding
            [0.00,  0.00,  0.00]   # padding
        ],
        [
            [0.00,  0.00,  0.00],
            [1.22,  0.00,  0.20],
            [1.82,  0.94,  0.00],
            [1.81, -0.93, -0.20]
        ],
        [
            [0.00,  0.00,  0.00],
            [1.23,  0.00,  0.00],
            [1.82,  0.94,  0.00],
            [0.00,  0.00,  0.00]   # padding
        ]
    ], device=device)

Atoms with ``species = 0`` and ``coordinates = [0.0, 0.0, 0.0]`` are ignored in model computation.


Device and GPU Usage
--------------------

PySEQM supports running on both CPU and GPU. To enable this, set the device automatically:

.. code-block:: python

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

This ensures that your code will use a GPU if one is available, and fall back to the CPU otherwise.

All tensors and model components should be explicitly moved to the selected device to avoid runtime errors:

.. code-block:: python

    tensor = tensor.to(device)

When creating new tensors, it’s recommended to specify the device directly:

.. code-block:: python

    coordinates = torch.tensor([...], device=device)

Using a consistent device across all tensors and operations is essential. Operations between tensors on different devices (e.g., one on CPU and one on GPU) will result in errors.



Precision
--------


PySEQM requires **double precision** floats (`torch.float64` for floats) to maintain numerical accuracy in electronic structure and molecular dynamics calculations.

Specify the dtype when creating new tensors. For example:

.. code-block:: python
    dtype=torch.float64  # for coordinates, etc.

Using lower precision (e.g., `torch.float32`) may result in:

- Inaccurate energy or force evaluations
- Unstable SCF convergence
- Incorrect gradient behavior in autograd



Imports  
---------------

**Required for all PySEQM calculations:**

These imports provide the core components needed to define molecules and access fundamental physical constants used throughout PySEQM simulations.

.. code-block:: python

    from seqm.seqm_functions.constants import Constants
    from seqm.Molecule import Molecule

**Required if reading molecular structures from a `.xyz` file:**

Use this to load molecular geometries from `.xyz` files, a standard format for storing atomic positions and elements.

.. code-block:: python

    from seqm.seqm_functions.read_xyz import read_xyz

**Required for ground and excited state electronic structure calculations:**

This module performs semiempirical quantum mechanical calculations, including total energies, forces, and excited states via methods like CIS.

.. code-block:: python

    from seqm.ElectronicStructure import Electronic_Structure

**Required for basic Molecular Dynamics simulations:**

Use this for standard molecular dynamics simulations with basic integration, suitable for energy conservation tests and observing free molecular motion.

.. code-block:: python

    from seqm.MolecularDynamics import Molecular_Dynamics_Basic

**Required for Langevin Thermostat Molecular Dynamics:**

Includes stochastic and frictional forces to model interaction with a heat bath, enabling temperature control during simulations.

.. code-block:: python

    from seqm.MolecularDynamics import Molecular_Dynamics_Basic, Molecular_Dynamics_Langevin

**Required for KSA-XL Born-Oppenheimer Molecular Dynamics:**

Implements an efficient Born-Oppenheimer MD scheme using extended Lagrangian and Krylov subspace methods for long, accurate simulations on quantum surfaces.

.. code-block:: python

    from seqm.MolecularDynamics import KSA_XL_BOMD








SEQM Parameters
---------------


The `seqm_parameters` dictionary defines settings for your semi-empirical quantum mechanics (SEQM) simulation.



.. code-block:: python

    seqm_parameters = {
        'method': 'AM1',
        'scf_eps': 1.0e-6,
        'scf_converger': [2, 0.0],
        'sp2': [False, 1.0e-5],
        'elements': elements,
        'learned': [],
        'pair_outer_cutoff': 1.0e10,
        'eig': True,
    }


**method**  
Specifies the semi-empirical method to use. Options:

`'MNDO'`

`'AM1'`

`'PM3'`

`'PM6'`

**scf_eps**  
Convergence threshold for the SCF (Self-Consistent Field) loop. The simulation stops if the energy change between steps is smaller than this value.

**scf_converger**  
Settings for the SCF convergence algorithm. Format: `[type, tolerance]`.  


**pair_outer_cutoff**  
Maximum distance between two atoms for considering interactions. Atoms farther apart are ignored.  

**eig**  
Whether or not to calculate the final molecular orbitals. If eig is set to False, then SCF calculates the converged density matrix only. If eig is set to true, the converged molecular orbitals are also calculated by diagonalizing the Fock matrix.

Must be added to run Excited States
------------------------

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
Convergence criterion for CIS/RPA excited states. By default it is set to 1e-6.


Must be added to run Molecular Dynamics 
----------------------------------

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
