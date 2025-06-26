Initialization
==============

Overview
--------
PYSEQM is built for **batch processing** of molecular systems, whether you have a single molecule or hundreds in one batch. This design maximizes GPU utilization for high-throughput semiempirical QM calculations and is especially useful in machine‐learning workflows where you often train or test on many samples at once.

Supported Molecular Input Formats
---------------------------------
You can supply molecular data (atoms and coordinates) in two ways:

- **PyTorch tensors** (in-memory, zero-padded batches)  
- **`.xyz` files** (read and padded automatically)


Using PyTorch Tensors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
When you provide tensors directly, the first dimension is the **batch size** (number of molecules). All molecules in the batch must share the same atom count, with zero-padding added to smaller molecules.
You should make the following two tensors:

- **`species`**: an integer tensor of shape `(batch, n_atoms)` containing atomic numbers, ordered from the highest to the lowest atomic number.
- **`coordinates`**: a float tensor of shape `(batch, n_atoms, 3)` with Cartesian coordinates in Å.

.. code-block:: python
   
    # Example: batch of 3 molecules, padded to 5 atoms each
    species = torch.as_tensor([
        [8, 6, 1, 0, 0],           # Molecule 1: O, C, H + 2 padding
        [1, 1, 0, 0, 0],           # Molecule 2: H2 + 3 padding
        [6, 1, 1, 1, 1],           # Molecule 3: CH₄ (no padding needed)
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
    ], dtype=torch.float64, device=device)


Reading from `.xyz` Files
~~~~~~~~~~~~~~~~~~~~~~~~~

PYSEQM can read multiple `.xyz` files and pad them to a uniform atom count:

An `.xyz` file includes:

1. Number of atoms (first line)
2. Comment or title (second line)
3. Atom symbol and coordinates per line (x, y, z)

Use `read_xyz` to load and pad.
The `read_xyz` function returns numpy tensors (species and coordinates) that have to be converted to PyTorch tensors

.. code-block:: python

    from seqm.seqm_functions.read_xyz import read_xyz

    species, coordinates = read_xyz([
        'data_one.xyz',
        'data_two.xyz',
        'data_three.xyz',
    ])

    species = torch.as_tensor(species,dtype=torch.int64,device=device)
    coordinates = torch.as_tensor(coordinates, device=device)


Device Selection & Precision
----------------------------

PySEQM supports running on both CPU and GPU. Set the device for your calculations using:

.. code-block:: python

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

This ensures that your calculation will use an NVIDIA CUDA GPU if one is available, and fall back to CPU otherwise.

All tensors and modules should be explicitly moved to the selected device to avoid runtime errors. For example,

.. code-block:: python

    const = Constants().to(device)

When creating new tensors, it’s recommended to specify the device directly:

.. code-block:: python

    coordinates = torch.tensor([...], device=device)

Using a consistent device across all tensors and operations is essential. Operations between tensors on different devices (e.g., one on CPU and one on GPU) will result in errors.


PySEQM requires **double precision** floats (`torch.float64`) to maintain numerical accuracy in electronic structure and molecular dynamics calculations.
Set this up using

.. code-block:: python

   torch.set_default_dtype(torch.float64)

Manually specify the datatype for tensors that don't need to have `torch.float64` type.



Units
-----

The default units used in PySEQM calculations are:

- **Length**: Å (angstroms)  
- **Energy**: eV (electronvolts)  
- **Temperature**: K (Kelvin)  
- **Time**: fs (femtoseconds)  
- **Force**: eV/Å  
- **Velocity**: Å/fs

These units are consistent across all molecular dynamics simulations and electronic structure calculations in PySEQM. Ensure that any input data (e.g., coordinates, velocities) and interpretation of output quantities use the appropriate units.


Imports  
---------------

**Required for all PySEQM calculations:**

These imports provide the core components needed to define molecules and access fundamental physical constants used throughout PySEQM simulations.

.. code-block:: python

    from seqm.seqm_functions.constants import Constants
    from seqm.Molecule import Molecule

**Required if reading molecular structures from a `.xyz` file:**

Use this to load molecular geometries from `.xyz` files

.. code-block:: python

    from seqm.seqm_functions.read_xyz import read_xyz

**Required for single-point ground-state and excited-state electronic structure calculations:**

This module performs semiempirical quantum mechanical calculations, including total energies, and gradients.

.. code-block:: python

    from seqm.ElectronicStructure import Electronic_Structure

**Required for Molecular Dynamics simulations:**

Use this for standard energy conserving molecular dynamics simulations

.. code-block:: python

    from seqm.MolecularDynamics import Molecular_Dynamics_Basic

**Required for Langevin Thermostat Molecular Dynamics:**

Includes stochastic and frictional forces to model interaction with a heat bath, enabling temperature control during simulations.

.. code-block:: python

    from seqm.MolecularDynamics import Molecular_Dynamics_Langevin

**Required for KSA-XL Born-Oppenheimer Molecular Dynamics:**

Implements an efficient Born-Oppenheimer MD scheme using extended Lagrangian and Krylov subspace methods for long, accurate simulations on quantum surfaces.

.. code-block:: python

    from seqm.MolecularDynamics import KSA_XL_BOMD






.. _seqm-parameters:

SEQM Parameters
---------------

The ``seqm_parameters`` dictionary defines settings for a semiempirical quantum mechanics (SEQM) simulation.
Below is a typical configuration:

.. code-block:: python

    seqm_parameters = {
        'method': 'AM1',
        'scf_eps': 1.0e-6,
        'scf_converger': [2,],
        'sp2': [False, 1.0e-5],
    }

Some of the basic parameters in the ``seqm_parameters`` dictionary are:

**method**  (`str`)
    Specifies the semiempirical model to use.

    :accepted values: ``'MNDO'``, ``'AM1'``, ``'PM3'``, ``'PM6'``

**scf_eps** (`float`) 
    Convergence threshold for the SCF (Self-Consistent Field) loop.
    The SCF iteration stops when the energy difference between steps is less than this value.

    :recommended: 1e-5 for single point SCF, 1e-8 for excited state calculations

**scf_converger** (`list`) 
    Specifies the alorithm used to update the density matrix to converge SCF.

    Available options:

    - ``[0, alpha]``  
      
      Constant linear mixing of the density matrix:  

      ``P_new = alpha * P_old + (1 - alpha) * P_candidate``  
      where ``alpha`` is the mixing coefficient (e.g., 0.2).

    - ``[1]``  
      
      Adaptive mixing, where instead of fixing `alpha`, you let the code estimate a nearly optimal `alpha` at each step, based on changes in the density matrix elements. This gives fast convergence when things are well-behaved, with automatic damping when needed. 

    - ``[1, K, L, M]``  
      
      Advanced adaptive mixing:  
        * Use linear mixing for the first ``M`` steps.  
        * Start with mixing coefficient ``K`` for the first 5 steps.  
        * Linearly transition from ``K`` to ``L`` between steps 6 and ``M``.  
        * After step ``M``, switch to adaptive mixing.

    - ``[2]``  
      
      Use adaptive mixing initially, then switch to Pulay DIIS algorithm.

**sp2**  (`list`)
    This is an alternative algorithm to update the density matrix at every step of the SCF procedure where density matrix expansion happens with second-order spectral projection polynomials (SP2).
    SP2 expands the density matrix in terms of the Hamiltonian Operator with an iterative scheme, which is much more efficient on modern GPU architectures.

    :format: ``[enabled, tolerance]``  
    * ``enabled`` (`bool`): Whether to activate SP2
    * ``tolerance`` (`float`): SP2 threshold. Recommened between 1e-3 to 1e-7

**eig**  (`bool`)
    Optional parameter to control whether to compute molecular orbitals after SCF convergence.

    - If ``True`` (default), the Fock matrix is diagonalized to obtain molecular orbitals.  
    - If ``False``, only the converged density matrix is computed.
