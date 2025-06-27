Single-Point SCF Calculations
=============================

Computes the electronic ground state wave function of molecular systems using the Self-Consistent Field (SCF) method. SCF iteratively solves the electronic Schrödinger equation within an approximate mean-field framework, where the motion of each electron is averaged over the positions of all others. This process refines the electronic density matrix until self-consistency is achieved.

See :ref:`seqm-parameters` for details on specifying the options for an SCF calculation.

The **Electronic_Structure** driver performs single-point ground-state SCF (and, with the right flags, excited-state) calculations on one or more molecules in a batch.  
Internally, it inherits from `torch.nn.Module`, so you can treat it like any other PyTorch model (e.g. move it to GPU, etc.).


Basic Usage
-----------

.. code-block:: python

    import torch
    from seqm.seqm_functions.constants import Constants
    from seqm.Molecule import Molecule
    from seqm.ElectronicStructure import Electronic_Structure

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


    const = Constants().to(device)

    seqm_parameters = {
    'method': 'AM1',
    'scf_eps': 1.0e-6,
    'scf_converger': [2, 0.0],
    }

    molecule = Molecule(const, seqm_parameters, coordinates, species).to(device)

    esdriver = Electronic_Structure(seqm_parameters).to(device)
    esdriver(molecule)
    
    print("Total Energy (eV):", molecule.Etot)
    print("Forces (eV/Å):", molecule.force)

Accessing Results
-----------------
After ``esdriver(molecule)`` has run, inspect the ``molecule`` attributes:

- ``molecule.Hf`` Heat of formation (eV)
- ``molecule.Etot`` Total Energy (electronic + nuclear energies), :math:`E_{\mathrm{tot}}` (eV)
- ``molecule.Eelec`` Electronic Energy (eV)
- ``molecule.Enuc`` Nuclear-nuclear repulsion energy (eV)
- ``molecule.force`` Forces on atoms (eV/Å), equal to :math:`-\nabla E_{\mathrm{tot}}`
- ``molecule.e_mo`` Molecular orbital eigenvalues (eV)
- ``molecule.e_gap`` HOMO–LUMO Gap (eV)

Batch Processing
----------------
SCF runs on all molecules in your batch simultaneously, so you’ll get
all result attributes as **batched** tensors.  For example, if you have
5 molecules in your input, then ``molecule.Etot`` is a length-5 tensor.
