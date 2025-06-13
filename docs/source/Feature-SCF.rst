SCF Ground State Calculations
-----------------------------

Computes the electronic ground state energy of a molecular system using the Self-Consistent Field (SCF) method. SCF iteratively solves the electronic Schrödinger equation within an approximate mean-field framework, where the motion of each electron is averaged over the positions of all others. This process refines the electron density until self-consistency is achieved.




See :ref:`seqm-parameters` for more personalized parameter settings for the calculation.


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
    elements = [0] + sorted(set(species.reshape(-1).tolist()))

    seqm_parameters = {
    'method': 'AM1',
    'scf_eps': 1.0e-6,
    'scf_converger': [2, 0.0],
    'sp2': [False, 1.0e-5],
    'elements': elements,
    'learned': [],
    'pair_outer_cutoff': 1.0e10,
    'eig': True
    }

    molecules = Molecule(const, seqm_parameters, coordinates, species).to(device)
    esdriver = Electronic_Structure(seqm_parameters).to(device)
    esdriver(molecules)