import torch

from seqm.ElectronicStructure import Electronic_Structure
from seqm.Molecule import Molecule
from seqm.seqm_functions.constants import Constants

from ..reference_data import assert_allclose, load_or_update_reference, reference_path


def test_pm6_batch_from_notebook(device):
    torch.set_default_dtype(torch.float64)
    species = torch.as_tensor(
        [[16, 16], [22, 22], [22, 16], [35, 17], [24, 22]], dtype=torch.int64, device=device
    )

    coordinates = torch.tensor(
        [
            [[0.0000, 0.0, 0.0000], [0.0000, 1.2, 0.0000]],
            [[0.0000, 0.0, 0.0000], [0.0000, 1.2, 0.0000]],
            [[0.0000, 0.0, 0.0000], [0.0000, 1.2, 0.0000]],
            [[0.0000, 0.0, 0.0000], [0.0000, 1.2, 0.0000]],
            [[0.0000, 0.0, 0.0000], [0.0000, 1.2, 0.0000]],
        ],
        dtype=torch.float64,
        device=device,
    )

    const = Constants().to(device)

    seqm_parameters = {
        "method": "PM6",
        "scf_eps": 1.0e-5,
        "scf_converger": [0, 0.2],
        "sp2": [False, 1.0e-5],
        "pair_outer_cutoff": 1.0e10,
        "eig": True,
        "Hf_flag": True,
        "scf_backward": 0,
        "UHF": False,
    }

    molecule = Molecule(const, seqm_parameters, coordinates, species).to(device)
    esdriver = Electronic_Structure(seqm_parameters).to(device)
    esdriver(molecule)

    data = {"Etot": molecule.Etot.detach().cpu().tolist(), "force": molecule.force.detach().cpu().tolist()}
    ref_path = reference_path("pm6_batch_notebook")
    ref = load_or_update_reference(ref_path, data)

    assert_allclose(data["Etot"], ref["Etot"], rtol=1e-5, atol=1e-5)
    assert_allclose(data["force"], ref["force"], rtol=1e-5, atol=1e-5)
