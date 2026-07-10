#!/usr/bin/env python3
"""Compare HiphopNN E/F/NAC with a fresh PySEQM calculation."""

import argparse

import numpy as np
import torch
from hiphop_predictor import HiphopPredictor
from hippynn_fssh import HANDOFF, pairs, states

from seqm.api import Constants, Electronic_Structure, Molecule, read_xyz

torch.set_default_dtype(torch.float64)


def error(ref, pred, phase_less=False):
    ref, pred = np.asarray(ref), np.asarray(pred)
    if phase_less:
        plus, minus = np.abs(ref - pred), np.abs(ref + pred)
        pred = np.where(
            plus.reshape(len(ref), -1).sum(1)[:, None] <= minus.reshape(len(ref), -1).sum(1)[:, None],
            pred.reshape(len(ref), -1),
            -pred.reshape(len(ref), -1),
        ).reshape(pred.shape)
    delta = pred - ref
    return np.abs(delta).mean(), np.sqrt(np.square(delta).mean())


def print_error(name, ref, pred, unit, phase_less=False):
    mae, rmse = error(ref, pred, phase_less)
    print(f"{name:<22} {mae:12.6e} {rmse:12.6e}  {unit}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default=HANDOFF / "model")
    p.add_argument("--xyz-file", nargs="+", required=True)
    p.add_argument("--states", default="0,1,2,3")
    p.add_argument("--method", default="AM1", help="PySEQM reference method; it must match training.")
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    label, nac_pairs = states(a.states), pairs(states(a.states))
    if 0 not in label:
        raise ValueError("include S0 in --states to compare absolute energies")
    dev = torch.device(
        "cuda"
        if a.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if a.device == "auto"
        else a.device
    )
    Z, R = read_xyz(a.xyz_file, sort=False)
    predictor = HiphopPredictor(
        a.model_dir, HANDOFF / "src", states=label, pairs=nac_pairs, device=dev, dtype="float64"
    )
    hip = predictor.predict(Z, R, gdv_source="forces")

    exc = [s for s in label if s]
    params = {
        "method": a.method,
        "elements": [1, 6, 8],
        "scf_eps": 1e-8,
        "scf_converger": [1],
        "active_state": 0,
        "analytical_gradient": [True],
        "do_all_forces": True,
        "excited_states": {"n_states": max(exc), "method": "cis"},
        # The HiphopNN NAC targets used this unrelaxed/no-response convention.
        "nonadiabatic": {
            "compute_nac": True,
            "pairs": [(int(x[0]), int(x[1])) for x in nac_pairs],
            "include_response_terms": False,
        },
    }
    mol = Molecule(
        Constants().to(dev),
        params,
        torch.as_tensor(R, dtype=torch.float64, device=dev),
        torch.as_tensor(Z, dtype=torch.long, device=dev),
    ).to(dev)
    mol.verbose = False
    Electronic_Structure(params).to(dev)(mol, do_force=True)

    print("quantity                         MAE         RMSE  unit")
    for s in label:
        seqm_E = (
            mol.Etot.detach().cpu().numpy()
            if s == 0
            else (mol.Etot + mol.cis_energies[:, s - 1]).detach().cpu().numpy()
        )
        print_error(f"E S{s}", seqm_E, hip["E_abs"][s], "eV")
        print_error(f"F S{s}", mol.all_forces[:, s].cpu().numpy(), hip["F"][s], "eV/A")
    for name in nac_pairs:
        i, j = map(int, name)
        gap = hip["sE"][j] - hip["sE"][i]
        hip_nac = hip["dENACR"][name] / gap[:, None]
        seqm_nac = mol.nac[i - 1, j - 1].cpu().numpy().reshape(len(Z), -1)
        print_error(f"NAC S{i}-S{j} (phase-less)", seqm_nac, hip_nac, "1/A", phase_less=True)
    print("PySEQM NAC reference: include_response_terms=False")


if __name__ == "__main__":
    main()
