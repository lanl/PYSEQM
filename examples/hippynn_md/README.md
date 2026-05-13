# HIPPYNN MD Example

This example runs `PYSEQM` molecular dynamics using a direct HIPPYNN energy/force driver.

Files included:

- `pyseqm_hippynn_md.py`: MD driver script
- `frame_77.xyz`: starting geometry
- `nve_md.input`: NVE example input
- `nvt_md.input`: NVT example input
- `model/experiment_structure.pt`
- `model/best_checkpoint.pt`
- `outputs/`: output directory

The model copy is minimal. Only the checkpoint files required by `load_checkpoint_from_cwd()` are included.

Run from inside the example directory:

```bash
cd examples/hippynn_md
python pyseqm_hippynn_md.py --infile nve_md.input
python pyseqm_hippynn_md.py --infile nvt_md.input
```

Outputs are written into `outputs/`.
