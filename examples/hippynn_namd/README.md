# HIPPYNN surface-hopping example

`pyseqm_hippynn_namd.py` runs PySEQM fewest-switches surface hopping while the
bundled HiphopNN model supplies every electronic-structure quantity: energies,
state forces, and derivative-coupling vectors.  It is the nonadiabatic analogue
of [`../hippynn_md/pyseqm_hippynn_md.py`](../hippynn_md/pyseqm_hippynn_md.py).

Run a short S1/S2/S3 trajectory from this directory:

```bash
python pyseqm_hippynn_namd.py \
  --model-dir enol_ef_nacr_developer_handoff/model \
  --xyz-file enol_ef_nacr_developer_handoff/golden_test/xyz/f_50273.xyz \
  --states 0,1,2,3 --initial-state 3 \
  --nacr-pairs 12,13,23 \
  --timestep 0.25 --nsteps 10 \
  --device cpu --output-dir outputs --output-prefix smoke
```

Without `--velocities-file`, velocities are sampled from a Maxwell-Boltzmann
distribution at `--temperature`, rescaled to that temperature, and have overall
translation/rotation removed. To supply them yourself, use a whitespace `N×3`
text file (or `.npy`) in **Å/fs**:

```bash
python pyseqm_hippynn_namd.py ... --velocities-file velocities.txt
```

Those values are used unchanged and reused for every trajectory.

Compare the model against a fresh PySEQM calculation at one or more geometries:

```bash
python compare_hippynn_pyseqm.py \
  --xyz-file enol_ef_nacr_developer_handoff/golden_test/xyz/f_50273.xyz \
  --states 0,1,2,3 --method AM1 --device cpu
```

The comparison uses `nonadiabatic.include_response_terms=False`, matching the
NAC targets used to train the HIPPYNN model. NAC errors are reported phase-less.

The model's `dENACR_ij` is converted to the NAC vector as
`d_ij = dENACR_ij / (E_j - E_i)`, and PySEQM obtains the time-derivative
coupling with `nac_dot = v · d`.  The same NAC vectors are returned to FSSH for
velocity rescaling following a hop.  The model remains loaded once; it is not
reloaded for each MD step.

`--states 0,1,2,3` evaluates S0 as the energy reference, but PySEQM propagates
only the excited manifold S1/S2/S3.  S0 is excluded because this model has no
S0–Sn NACR heads.  Pair labels are physical excited-state labels, so the shown
space must use `12,13,23`.  Output nonadiabatic state labels are local FSSH
labels (therefore S1/S2/S3 for the shown state space).

Important limitations inherited from the supplied model:

- Its NACR head uses GDVs built from the model's own forces in this driver. That
  self-consistent path is mechanically implemented but not validated by the
  handoff package.
- Training used phase-less NACR targets. The driver picks a temporally
  continuous, pairwise-consistent state-sign gauge before `nac_dot_v` is used,
  but model quality near crossings still needs validation.

The `enol_ef_nacr_developer_handoff/README.md` documents the model, units, and
the prerequisite HIPPYNN version. First validate its stand-alone inference
command before relying on a dynamics trajectory.
