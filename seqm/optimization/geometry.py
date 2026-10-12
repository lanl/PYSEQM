from ._geometric import run_geometric


def geomeTRIC_optimization(
    molecule, traj_file: str = "geom_opt_traj.xyz", output_file: str = "optimized.xyz", **run_kwargs
):
    """Optimize a single PYSEQM molecule with geomeTRIC."""

    def evaluate(driver):
        driver(molecule, P0=molecule.dm, dm_prop="SCF")
        return molecule.Etot.item(), -molecule.force[0], None

    return run_geometric(molecule, molecule.seqm_parameters, evaluate, traj_file, output_file, run_kwargs)
