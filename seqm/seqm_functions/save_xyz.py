def save_xyz(molecules, name, Forces=False):
    """
    create xyz file
    """
    write_forces = Forces
    for i in range(molecules.coordinates.shape[0]):
        species = molecules.species[i].detach().cpu().numpy()
        coordinates = molecules.coordinates[i].detach().cpu().numpy()
        forces = molecules.force[i].detach().cpu().numpy() if write_forces else None
        real_atoms = species > 0
        n_atoms = int(real_atoms.sum())
        fn = name + "." + str(i) + ".xyz"
        with open(fn, "w+") as f:
            try:
                energy = float(molecules.Etot[i].detach().cpu())
                f.write(f"{n_atoms}\nEtot = {energy:12.6f} \n")
            except (AttributeError, IndexError, TypeError):
                f.write(f"{n_atoms}\n\n")
            for atom in range(coordinates.shape[0]):
                if real_atoms[atom]:
                    label = molecules.const.label[species[atom]]
                    x, y, z = coordinates[atom]
                    if write_forces:
                        fx, fy, fz = forces[atom]
                        f.write(f"{label} {x:15.5f} {y:15.5f} {z:15.5f} {fx:15.5f} {fy:15.5f} {fz:15.5f} \n")
                    else:
                        f.write(f"{label} {x:15.5f} {y:15.5f} {z:15.5f}\n")
