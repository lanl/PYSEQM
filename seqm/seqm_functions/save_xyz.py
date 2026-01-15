import torch


def save_xyz(molecules, name, Forces=False):
    """
    create xyz file
    """
    for i in range(0, molecules.coordinates.shape[0]):
        fn = name + "." + str(i) + ".xyz"
        f = open(fn, "w+")
        try:
            f.write("{}\nEtot = {:12.6f} \n".format(torch.sum(molecules.species[i] > 0), molecules.Etot[i]))
        except:
            f.write("{}\n\n".format(torch.sum(molecules.species[i] > 0)))
        for atom in range(molecules.coordinates.shape[1]):
            if molecules.species[i, atom] > 0:
                if Forces:
                    f.write(
                        "{} {:15.5f} {:15.5f} {:15.5f} {:15.5f} {:15.5f} {:15.5f} \n".format(
                            molecules.const.label[molecules.species[i, atom].item()],
                            molecules.coordinates[i, atom, 0],
                            molecules.coordinates[i, atom, 1],
                            molecules.coordinates[i, atom, 2],
                            molecules.force[i, atom, 0],
                            molecules.force[i, atom, 1],
                            molecules.force[i, atom, 2],
                        )
                    )

                else:
                    f.write(
                        "{} {:15.5f} {:15.5f} {:15.5f}\n".format(
                            molecules.const.label[molecules.species[i, atom].item()],
                            molecules.coordinates[i, atom, 0],
                            molecules.coordinates[i, atom, 1],
                            molecules.coordinates[i, atom, 2],
                        )
                    )
        f.close()
