import numpy as np

def read_xyz(files):
    '''
    reads xyz structure from a list (files) of files names
    '''
    SPECIES = []
    COORDINATES = []
    for file in files:
        f = open(file)
        lines = f.readlines()
        f.close()
        Natoms = int(lines[0])
        species = []
        coords = []
        for i in range(2, 2+Natoms):
            species.append(int(lines[i].split()[0]))
            coords.append([int(lines[i].split()[0]), float(lines[i].split()[1]), float(lines[i].split()[2]), float(lines[i].split()[3])])
        SPECIES.append(species)
        COORDINATES.append(coords)
        COORDINATES = np.array(COORDINATES)
        COORDINATES = COORDINATES[:,(-1*COORDINATES[:, :, 0]).argsort()]
        SPECIES =COORDINATES[:,:,:,0].astype(int)[0]
        COORDINATES = COORDINATES[:,:,:,1:4][0]

    return SPECIES, COORDINATES
