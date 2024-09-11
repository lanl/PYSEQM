import numpy as np

def read_xyz(files, sort = True):
    '''
    reads xyz structure from a list (files) of files names
    '''
    element_dict = {
                'H':  1,                                                                                                                                 'He':2,
                'Li': 3, 'Be': 4,                                                                                'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne':10,
                'Na': 11,'Mg':12,                                                                                'Al':13,'Si':14,'P': 15,'S': 16,'Cl':17,'Ar':18,
                'K':  19,'Ca':20,'Sc':21,'Ti':22,'V': 23,'Cr':24,'Mn':25,'Fe':26,'Co':27,'Ni':28,'Cu':29,'Zn':30,'Ga':31,'Ge':32,'As':33,'Se':34,'Br':35,'Kr':36,
                'Rb': 37,'Sr':38,'Y': 39,'Zr':40,'Nb':41,'Mo':42,'Tc':43,'Ru':44,'Rh':45,'Pd':46,'Ag':47,'Cd':48,'In':49,'Sn':50,'Sb':51,'Te':52,'I': 53,'Xe':54,
                }

    COORDINATES = []
    for file in files:
        f = open(file)
        lines = f.readlines()
        f.close()
        Natoms = int(lines[0])
        coords = []
        try:
            int(lines[2].split()[0])
            atoms_are_number = True
        except:
            atoms_are_number = False
        for i in range(2, 2+Natoms):
            #species.append(int(lines[i].split()[0]))
            if atoms_are_number:
                coords.append([int(lines[i].split()[0]), float(lines[i].split()[1]), float(lines[i].split()[2]), float(lines[i].split()[3])])
            else:
                coords.append([element_dict[lines[i].split()[0]], float(lines[i].split()[1]), float(lines[i].split()[2]), float(lines[i].split()[3])])
        COORDINATES.append(coords)
    COORDINATES = np.array(COORDINATES)
    if sort: COORDINATES = np.array([x[(-1*x[ :, 0]).argsort()] for x in COORDINATES])

    SPECIES =COORDINATES[:,:,0].astype(int)
    COORDINATES = COORDINATES[:,:,1:4]

    return SPECIES, COORDINATES