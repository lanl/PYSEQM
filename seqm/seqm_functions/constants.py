import torch

ev = 27.21 #used in mopac7
#1 hatree = 27.211386 eV
#ev =  27.211386

# a0=0.529167  #used in mopac7
a0 = 0.529167 # used in amber and nexmd
#a0=0.5291772109
ev_kcalpmol = 23.061  # 1 eV = 23.061 kcal/mol

charge_on_electron = 1.60217733e-19
speed_of_light = 2.99792458e8
to_debye = charge_on_electron*1e-10*speed_of_light/1e-21
debye_to_AU = 0.393456

"""
in mopac the cutoff for overlap is 10 Angstrom
Atomic units is used here
set cutoff = 20.0 Angstrom and 20/0.529167 = 37.8
"""
overlap_cutoff = 40.0

class Constants(torch.nn.Module):
    """
    Constants used in seqm
    """

    def __init__(self, length_conversion_factor=(1.0/a0), energy_conversion_factor=1.0):
        """
        Constructor
        length_conversion_factor : atomic unit is used for length inside seqm
            convert the length by  oldlength*length_conversion_factor  to atomic units
            default value assume Angstrom used outside, and times 1.0/bohr_radius
        energy_conversion_factor : eV usedfor energy inside sqem
            convert by multiply energy_conversion_factor
            default value assumes eV used outside
        """

        super().__init__()

        # atomic unit for length is used in seqm
        # 1.8897261246364832 = 1.0/0.5291772109  1.0/bohr radius
        # factor convert length to atomic unit (default is from Angstrom to atomic unit)
        self.length_conversion_factor = length_conversion_factor
        #self.a0 = 0.529167  #used in mopac7
        #self.a0=0.5291772109

        # factor converting energy to eV (default is eV)
        self.energy_conversion_factor = energy_conversion_factor
        #self.ev = 27.21 #used in mopac7
        #1 hatree = 27.211386 eV
        #self.ev =  27.211386
        #
        # valence shell charge for each atom type
        #tore[1]=1, Hydrogen has 1.0 charge on valence shell
        self.label=['0',
               'H',                                                                                            'He',
               'Li','Be',                                                            'B', 'C',  'N', 'O', 'F', 'Ne',
               'Na','Mg',                                                            'Al','Si',' P',' S', 'Cl','Ar',
               'K', 'Ca','Sc', 'Ti', 'V ', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga','Ge',' As','Se','Br','Kr',
               'Rb','Sr','Y ', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In','Sn',' Sb','Te','I','Kr',
               'Cs','Ba','La', 'Hf', 'Ta', 'W ', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl','Pb',' Bi','Po','At','Rn']

        atomic_num=torch.as_tensor([0.0,
               1.0,                                                                                                   0.0,
               3.0,4.0,                                                                  5.0,  6.0,  7.0,  8.0,  9.0, 0.0,
               11.0,12.0,                                                               13.0, 14.0, 15.0, 16.0, 17.0, 0.0,
               19.0,20.0,   21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 0.0,
               1.0,2.0,3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 3.0,4.0,5.0,6.0,7.0,0.0,
               1.0,2.0,3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 3.0,4.0,5.0,6.0,7.0,0.0] )
        
        
        tore=torch.as_tensor([0.0,
               1.0,                                                                             0.0,
               1.0,2.0,                                                     3.0,4.0,5.0,6.0,7.0,0.0,
               1.0,2.0,                                                     3.0,4.0,5.0,6.0,7.0,0.0,
               1.0,2.0,3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 3.0,4.0,5.0,6.0,7.0,0.0,
               1.0,2.0,3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 3.0,4.0,5.0,6.0,7.0,0.0,
               1.0,2.0,3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 3.0,4.0,5.0,6.0,7.0,0.0] )


        #
        #principal quantum number for valence shell
        # qn[1] = 1, principal quantum number for the valence shell of Hydrogen is 1
        qn = torch.as_tensor([0.0,
               1.0,                                                                          1.0,
               2.0,2.0,                                                  2.0,2.0,2.0,2.0,2.0,2.0,
               3.0,3.0,                                                  3.0,3.0,3.0,3.0,3.0,3.0,
               4.0,4.0,4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0,4.0,4.0,4.0,4.0,4.0,
               5.0,5.0,5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,5.0,5.0,5.0,5.0,5.0,
               6.0,6.0,6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0,6.0,6.0,6.0,6.0,6.0])



        qnd = torch.as_tensor([0.0,
               0.0,                                                                          0.0,
               0.0,0.0,                                                  0.0,0.0,0.0,0.0,0.0,0.0,
               0.0,0.0,                                                  3.0,3.0,3.0,3.0,3.0,0.0,
               0.0,0.0,3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 0.0,0.0,4.0,4.0,4.0,0.0,
               0.0,0.0,4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 0.0,0.0,5.0,5.0,5.0,0.0,
               0.0,0.0,5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 0.0,0.0,0.0,0.0,0.0,0.0])

        #
        qn_int = qn.type(torch.int64)
        qnD_int = qnd.type(torch.int64)
        #number of s electrons for each element

        iso=torch.as_tensor([0.0,
                              0.0,                                                                0.0,
                              0.0,0.0,                                        0.0,0.0,0.0,0.0,0.0,0.0,
                              0.0,0.0,                                        0.0,0.0,0.0,0.0,0.0,0.0,
                              0.0,0.0,-34.000,-63.460,-100.614,-185.725,-195.802,-426.835,-167.657,-485.166,-656.595,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
                              0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
                              0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])


        ussc=torch.as_tensor([0.0,
                              1.0,                                                                0.0,
                              1.0,2.0,                                        2.0,2.0,2.0,2.0,2.0,0.0,
                              1.0,2.0,                                        2.0,2.0,2.0,2.0,2.0,0.0,
                              1.0,2.0,2.0,2.0,2.0,1.0,2.0,2.0,2.0,2.0,1.0,2.0,2.0,2.0,2.0,2.0,2.0,0.0,
                              1.0,2.0,2.0,2.0,1.0,1.0,2.0,1.0,1.0,0.0,1.0,2.0,2.0,2.0,2.0,2.0,2.0,0.0,
                              1.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,1.0,1.0,2.0,2.0,2.0,2.0,2.0,2.0,0.0])
        #
        #number of p electrons for each element
        uppc=torch.as_tensor([0.0,
                              0.0,                                                                0.0,
                              0.0,0.0,                                        1.0,2.0,3.0,4.0,5.0,6.0,
                              0.0,0.0,                                        1.0,2.0,3.0,4.0,5.0,6.0,
                              0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,2.0,3.0,4.0,5.0,6.0,
                              0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,2.0,3.0,4.0,5.0,6.0,
                              0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,2.0,3.0,4.0,5.0,6.0])
        #
        #
        gssc=torch.as_tensor([0.0,
                              0.0,                                                                0.0,
                              0.0,1.0,                                        1.0,1.0,1.0,1.0,1.0,0.0,
                              0.0,1.0,                                        1.0,1.0,1.0,1.0,1.0,0.0,
                              0.0,1.0,1.0,1.0,1.0,0.0,1.0,1.0,1.0,1.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0,
                              0.0,1.0,1.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0,
                              0.0,1.0,1.0,1.0,1.0,0.0,1.0,1.0,1.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0])
        #
        #
        gspc=torch.as_tensor([0.0,
                              0.0,                                                                 0.0,
                              0.0,0.0,                                        2.0,4.0,6.0,8.0,10.0,0.0,
                              0.0,0.0,                                        2.0,4.0,6.0,8.0,10.0,0.0,
                              0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,2.0,4.0,6.0,8.0,10.0,0.0,
                              0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,2.0,4.0,6.0,8.0,10.0,0.0,
                              0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,2.0,4.0,6.0,8.0,10.0,0.0])
        #
        hspc=torch.as_tensor([0.0,
                              0.0,                                                                     0.0,
                              0.0,0.0,                                        -1.0,-2.0,-3.0,-4.0,-5.0,0.0,
                              0.0,0.0,                                        -1.0,-2.0,-3.0,-4.0,-5.0,0.0,
                              0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-1.0,-2.0,-3.0,-4.0,-5.0,0.0,
                              0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-1.0,-2.0,-3.0,-4.0,-5.0,0.0,
                              0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-1.0,-2.0,-3.0,-4.0,-5.0,0.0])
        #
        gp2c=torch.as_tensor([0.0,
                              0.0,                                                                 0.0,
                              0.0,0.0,                                        0.0,1.5,4.5,6.5,10.0,0.0,
                              0.0,0.0,                                        0.0,1.5,4.5,6.5,10.0,0.0,
                              0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.5,4.5,6.5,10.0,0.0,
                              0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.5,4.5,6.5,10.0,0.0,
                              0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.5,4.5,6.5,10.0,0.0])
        #
        gppc=torch.as_tensor([0.0,
                              0.0,                                                                   0.0,
                              0.0,0.0,                                        0.0,-0.5,-1.5,-0.5,0.0,0.0,
                              0.0,0.0,                                        0.0,-0.5,-1.5,-0.5,0.0,0.0,
                              0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-0.5,-1.5,-0.5,0.0,0.0,
                              0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-0.5,-1.5,-0.5,0.0,0.0,
                              0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-0.5,-1.5,-0.5,0.0,0.0])

        #
        #heat of formation for each individual atom
        #experimental value, taken from block.f
        #unit kcal/mol
        eheat=torch.as_tensor([ 0.000,
                               52.102,                                                                                                                                     0.0,
                               38.410, 76.960,                                                                                  135.700, 170.890, 113.000, 59.559, 18.890, 0.0,
                               25.850, 35.000,                                                                                   79.490, 108.390,  75.570, 66.400, 28.990, 0.0,
                               21.420, 42.600, 90.300, 112.300, 122.300, 95.000, 67.700, 99.300, 102.400, 102.800, 80.700, 31.170, 65.400, 89.500, 72.300, 54.300, 26.740, 0.000,
                               19.600, 39.100, 101.500, 145.500, 172.400, 157.300, 0.000, 155.500, 133.000, 90.000, 68.100, 26.720, 58.000, 72.200, 63.200, 47.000, 25.517, 0.000,
                               18.700, 42.500, 0.000, 148.000, 186.900, 203.100, 185.000, 188.000, 160.000, 135.200, 88.000, 14.690, 43.550, 46.620, 50.100, 0.000, 0.000, 0.000])
        #
        #mass of atom
        mass=torch.as_tensor([ 0.00000,
                               1.00790,                                                                                                                                                                                   4.00260,
                               6.94000,   9.01218,                                                                                                                10.81000,  12.01100,  14.00670,  15.99940,  18.99840,  20.17900,
                              22.98977,  24.30500,                                                                                                                26.98154,  28.08550,  30.97376,  32.06000,  35.45300,  39.94800,
                              39.09800,  40.07800,  44.95600,  47.86700,  50.94200,  51.99600,  54.93800,  55.84500,  58.93300,  58.69300,  63.54600,  65.38000,  69.72300,  72.63000,  74.92200,  78.97100,  79.90400,  83.79800,
                              85.46800,  87.62000,  88.90600,  91.22400,  92.90600,  95.95000,  97.00000, 101.07000, 102.91000, 106.42000, 107.87000, 112.41000, 114.82000, 118.71000, 121.76000, 127.60000, 126.90000, 131.29000,
                             132.91000, 137.33000, 174.97000, 178.49000, 180.95000, 183.84000, 186.21000, 190.23000, 192.22000, 195.08000, 196.97000, 200.59000, 204.38000, 207.20000, 208.98000, 209.00000, 210.00000, 222.00000 ])



        self.atomic_num   = torch.nn.Parameter(atomic_num,   requires_grad=False)
        self.tore   = torch.nn.Parameter(tore,   requires_grad=False)
        self.iso   = torch.nn.Parameter(iso,   requires_grad=False)
        self.qn     = torch.nn.Parameter(qn,     requires_grad=False)
        self.qn_int = torch.nn.Parameter(qn_int, requires_grad=False)
        self.qnD_int = torch.nn.Parameter(qnD_int, requires_grad=False)
        self.ussc   = torch.nn.Parameter(ussc,   requires_grad=False)
        self.uppc   = torch.nn.Parameter(uppc,   requires_grad=False)
        self.gssc   = torch.nn.Parameter(gssc,   requires_grad=False)
        self.gspc   = torch.nn.Parameter(gspc,   requires_grad=False)
        self.hspc   = torch.nn.Parameter(hspc,   requires_grad=False)
        self.gp2c   = torch.nn.Parameter(gp2c,   requires_grad=False)
        self.gppc   = torch.nn.Parameter(gppc,   requires_grad=False)
        self.eheat  = torch.nn.Parameter(eheat/ev_kcalpmol,  requires_grad=False)
        self.mass   = torch.nn.Parameter(mass,   requires_grad=False)
        self.do_timing = False
        if self.do_timing:
            self.timing = {"Hcore + STO Integrals" : [],
                           "SCF"                   : [],
                           "Force"                 : [],
                           "MD"                    : [],
                           "D*"                    : [],
                           "CIS/RPA"               : [],
                          }


    def forward(self):
        pass
        """
        return self.length_conversion_factor, \
               self.energy_conversion_factor, \
               self.tore, \
               self.qn, \
               self.qn_int
        """
