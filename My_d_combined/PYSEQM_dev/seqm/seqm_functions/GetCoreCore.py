import torch
from .constants import ev
import sys
import numpy
import math


#core/core repulsions
def GetCoreCore(rij, xij,ni,nj, const, rho0xi,rho0xj, alpha, chi):
    print(1111111111)
    corecore = torch.zeros(xij.shape[0], 9, 9)
    expo2 = const.tore[ni]*const.tore[nj]*(1.0+chi[nj,nj]*torch.pow(math.e,-alpha[ni,nj]*rij+0.0003*torch.pow(rij,6)))*1/torch.sqrt(rij*rij+(rho0xi*rho0xi+rho0xj*rho0xj))
    print(ev*expo2)
    sys.exit()


    return corecore
