import torch
from seqm.basics import *

def Spherical_Pot_Force(molecule, radius, k=1.0, center=[0.0,0.0,0.0]):
    '''
    Spherical potential around zero-potential bubble.
    
    molecule - pyseqm molecule object
    center - [x,y,z] of the potential center
    radius - the minimum distance from the center when the potential starts to act
    k - Hooke's constant in E = 0.5*k*x^2
    '''
    
    center = torch.tensor(center, device=molecule.coordinates.device)
    
    r_from_center = torch.norm(molecule.coordinates-center, dim=2).unsqueeze(-1)
    force_mask = r_from_center > radius # find atoms beyond radius
    closest_point_on_sphere = center + radius*(molecule.coordinates - center)/r_from_center
    
    dxdydz = (molecule.coordinates - closest_point_on_sphere)*force_mask
    force = -k*dxdydz
    
    E = 0.5*k*torch.sum(torch.square(dxdydz), dim=(1,2))
    
    return E, force
