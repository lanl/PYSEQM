import torch


def Spherical_Pot_Force(molecule, radius, k=1.0, center=(0.0, 0.0, 0.0)):
    """
    Spherical potential around zero-potential bubble.

    molecule - pyseqm molecule object
    center - [x,y,z] of the potential center
    radius - the minimum distance from the center when the potential starts to act
    k - Hooke's constant in E = 0.5*k*x^2
    """

    center = torch.as_tensor(center, dtype=molecule.coordinates.dtype, device=molecule.coordinates.device)

    displacement = molecule.coordinates - center
    distance = torch.linalg.vector_norm(displacement, dim=2, keepdim=True)
    extension = torch.clamp_min(distance - radius, 0.0)
    safe_distance = distance.clamp_min(torch.finfo(distance.dtype).tiny)
    dxdydz = displacement * (extension / safe_distance)
    force = -k * dxdydz

    E = 0.5 * k * torch.sum(torch.square(dxdydz), dim=(1, 2))

    return E, force
