import torch
from .diag import sym_eig_trunc, sym_eig_trunc1
from .pack import unpack

#@torch.jit.script
def Fermi_Q(H0,T, Nocc, nHeavy, nHydro, kB, scf_backward):
    '''
    Fermi operator expansion, eigenapirs [QQ,e], and entropy S_Ent
    '''
    #print('Doing Fermi_Q.')
    if H0.dtype == torch.float64:
        occ_tol=1e-9,
        entropy_eps=1e-14,
    elif H0.dtype == torch.float32:
        occ_tol=1e-5,
        entropy_eps=1e-7,
    else:
        raise RuntimeError

    N = max(H0.shape)
    Fe_vec = torch.zeros(N,dtype=H0.dtype,device=H0.device)

    if scf_backward>=1:
        e, QQ = sym_eig_trunc1(H0,nHeavy, nHydro, Nocc, eig_only=True)
    else:
        e, QQ = sym_eig_trunc( H0,nHeavy, nHydro, Nocc, eig_only=True)

    e = e[:,0:QQ.shape[-1]]
    mu0 = torch.zeros(e.shape)
    mu0 = (e.gather(1, Nocc.unsqueeze(0).T-1) + e.gather(1, Nocc.unsqueeze(0).T))/2
    OccErr = torch.ones(Nocc.shape)
    OccErr_mask = OccErr > occ_tol
    beta = 1./(kB*T) # Temp in Kelvin
    norb = nHeavy*4+nHydro

    Occ_mask =  torch.zeros(e.shape, device=H0.device, dtype=H0.dtype)

    while True in OccErr_mask:
        Occ = torch.zeros(Nocc.shape, device=H0.device, dtype=H0.dtype)
        # Occ_I = 1/(torch.exp(beta*(e-mu0)) +1.0)
        Occ_I = torch.sigmoid(-beta*(e-mu0))

        # $$$
        for i,j in zip(range(0,len(Occ_mask)), norb.unsqueeze(0).T):
            Occ_mask[i,0:j]=1

        Occ_mask = Occ_mask.clone()

        Occ_I = Occ_I*Occ_mask
        Fe_vec = Occ_I
        Occ = Occ + Occ_I.sum(1)
        dOcc = (beta*Occ_I*(1.0 - Occ_I)).sum(1)
        OccErr = torch.abs(Nocc-Occ)
        OccErr_mask = OccErr > occ_tol
        indices_of_high_errors = torch.nonzero(OccErr_mask)
        if True in OccErr_mask:
            mu0[indices_of_high_errors] += ((Nocc-Occ)/dOcc).unsqueeze(0).T[indices_of_high_errors]

    X  = QQ * Fe_vec.unsqueeze(1)
    D0 = X @ QQ.transpose(1, 2)
    D0 = 2*unpack(D0, nHeavy, nHydro, H0.shape[-1]) # bring to block form
    S = torch.zeros(Nocc.shape, device=H0.device, dtype=H0.dtype)
    S_temp =  - kB*(Fe_vec*torch.log(Fe_vec) + (1-Fe_vec)*torch.log(1-Fe_vec) )

    mask_S = (Fe_vec > entropy_eps) & ((1.0-Fe_vec) > entropy_eps)

    # $$$
    for i,j in zip(range(0,len(S)), mask_S):
        S[i] = S_temp[i, j].sum()
    # print('mu:', mu0, 'S_elec:', S_temp)

    return D0, S, QQ, e, Fe_vec, mu0, Occ_mask
