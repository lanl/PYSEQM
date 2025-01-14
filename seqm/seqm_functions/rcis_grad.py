import torch
from seqm.seqm_functions.rcis import makeA_pi

def rcis_grad(mol, amp, w, e_mo):

    device = amp.device
    dtype = amp.dtype
    norb = mol.norb[0]
    nocc = mol.nocc[0]
    nvirt = norb - nocc

    # CIS unrelaxed density B = \sum_iab C_\mu a * t_ai * t_bi * C_\nu b - \sum_ija C_\mu i * t_ai * t_aj * C_\nu j 
    C = mol.eig_vec 
    Cocc = C[0,:,:nocc]
    Cvirt = C[0,:,nocc:norb]
    # occ = slice(None,nocc)
    # virt = slice(nocc,norb)
    amp_ia = amp.view(nocc,nvirt)

    dens_BR = torch.empty(2,norb,norb,device=device,dtype=dtype)

    B_virt  = torch.matmul(Cvirt,amp_ia.T)
    B_occ  = torch.matmul(Cocc,amp_ia)
    dens_BR[0] = torch.matmul(B_virt,B_virt.T) - torch.matmul(B_occ,B_occ.T)
    del B_occ, B_virt

    # CIS transition density R = \sum_ia C_\mu i * t_ia * C_\nu a 
    dens_BR[1] = torch.einsum('mi,ia,na',Cocc,amp_ia,Cvirt)

    # Calculate z-vector
    
    # make RHS of the CPSCF equation:
    BR_pi = makeA_pi(mol,dens_BR,w)*2.0
    RHS = -torch.einsum('ma,mn,ni->ai',Cvirt,BR_pi[0],Cocc)
    RHS -= torch.einsum('ma,mn,nb,ib->ai',Cvirt,BR_pi[1],Cvirt,amp_ia)
    RHS += torch.einsum('ja,mj,mn,ni->ai',amp_ia,Cocc,BR_pi[1],Cocc)


    return


def make_A_times_zvector(mol, z, w, e_mo):
    # C: Molecule Orbital Coefficients
    C = mol.eig_vec
    nmol  = mol.nmol
    norb = mol.norb[0]
    nocc = mol.nocc[0]
    nvirt = norb - nocc
    ea_ei = e_mo[0,nocc:norb].unsqueeze(0)-e_mo[0,:nocc].unsqueeze(1)

    # From here I assume there is only one molecule. I'll worry about batching later
    if (nmol != 1):
        raise Exception("Not yet implemented for more than one molecule")

    Via = z.view(-1,nocc,nvirt) 
    P_xi = torch.einsum('mi,ria,na->rmn', C[0, :, :nocc],Via, C[0, :, nocc:])
    P_xi.add_(P_xi.transpose(1,2))
    
    F0 = makeA_pi(mol,P_xi,w,allSymmetric=True)

    # since I assume that we have only 1 molecule I'll use this index to get the data for the 1st molecule
    mol0 = 0

    # TODO: why am I multiplying by 2?
    A = torch.einsum('mi,rmn,na->ria', C[mol0, :, :nocc], F0,C[mol0,:, nocc: ])*2.0

    A += Via*ea_ei.unsqueeze(0)

    nnewRoots = z.shape[0]
    return A.view(nnewRoots,-1)
