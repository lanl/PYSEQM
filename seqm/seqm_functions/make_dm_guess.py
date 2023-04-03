import torch
from torch.autograd import grad
from .fock import fock
from .fock_u_batch import fock_u_batch
from .hcore import hcore
from .energy import elec_energy
from .SP2 import SP2
from .fermi_q import Fermi_Q
from .G_XL_LR import G
from seqm.seqm_functions.canon_dm_prt import Canon_DM_PRT
from seqm.basics import Pack_Parameters
from .pack import *
from .diag import DEGEN_EIGENSOLVER, degen_symeig, pytorch_symeig#sym_eig_trunc, sym_eig_trunc1, pseudo_diag
import warnings
import time


CHECK_DEGENERACY = False


def make_dm_guess(molecule, seqm_parameters, mix_homo_lumo=False, mix_coeff=0.4, learned_parameters=dict(), overwrite_existing_dm=False):
    sym_eigh = degen_symeig.apply if DEGEN_EIGENSOLVER else pytorch_symeig
    packpar = Pack_Parameters(seqm_parameters).to(molecule.coordinates.device)
    
    if callable(learned_parameters):
        adict = learned_parameters(molecule.species, molecule.coordinates)
        parameters = packpar(molecule.Z, learned_params=adict)    
    else:
        parameters = packpar(molecule.Z, learned_params=learned_parameters)
    
    beta = torch.cat((parameters['beta_s'].unsqueeze(1), parameters['beta_p'].unsqueeze(1)), dim=1)
    Kbeta = parameters.get('Kbeta', None)
    zetas = parameters['zeta_s']
    zetap = parameters['zeta_p']
    uss = parameters['U_ss']
    upp = parameters['U_pp']
    gss = parameters['g_ss']
    gsp = parameters['g_sp']
    gpp = parameters['g_pp']
    gp2 = parameters['g_p2']
    hsp = parameters['h_sp']

    nmol = molecule.nHeavy.shape[0]
    tore = molecule.const.tore
    
    M, w = hcore(molecule.const, nmol, molecule.molsize, molecule.maskd, molecule.mask, molecule.idxi, molecule.idxj, molecule.ni,molecule.nj,molecule.xij,molecule.rij, molecule.Z, \
                     zetas,
                     zetap,
                     uss,
                     upp,
                     gss,
                     gpp,
                     gp2,
                     hsp,
                     beta,
                     Kbeta=Kbeta)
    
    if not torch.is_tensor(molecule.dm) or overwrite_existing_dm==True:
        print('Reinitializing DM')
        P0 = torch.zeros_like(M)  # density matrix
        P0[molecule.maskd[molecule.Z>1],0,0] = tore[molecule.Z[molecule.Z>1]]/4.0
        P0[molecule.maskd,1,1] = P0[molecule.maskd,0,0]
        P0[molecule.maskd,2,2] = P0[molecule.maskd,0,0]
        P0[molecule.maskd,3,3] = P0[molecule.maskd,0,0]
        P0[molecule.maskd[molecule.Z==1],0,0] = 1.0
        #print('P0:\n', P0)
        #P0 += torch.randn(P0.shape,dtype=P0.dtype, device=P0.device)*0.01
        P = P0.reshape(nmol, molecule.molsize, molecule.molsize, 4, 4) \
            .transpose(2, 3) \
            .reshape(nmol, 4*molecule.molsize, 4*molecule.molsize)
        if molecule.nocc.dim() == 2:
            P = torch.stack((0.5 * P, 0.5 * P), dim=1)
        molecule.dm = P
    if molecule.nocc.dim() == 2:
        P = molecule.dm
        if mix_homo_lumo:
            x = fock_u_batch(nmol, molecule.molsize, P, M, molecule.maskd, molecule.mask, molecule.idxi, molecule.idxj, w, gss, gpp, gsp, gp2, hsp)
            Hcore = M.reshape(nmol, molecule.molsize, molecule.molsize, 4, 4) \
                     .transpose(2, 3) \
                     .reshape(nmol, 4*molecule.molsize, 4*molecule.molsize)
            
            # modified sym_eig_trunc below:
            dtype = x.dtype
            device = x.device
            
            nheavyatom = molecule.nHeavy.repeat_interleave(2)
            nH = molecule.nHydro.repeat_interleave(2)
            nocc = molecule.nocc.flatten()
            #Gershgorin circle theorem estimate upper bounds of eigenvalues  
            x_orig_shape = x.size()
            x0 = pack(x, nheavyatom, nH)
            nmol, size, _ = x0.shape
            
            aii = x0.diagonal(dim1=1, dim2=2)
            ri = torch.sum(torch.abs(x0), dim=2)-torch.abs(aii)
            hN = torch.max(aii + ri, dim=1)[0]
            dE = hN - torch.min(aii - ri, dim=1)[0] #(maximal - minimal) get range
            norb = nheavyatom * 4 + nH
            pnorb = size - norb
            nn = torch.max(pnorb).item()
            dx = 0.005
            mutipler = torch.arange(1.0+dx, 1.0+nn*dx+dx, dx, dtype=dtype, device=device)[:nn]
            ind = torch.arange(size, dtype=torch.int64, device=device)
            cond = pnorb>0
            for i in range(nmol):
                if cond[i]:
                    x0[i,ind[norb[i]:], ind[norb[i]:]] = mutipler[:pnorb[i]]*dE[i]+hN[i]
            try:
                e0, v = sym_eigh(x0)
            except:
                if torch.isnan(x0).any(): print(x0)
                #print(x0.detach().data.numpy())
                e0, v = sym_eigh(x0)
            e = torch.zeros((nmol, x.shape[-1]), dtype=dtype, device=device)
            e[...,:size] = e0
            for i in range(nmol):
                if cond[i]: e[i,norb[i]:size] = 0.0
            
            # $$$ the code below can and SHOULD be optimized. Too many reshapes
            
            #print(v.shape)
            e = e.reshape(x_orig_shape[0:3])
            v = v.reshape(int(v.shape[0]/2), 2, v.shape[1], v.shape[2])
            #print(v.shape)
            
            v_lumo = v[:,0].gather(2, molecule.nocc[:,0].unsqueeze(0).unsqueeze(0).T.repeat(1,v.shape[-1],1))
            v_homo = v[:,0].gather(2, molecule.nocc[:,0].unsqueeze(0).unsqueeze(0).T.repeat(1,v.shape[-1],1)-1)

            mix_coeff = torch.tensor([mix_coeff], device=device)
            
            v_a_homo = (1 - mix_coeff) * v_homo + (mix_coeff) * v_lumo
            #v_a_lumo = -(mix_coeff)*v_homo + (1-mix_coeff)*v_lumo

            #v_b_homo = (1-mix_coeff)*v_homo - torch.sin(mix_coeff)*v_lumo
            #v_b_lumo =  (mix_coeff)*v_homo + (1-mix_coeff)*v_lumo

            v[:,0].scatter_(2, molecule.nocc[:,0].unsqueeze(0).unsqueeze(0).T.repeat(1,v.shape[-1],1)-1, v_a_homo)
            #v[:,0].scatter_(2, molecule.nocc[:,0].unsqueeze(0).unsqueeze(0).T.repeat(1,v.shape[-1],1), v_a_lumo)

            #v[:,1].scatter_(2, molecule.nocc[:,1].unsqueeze(0).unsqueeze(0).T.repeat(1,v.shape[-1],1)-1, v_b_homo)
            #v[:,1].scatter_(2, molecule.nocc[:,1].unsqueeze(0).unsqueeze(0).T.repeat(1,v.shape[-1],1), v_b_lumo)
            
            v = v.reshape(int(v.shape[0]*2),v.shape[2],v.shape[3])
            
            if CHECK_DEGENERACY:
                t = torch.stack(list(map(lambda a,b,n : construct_P(a, b, n), e, v, nocc)))
            else:
                #list(map(lambda a,n : print('norm', torch.norm(v, dim=0), n), v, nocc))
                #print(torch.norm())
                t = 2.0*torch.stack(list(map(lambda a,n : torch.matmul(a[:,:n], a[:,:n].transpose(0,1)), v, nocc)))

            # print(t.shape)
            # print(nheavyatom.shape)
            # print(nH.shape)
            # print(x.shape)
            P = unpack(t, nheavyatom, nH, x.shape[-1])
            
            v = v.reshape(int(v.shape[0]/2),2,v.shape[1],v.shape[2])
            P = P.reshape(x_orig_shape)
            molecule.dm = P / 2
            return P, v
        else:
            return P, None
    else:
        return P, None
    
