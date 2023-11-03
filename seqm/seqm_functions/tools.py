import torch
from .diat_overlap import diatom_overlap_matrix
from .diat_overlapD import diatom_overlap_matrixD
from .diat_overlap_PM6_SP import diatom_overlap_matrix_PM6_SP
from .two_elec_two_center_int import two_elec_two_center_int as TETCI
from .constants import overlap_cutoff
import time


def get_hcore(molecule):

    
    
    
    nmol = nHeavy.shape[0]
    
    M, w,rho0xi,rho0xj = hcore(const, nmol, molsize, maskd, mask, idxi, idxj, ni,nj,xij,rij, Z, \
                     zetas,zetap, zetad, zs, zp, zd,  uss, upp , udd, gss, gpp, gp2, hsp,F0SD, G2SD, rho_core, alpha, chi, themethod,  beta, Kbeta=Kbeta)
    


def hcore(const,nmol, molsize, maskd, mask, idxi, idxj, ni,nj,xij,rij, Z, zetas, zetap, zetad, zs, zp, zd,  uss, upp , udd,  gss, gpp, gp2, hsp, F0SD, G2SD,  rho_core, alpha, chi, themethod, beta, Kbeta=None):
    """
    Get Hcore and two electron and two center integrals
    """
    #t0 = time.time()
    dtype = xij.dtype
    device = xij.device
    qn_int = const.qn_int
    qnD_int = const.qnD_int
    # pair type tensor: idxi, idxj, ni,nj,xij,rij, mask (batch dim is pair)
    # atom type tensor: Z, zetas,zetap, uss, upp , gss, gpp, gp2, hsp, beta(isbeta_pair=False)
    #                   (batch dim is atom)
    #
    # nmol, number of molecules in this batch
    #ntotatoms = nmol * molsize, i.e. the padding zero is also included
    #will call diat and rotate to create overlap matrix and two electron two
    #center integrals
    #and return Hcore and structured two electron two center integrals

    #molsize : number of atoms in each molecule, including the padding zero
    #mask: tell the postion of each pair, shape (npairs,)

    #idxi, idxj, index for atom i and j in the current batch, shape (nparis,)
    #in the data_loader.py, the index for for each atom is the index across whole dataset
    #should take the remainder before passing into this funcition %(batch_size*molsize)

    #ni, nj atomic number, shape (npairs,)
    #xij, unit vector from i to j (xj-xi)/|xj-xi|, shape (npairs,3)
    #rij, distance between i and j, in atomic units, shape (npairs,)
    # Z, atomic number, shape (ntotatoms,)
    #zetas,zetap: zeta parameters for s and p orbitals, shape(ntotatoms,)
    #will use it to create zeta_a and zeta_b
    #zeta_a, zeta_b: zeta for atom i and j, shape (npairs,2), for s and p orbitals
    # uss, upp: Uss Upp energy for each atom, shape (ntotatoms,)
    #gss, gpp, gp2, hsp: parameters, shape (ntotatoms,)

    #isbeta_pair : beta is for each pair in the molecule, shape (npairs, 4) or
    #              for each atom in the molecule, shape (ntotatoms, 2)
    #              check diat.py for detail

    #calpar will create dd, qq, rho0, rho1, rho2 used in rotate from zetas, zetap
    # and qn, gss, hsp, hpp (hpp = 0.5*(gpp-gp2))
    # all the hpp in the code is replaced with gpp and gp2, and thus not used

    #qn : principal quantum number for valence shell
    #tore: charge for the valence shell of each atom, will used as constants

    #rotate(ni,nj,xij,rij,tore,da,db, qa,qb, rho0a,rho0b, rho1a,rho1b, rho2a,rho2b) => w, e1b, e2a
    #h1elec(idxi, idxj, ni, nj, xij, rij, zeta_a, zeta_b, beta, ispair=False) =>  beta_mu_nu

    #use uss upp to the diagonal block for hcore
    if(themethod == 'PM6'):
        zeta = torch.cat((zetas.unsqueeze(1), zetap.unsqueeze(1), zetad.unsqueeze(1)),dim=1)
    else:
        zeta = torch.cat((zetas.unsqueeze(1), zetap.unsqueeze(1)),dim=1)
    overlap_pairs = rij<=overlap_cutoff
    #di=th.zeros((npairs,4,4),dtype=dtype, device=device)

    if(themethod == 'PM6'):
        di = torch.zeros((xij.shape[0], 9, 9),dtype=dtype, device=device)
        di[overlap_pairs] = diatom_overlap_matrixD(ni[overlap_pairs],
                                   nj[overlap_pairs],
                                   xij[overlap_pairs],
                                   rij[overlap_pairs],
                                   zeta[idxi][overlap_pairs],
                                   zeta[idxj][overlap_pairs],
                                   qn_int, qnD_int)
    elif themethod == 'PM6_SP':
        di = torch.zeros((xij.shape[0], 4, 4),dtype=dtype, device=device)
        di[overlap_pairs] = diatom_overlap_matrix_PM6_SP(ni[overlap_pairs],
                                   nj[overlap_pairs],
                                   xij[overlap_pairs],
                                   rij[overlap_pairs],
                                   zeta[idxi][overlap_pairs],
                                   zeta[idxj][overlap_pairs],
                                   qn_int)
    
    else:
        di = torch.zeros((xij.shape[0], 4, 4),dtype=dtype, device=device)
        di[overlap_pairs] = diatom_overlap_matrix_PM6_SP(ni[overlap_pairs],
                                   nj[overlap_pairs],
                                   xij[overlap_pairs],
                                   rij[overlap_pairs],
                                   zeta[idxi][overlap_pairs],
                                   zeta[idxj][overlap_pairs],
                                   qn_int)

#    print("DI:", time.time() - t0)
    t0 = time.time()
    #print('DOING 2c2e time test')
    #di shape (npairs,4,4)
    #print("HI")
    
    w, e1b, e2a,rho0xi,rho0xj = TETCI(const, idxi, idxj, ni, nj, xij, rij, Z, zetas, zetap, zetad, zs, zp, zd,  gss, gpp, gp2, hsp, F0SD, G2SD, rho_core, alpha, chi, themethod)
    #print("TETC:", time.time() - t0)
#    sys.exit()
    #w shape (napirs, 10,10)
    #e1b, e2a shape (npairs, 10)
    #di shape (npairs,4,4), unit eV, core part for AO on different centers(atoms)
    ntotatoms = nmol * molsize
    if(themethod == 'PM6'):
        M = torch.zeros(nmol*molsize*molsize,9,9,dtype=dtype,device=device)
    else:
        M = torch.zeros(nmol*molsize*molsize,4,4,dtype=dtype,device=device)

    #fill the upper triangle part
    #unlike the mopac, which fills the lower triangle part
    #Hcore is symmetric
    #diagonal part in Hcore
    #t1 = torch.tensor([i*molsize+i for i in range(molsize)],dtype=dtypeint,device=device).reshape((1,-1))
    #t2 = torch.tensor([i*molsize**2 for i in range(nmol)],dtype=dtypeint,device=device).reshape((-1,1))
    #maskd = (t1+t2).reshape(-1) # mask for diagonal blocks
    #M[...,0,0].index_add_(0,maskd,uss)
    if(themethod == 'PM6'):
        M[maskd,0,0] = uss
        M[maskd,1,1] = upp
        M[maskd,2,2] = upp
        M[maskd,3,3] = upp
        M[maskd,4,4] = udd
        M[maskd,5,5] = udd
        M[maskd,6,6] = udd
        M[maskd,7,7] = udd
        M[maskd,8,8] = udd
        M.index_add_(0,maskd[idxj], e1b)
        M.index_add_(0,maskd[idxi], e2a)

    else:
        M[maskd,0,0] = uss
        M[maskd,1,1] = upp
        M[maskd,2,2] = upp
        M[maskd,3,3] = upp
        M.index_add_(0,maskd[idxi], e1b)
        M.index_add_(0,maskd[idxj], e2a)

    #warning, as for all the pairs, ni>=nj, or idxi<idxj, i.e, pairs are not symmetric
    #so in the summation below, there is no need to divide by 2
    # V_{mu,nv,B} = -ZB*(mu^A nv^A, s^B s^B), stored on e1b, e2a
    # \sum_B V_{ss,B}
    # e1b ==> V_{,B} E1B = ELECTRON ON ATOM NI ATTRACTING NUCLEUS OF NJ.
    # e2a ==> V_{,A}
    #e1b, e2a order: (s s/), (px s/), (px px/), (py s/), (py px/), (py py/)
    #                (pz s/), (pz px/) (pz py/) (pz pz/)

    #diagonal block, elecron nuclear interation
    """

    #(s s)
    M[...,0,0].index_add_(0,maskd[idxi], e1b[...,0])
    M[...,0,0].index_add_(0,maskd[idxj], e2a[...,0])
    #(px s/s s) = (s px/s s)
    M[...,0,1].index_add_(0,maskd[idxi], e1b[...,1])
    M[...,0,1].index_add_(0,maskd[idxj], e2a[...,1])
    #(px px/)
    M[...,1,1].index_add_(0,maskd[idxi], e1b[...,2])
    M[...,1,1].index_add_(0,maskd[idxj], e2a[...,2])
    #(py s/)
    M[...,0,2].index_add_(0,maskd[idxi], e1b[...,3])
    M[...,0,2].index_add_(0,maskd[idxj], e2a[...,3])
    #(py px/)
    M[...,1,2].index_add_(0,maskd[idxi], e1b[...,4])
    M[...,1,2].index_add_(0,maskd[idxj], e2a[...,4])
    #(py py/)
    M[...,2,2].index_add_(0,maskd[idxi], e1b[...,5])
    M[...,2,2].index_add_(0,maskd[idxj], e2a[...,5])
    #(pz s/)
    M[...,0,3].index_add_(0,maskd[idxi], e1b[...,6])
    M[...,0,3].index_add_(0,maskd[idxj], e2a[...,6])
    #(pz px/)
    M[...,1,3].index_add_(0,maskd[idxi], e1b[...,7])
    M[...,1,3].index_add_(0,maskd[idxj], e2a[...,7])
    #(pz py/)
    M[...,2,3].index_add_(0,maskd[idxi], e1b[...,8])
    M[...,2,3].index_add_(0,maskd[idxj], e2a[...,8])
    #(pz pz/)
    M[...,3,3].index_add_(0,maskd[idxi], e1b[...,9])
    M[...,3,3].index_add_(0,maskd[idxj], e2a[...,9])
    """
    #e1b, e2a are reshaped to be (...,4,4) in rotate.py

    if(themethod == 'PM6'):
        #di = di.transpose(1,2)
        #print(di)
        if torch.is_tensor(Kbeta):
            M[mask,0,0]   = di[...,0,0]*(beta[idxi,0]+beta[idxj,0])/2.0* Kbeta[:,0]
            M[mask,0,1:4]  = di[...,0,1:4]*(beta[idxi,0:1]+beta[idxj,1:2])/2.0*Kbeta[:,1,None]
            M[mask,1:4,0]  = di[...,1:4,0]*(beta[idxi,1:2]+beta[idxj,0:1])/2.0*Kbeta[:,2,None]
            M[mask,1:4,1:4] = di[...,1:4,1:4]*(beta[idxi,1:2,None]+beta[idxj,1:2,None])/2.0** Kbeta[:,3:,None]

            M[mask,0,4:]  = di[...,0,4:]*(beta[idxi,0:1]+beta[idxj,2:3])/2.0
            M[mask,4:,0]  = di[...,4:,0]*(beta[idxi,2:3]+beta[idxj,0:1])/2.0


            M[mask,1:4,4:] = di[...,1:4,4:]*(beta[idxi,1:2,None]+beta[idxj,2:3,None])/2.0
            M[mask,4:,1:4] = di[...,4:,1:4]*(beta[idxi,2:3,None]+beta[idxj,1:2,None])/2.0

            M[mask,4:,4:] =  di[...,4:,4:]*(beta[idxi,2:3,None]+beta[idxj,2:3,None])/2.0


        else:
            M[mask,0,0]   = di[...,0,0]*(beta[idxi,0]+beta[idxj,0])/2.0
            M[mask,0,1:4]  = di[...,0,1:4]*(beta[idxi,0:1]+beta[idxj,1:2])/2.0
            M[mask,1:4,0]  = di[...,1:4,0]*(beta[idxi,1:2]+beta[idxj,0:1])/2.0
            M[mask,1:4,1:4] = di[...,1:4,1:4]*(beta[idxi,1:2,None]+beta[idxj,1:2,None])/2.0

            M[mask,0,4:]  = di[...,0,4:]*(beta[idxi,0:1]+beta[idxj,2:3])/2.0
            M[mask,4:,0]  = di[...,4:,0]*(beta[idxi,2:3]+beta[idxj,0:1])/2.0


            M[mask,1:4,4:] = di[...,1:4,4:]*(beta[idxi,1:2,None]+beta[idxj,2:3,None])/2.0
            M[mask,4:,1:4] = di[...,4:,1:4]*(beta[idxi,2:3,None]+beta[idxj,1:2,None])/2.0

            M[mask,4:,4:] =  di[...,4:,4:]*(beta[idxi,2:3,None]+beta[idxj,2:3,None])/2.0


    else:
        if torch.is_tensor(Kbeta):
            M[mask,0,0]   = di[...,0,0]*(beta[idxi,0]+beta[idxj,0])/2.0 * Kbeta[:,0]
            M[mask,0,1:]  = di[...,0,1:]*(beta[idxi,0:1]+beta[idxj,1:2])/2.0 * Kbeta[:,1,None]
            M[mask,1:,0]  = di[...,1:,0]*(beta[idxi,1:2]+beta[idxj,0:1])/2.0 * Kbeta[:,2,None]
            M[mask,1:,1:] = di[...,1:,1:]*(beta[idxi,1:2,None]+beta[idxj,1:2,None])/2.0 * Kbeta[:,3:,None]
            #raise ValueError('Kbeta for each pair is not implemented yet')
        else:
            #beta is for each atom in the molecules, shape (ntotatoms,2)
            M[mask,0,0]   = di[...,0,0]*(beta[idxi,0]+beta[idxj,0])/2.0
            M[mask,0,1:]  = di[...,0,1:]*(beta[idxi,0:1]+beta[idxj,1:2])/2.0
            M[mask,1:,0]  = di[...,1:,0]*(beta[idxi,1:2]+beta[idxj,0:1])/2.0
            M[mask,1:,1:] = di[...,1:,1:]*(beta[idxi,1:2,None]+beta[idxj,1:2,None])/2.0

        #caution
        #the lower triangle part is not filled here

    """
    #it is easier to use M for the construction of fock matrix with density matrix

    Hcore = M.reshape(nmol,molsize,molsize,4,4) \
             .transpose(2,3) \
             .reshape(nmol, 4*molsize, 4*molsize) #\
             #.contiguous().clone()
    #not sure if need contiguous, clone

    #what will be used in SCF is Hcore and w
    return Hcore, w
    #"""
    #"""
    ##print(di[...,1:4,4:])
    ##print(di[...,4:,1:4])
    ##print(di[...,4:,4:] )
    #print(M[mask,:,:])
    ##print("TOTAL HCORE+INTEGRALS: ", time.time()-t0)
    ##sys.exit()
    ##print(di[...,4:,4:]*(beta[idxi,1:2,None]+beta[idxj,1:2,None])/2.0)
    #print("HCORE:", time.time() - t0)
    return M, w,rho0xi,rho0xj