import torch

# it is better to define mask as the same way defining maskd
# as it will be better to do summation using the representation of P in

def fock_u_batch(nmol, molsize, P0, M, maskd, mask, idxi, idxj, w, gss, gpp, gsp, gp2, hsp):
    """
    construct fock matrix
    """
    dtype = M.dtype
    device = M.device
    # P0 : total density matrix, P0 = Palpha + Pbeta, Palpha==Pbeta,
    #     shape (nmol, 4*molsize, 4*molsize)
    #     for closed shell molecule only, RHF is used, alpha and beta has same WF
    # M : Hcore in the shape of (nmol*molsize**2,4,4)
    # to construct Hcore from M, check hcore.py
    # Hcore = M.reshape(nmol,molsize,molsize,4,4) \
    #          .transpose(2,3) \
    #          .reshape(nmol, 4*molsize, 4*molsize)

    # maskd : mask for diagonal block for M, shape(ntotatoms,)
    # M[maskd] take out the diagonal block
    # gss, gpp, gsp, shape (ntotatoms, )
    # P0: shape (nmol, 4*molsize, 4*molsize)
    
    P = (P0[:,0]+P0[:,1]).reshape((nmol,molsize,4,molsize,4)) \
          .transpose(2,3).reshape(nmol*molsize*molsize,4,4)
        
    PAlpha_ = P0.transpose(0,1).reshape((2,nmol,molsize,4,molsize,4)) \
          .transpose(3,4).reshape(2,nmol*molsize*molsize,4,4)
    
    
    #at this moment,  P has the same shape as M, as it is more convenient
    # to use here
    # while for diagonalization, may have to reshape

    # for the diagonal block, the summation over ortitals on the same atom in Fock matrix
    F_ = M.expand(2,-1,-1,-1).clone()

    Pptot = P[...,1,1]+P[...,2,2]+P[...,3,3]
    PAlpha_ptot_ = PAlpha_[...,1,1]+PAlpha_[...,2,2]+PAlpha_[...,3,3]

    #  F_mu_mu = Hcore + \sum_nu^A P_nu_nu (g_mu_nu - 0.5 h_mu_nu) + \sum^B
    """
    #(s,s)
    F[maskd,0,0].add_( 0.5*P[maskd,0,0]*gss + Pptot[maskd]*(gsp-0.5*hsp) )

    for i in range(1,4):
        #(p,p)
        F[maskd,i,i].add_( P[maskd,0,0]*(gsp-0.5*hsp) + 0.5*P[maskd,i,i]*gpp \
                        + (Pptot[maskd] - P[maskd,i,i]) * (1.25*gp2-0.25*gpp) )
        #(s,p) = (p,s) upper triangle
        F[maskd,0,i].add_( P[maskd,0,i]*(1.5*hsp - 0.5*gsp) )

    #(p,p*)
    for i,j in [(1,2),(1,3),(2,3)]:
        F[maskd,i,j].add_( P[maskd,i,j]* (0.75*gpp - 1.25*gp2) )
    #
    """
    ### http://openmopac.net/manual/1c2e.html
    #(s,s)
    TMP_ = torch.zeros_like(F_)
    TMP_[:,maskd,0,0] = PAlpha_[[1,0]][:,maskd,0,0]*gss + Pptot[maskd]*gsp-PAlpha_ptot_[:,maskd]*hsp
    for i in range(1,4):
        #(p,p)
        TMP_[:,maskd,i,i] = P[maskd,0,0]*gsp-PAlpha_[:,maskd,0,0]*hsp + PAlpha_[[1,0]][:,maskd,i,i]*gpp \
                        +(Pptot[maskd]-P[maskd,i,i])*gp2 - 0.5*(PAlpha_ptot_[:,maskd]-PAlpha_[:,maskd,i,i])*(gpp-gp2)
        
        #(s,p) = (p,s) upper triangle
        TMP_[:,maskd,0,i] = 2*P[maskd,0,i]*hsp - PAlpha_[:,maskd,0,i]*(hsp+gsp)
        
    #(p,p*)
    for i,j in [(1,2),(1,3),(2,3)]:
        TMP_[:,maskd,i,j] = P[maskd,i,j] * (gpp - gp2) - 0.5*PAlpha_[:,maskd,i,j]*(gpp + gp2)

    #print('1c-2e\n', TMP)
    F_.add_(TMP_)
    
    ###############
    
    
    
    # sumation over two electron two center integrals over the neighbor atoms

    #for the diagonal block, check JAB in fock2.f
    #F_mu_nv = Hcore + \sum^B \sum_{lambda, sigma} P^B_{lambda, sigma} * (mu nu, lambda sigma)
    #as only upper triangle part is done, and put in order
    # (ss ), (px s), (px px), (py s), (py px), (py py), (pz s), (pz px), (pz py), (pz pz)
    #weight for them are
    #  1       2       1        2        2        1        2       2        2       1

    weight = torch.tensor([1.0,
                           2.0, 1.0,
                           2.0, 2.0, 1.0,
                           2.0, 2.0, 2.0, 1.0],dtype=dtype, device=device).reshape((-1,10))
    #
    #P[maskd[idxi]] : P^tot_{mu,nu \in A} shape (npairs, 4,4)
    #P[maskd[idxj]] : P^tot_{mu,nu \in B} shape (npairs, 4,4)

    #take out the upper triangle part in the same order as in W
    #shape (nparis, 10)

    PA = (P[maskd[idxi]][...,(0,0,1,0,1,2,0,1,2,3),(0,1,1,2,2,2,3,3,3,3)]*weight).reshape((-1,10,1))
    PB = (P[maskd[idxj]][...,(0,0,1,0,1,2,0,1,2,3),(0,1,1,2,2,2,3,3,3,3)]*weight).reshape((-1,1,10))


    #suma \sum_{mu,nu \in A} P_{mu, nu in A} (mu nu, lamda sigma) = suma_{lambda sigma \in B}
    #suma shape (npairs, 10)
    suma = torch.sum(PA*w,dim=1)
    #sumb \sum_{l,s \in B} P_{l, s inB} (mu nu, l s) = sumb_{mu nu \in A}
    #sumb shape (npairs, 10)
    sumb = torch.sum(PB*w,dim=2)
    #print('suma:\n',suma)
    #reshape back to (npairs 4,4)
    # as will use index add in the following part
    sumA = torch.zeros(w.shape[0],4,4,dtype=dtype, device=device)
    sumB = torch.zeros_like(sumA)
    
    sumA[...,(0,0,1,0,1,2,0,1,2,3),(0,1,1,2,2,2,3,3,3,3)] = suma # ^^^^^^^
    sumB[...,(0,0,1,0,1,2,0,1,2,3),(0,1,1,2,2,2,3,3,3,3)] = sumb # ^^^^^^^
    
    # sumA = sumA.unsqueeze(0)
    # sumB = sumB.unsqueeze(0)
    
    torch.use_deterministic_algorithms(True)

    #F^A_{mu, nu} = Hcore + \sum^A + \sum_{B} \sum_{l, s \in B} P_{l,s \in B} * (mu nu, l s)
    
    # $$$ index_add_ below could be done in a more efficient way
    #\sum_A
    #F_.index_add_(1, maskd[idxj], sumA)
    
    F_[0].index_add_(0, maskd[idxj], sumA)
    F_[1].index_add_(0, maskd[idxj], sumA)
    
    
    #\sum_B
    #F_.index_add_(1, maskd[idxi], sumB)
    F_[0].index_add_(0, maskd[idxi], sumB)
    F_[1].index_add_(0, maskd[idxi], sumB)


     ###################

    # off diagonal block part, check KAB in forck2.f
    # mu, nu in A
    # lambda, sigma in B
    # F_mu_lambda = Hcore - 0.5* \sum_{nu \in A} \sum_{sigma in B} P_{nu, sigma} * (mu nu, lambda, sigma)
    sum_ = torch.zeros(2, w.shape[0],4,4,dtype=dtype, device=device) # ^^^^^

    # (ss ), (px s), (px px), (py s), (py px), (py py), (pz s), (pz px), (pz py), (pz pz)
    #   0,     1         2       3       4         5       6      7         8        9
    ind = torch.tensor([[0,1,3,6],
                        [1,2,4,7],
                        [3,4,5,8],
                        [6,7,8,9]],dtype=torch.int64, device=device)
    # Pp =P[mask], P_{mu \in A, lambda \in B}
    #Pp = -0.5*P[mask]
    for i in range(4):
        for j in range(4):
            #\sum_{nu \in A} \sum_{sigma \in B} P_{nu, sigma} * (mu nu, lambda, sigma)
            sum_[...,i,j] = torch.sum(-PAlpha_[:,mask]*w[...,ind[i],:][...,:,ind[j]],dim=(2,3))

            
    #
    F_.index_add_(1,mask,sum_)
    
    torch.use_deterministic_algorithms(False)
    
    
    ####################
    
    F0_ = F_.reshape(2, nmol, molsize,molsize,4,4) \
             .transpose(3,4) \
             .reshape(2, nmol, 4*molsize, 4*molsize).transpose(0,1)
    
    #
    F0_.add_(F0_.triu(1).transpose(2,3))

    return F0_
