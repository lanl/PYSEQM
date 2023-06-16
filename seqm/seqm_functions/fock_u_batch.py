import torch
#from .build_two_elec_one_center_int import calc_integral
import time
# it is better to define mask as the same way defining maskd
# as it will be better to do summation using the representation of P in

def fock_u_batch(nmol, molsize, P0, M, maskd, mask, idxi, idxj, w, W, gss, gpp, gsp, gp2, hsp, themethod, zetas, zetap, zetad, Z, F0SD, G2SD):
    """
    construct fock matrix

    P0 : total density matrix, P0 = Palpha + Pbeta, Palpha==Pbeta,
        shape (nmol, 4*molsize, 4*molsize)
        for closed shell molecule only, RHF is used, alpha and beta has same WF
    M : Hcore in the shape of (nmol*molsize**2,4,4)
    to construct Hcore from M, check hcore.py
    Hcore = M.reshape(nmol,molsize,molsize,4,4) \
             .transpose(2,3) \
             .reshape(nmol, 4*molsize, 4*molsize)

    maskd : mask for diagonal block for M, shape(ntotatoms,)
    M[maskd] take out the diagonal block
    gss, gpp, gsp, shape (ntotatoms, )
    P0: shape (nmol, 4*molsize, 4*molsize)
    """
    dtype = M.dtype
    device = M.device

    if(themethod == 'PM6'):
        P = (P0[:,0]+P0[:,1]).reshape((nmol,molsize,9,molsize,9)) \
              .transpose(2,3).reshape(nmol*molsize*molsize,9,9)

        PAlpha_ = P0.transpose(0,1).reshape((2,nmol,molsize,9,molsize,9)) \
              .transpose(3,4).reshape(2,nmol*molsize*molsize,9,9)
    else:
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
    if(themethod == 'PM6'):
        #(s,s)
        TMP_ = torch.zeros_like(F_)
        size = nmol*molsize*molsize
        #AllIntegrals = calc_integral(zetas, zetap, zetad, Z, size, maskd, P, F0SD, G2SD)
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
            
        ##(d,d) should go here.  Do not know exact form
        ##for i in range(4,9):
            ##TMP[maskd,i,i] = P[maskd,i,i]*(AllIntegrals[maskd,i,i])
            ##print( P[maskd,i,i],AllIntegrals[maskd,i,i])
            #(d,s) should go here.  Do not know exact form
            ##TMP[maskd,0,i] = P[maskd,0,i]*(AllIntegrals[maskd,i,i])
        #(d,d*) should go here.  Do not know exact form
        ##for i,j in [(4,5), (4,6), (4,7), (4,8), (5,6), (5,7), (5,8), (6,7), (6,8), (7,8)]:
            ##TMP[maskd,i,j] = P[maskd,i,j]*(AllIntegrals[maskd,i,j])

        #(d,p) should go here.  Do not know exact form
        ##for i,j in [(1,4), (1,5), (1,6), (1,7), (1,8), (2,4), (2,5), (2,6), (2,7), (2,8), (3,4), (3,5), (3,6), (3,7), (3,8)]:
            ##TMP[maskd,i,j] = P[maskd,i,j]*(AllIntegrals[maskd,i,j])
        ##print(TMP)
        ##print(F.add(AllIntegrals))
        F_.add_(TMP_)
        
        
        
        FLocal = torch.zeros(2,size,45, device = device)
        TMP_d =  torch.zeros(2,size,9,9, device = device )

        IntIJ = [ \
         1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, \
         4, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 8, 8, 8, 8, 9, \
         9, 9, 9,10,10,10,10,10,10,11,11,11,11,11,11,12,12,12,12,12, \
        13,13,13,13,13,14,14,14,15,15,15,15,15,15,15,15,15,15,16,16, \
        16,16,16,17,17,17,17,17,18,18,18,19,19,19,19,19,20,20,20,20, \
        20,21,21,21,21,21,21,21,21,21,21,21,21,22,22,22,22,22,22,22, \
        22,22,23,23,23,23,23,24,24,24,24,24,25,25,25,25,26,26,26,26, \
        26,26,27,27,27,27,27,28,28,28,28,28,28,28,28,28,28,29,29,29, \
        29,29,30,30,30,31,31,31,31,31,32,32,32,32,32,33,33,33,33,33, \
        34,34,34,34,35,35,35,35,35,36,36,36,36,36,36,36,36,36,36,36, \
        36,37,37,37,37,38,38,38,38,38,39,39,39,39,39,40,40,40,41,42, \
        42,42,42,42,43,43,43,43,44,44,44,44,44,45,45,45,45,45,45,45, \
        45,45,45  ]

        IntKL = [ \
        15,21,28,36,45,12,19,23,39,11,15,21,22,26,28,36,45,13,24,32, \
        38,34,37,43,11,15,21,22,26,28,36,45,17,25,31,16,20,27,44,29, \
        33,35,42,15,21,22,28,36,45, 3, 6,11,21,26,36, 2,12,19,23,39, \
         4,13,24,32,38,14,17,31, 1, 3, 6,10,15,21,22,28,36,45, 8,16, \
        20,27,44, 7,14,17,25,31,18,30,40, 2,12,19,23,39, 8,16,20,27, \
        44, 1, 3, 6,10,11,15,21,22,26,28,36,45, 3, 6,10,15,21,22,28, \
        36,45, 2,12,19,23,39, 4,13,24,32,38, 7,17,25,31, 3, 6,11,21, \
        26,36, 8,16,20,27,44, 1, 3, 6,10,15,21,22,28,36,45, 9,29,33, \
        35,42,18,30,40, 7,14,17,25,31, 4,13,24,32,38, 9,29,33,35,42, \
         5,34,37,43, 9,29,33,35,42, 1, 3, 6,10,11,15,21,22,26,28,36, \
        45, 5,34,37,43, 4,13,24,32,38, 2,12,19,23,39,18,30,40,41, 9, \
        29,33,35,42, 5,34,37,43, 8,16,20,27,44, 1, 3, 6,10,15,21,22, \
        28,36,45]
        
        j = 0
        filla = [0,1,1,2,2,2,3,3,3,3,4,4,4,4,4,5,5,5,5,5,5,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8,8]
        fillb = [0,0,1,0,1,2,0,1,2,3,0,1,2,3,4,0,1,2,3,4,5,0,1,2,3,4,5,6,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,8]

        PTMP = P.clone()
        PTMP_alpha = PAlpha_.clone()
        
        t0=time.time()

        i = 0 
        while ( i < 9):
            j = 0
            while(j < 9):
                PTMP[...,i,j] = P[...,j,i]
                PTMP_alpha[...,i,j] = PAlpha_[...,j,i]
                
                if (i != j):
                    PTMP[...,i,j] = 2.0*PTMP[...,i,j]
                    PTMP_alpha[...,i,j] = 2.0*PTMP_alpha[...,i,j]
                j = j + 1
            i = i +1

        Pnew = torch.zeros(size,45,device = device)
        Pnew_alpha = torch.zeros(2, size,45,device = device) 
        i = 0 
        while ( i < 45):
            Pnew[...,i] = PTMP[...,filla[i],fillb[i]]
            Pnew_alpha[...,i] = PTMP_alpha[...,filla[i],fillb[i]]
            i = i + 1
        j = 0
        
        while (j < 243):
            ij=IntIJ[j]-1
            kl=IntKL[j]-1
            #FLocal[...,ij]=FLocal[...,ij] + W[0][...,j]*Pnew[...,kl] - 0.5*W[1][...,j]*Pnew_alpha[...,kl]
            #print(Pnew_alpha*2-Pnew)
            FLocal[...,ij]=FLocal[...,ij] + (W[0][...,j]-W[1][...,j])*Pnew[...,kl] #- 2*W[1][...,j]*Pnew_alpha[...,kl]


            j = j + 1
        j = 0

        while( j <45):
            k = filla[j]
            l = fillb[j]
            TMP_d[...,k,l] = FLocal[...,j]
            j = j + 1
        TMP2 = TMP_d.clone()
        i = 0 
        while (i < 9):
            j = 0
            while(j < 9):
                TMP2[...,j,i] = TMP_d[...,i,j]
                j = j + 1
            i = i + 1
        
        
        i = 0
        j = 0
        while (i < TMP2.shape[2]):
            j = 0
            while(j < TMP2.shape[3]):
                if(  (i > j)):
                    TMP2[...,i,j] = 0.0
                j = j + 1
            i = i + 1
        
        #print(time.time()-t0)
        F_.add_(TMP2)
        #del TMP_d, TMP2, PTMP, Pnew, Pptot
        
        

    else:
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

        F_.add_(TMP_)
    
    ###############
    
    

    # sumation over two electron two center integrals over the neighbor atoms

    #for the diagonal block, check JAB in fock2.f
    #F_mu_nv = Hcore + \sum^B \sum_{lambda, sigma} P^B_{lambda, sigma} * (mu nu, lambda sigma)
    #as only upper triangle part is done, and put in order
    # (ss ), (px s), (px px), (py s), (py px), (py py), (pz s), (pz px), (pz py), (pz pz)
    #weight for them are
    #  1       2       1        2        2        1        2       2        2       1

    if(themethod == 'PM6'):
        weight = torch.tensor([1.0,
                               2.0, 1.0,
                               2.0, 2.0, 1.0,
                               2.0, 2.0, 2.0, 1.0,
                               2.0, 2.0, 2.0, 2.0, 1.0,
                               2.0, 2.0, 2.0, 2.0, 2.0, 1.0,
                               2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0,
                               2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0,
                               2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0],dtype=dtype, device=device).reshape((-1,45))


    else:
        weight = torch.tensor([1.0,
                               2.0, 1.0,
                               2.0, 2.0, 1.0,
                               2.0, 2.0, 2.0, 1.0],dtype=dtype, device=device).reshape((-1,10))
    #
    #P[maskd[idxi]] : P^tot_{mu,nu \in A} shape (npairs, 4,4)
    #P[maskd[idxj]] : P^tot_{mu,nu \in B} shape (npairs, 4,4)

    #take out the upper triangle part in the same order as in W
    #shape (nparis, 10)

    if(themethod == 'PM6'):

        PA = (P[maskd[idxj]][...,(0,0,1,0,1,2,0,1,2,3,0,1,2,3,4,0,1,2,3,4,5,0,1,2,3,4,5,6,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,8),(0,1,1,2,2,2,3,3,3,3,4,4,4,4,4,5,5,5,5,5,5,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8,8)]*weight).reshape((-1,45,1))
        PB = (P[maskd[idxi]][...,(0,0,1,0,1,2,0,1,2,3,0,1,2,3,4,0,1,2,3,4,5,0,1,2,3,4,5,6,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,8),(0,1,1,2,2,2,3,3,3,3,4,4,4,4,4,5,5,5,5,5,5,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8,8)]*weight).reshape((-1,1,45))

    else:
        PA = (P[maskd[idxi]][...,(0,0,1,0,1,2,0,1,2,3),(0,1,1,2,2,2,3,3,3,3)]*weight).reshape((-1,10,1))
        PB = (P[maskd[idxj]][...,(0,0,1,0,1,2,0,1,2,3),(0,1,1,2,2,2,3,3,3,3)]*weight).reshape((-1,1,10))

    #suma \sum_{mu,nu \in A} P_{mu, nu in A} (mu nu, lamda sigma) = suma_{lambda sigma \in B}
    #suma shape (npairs, 10)
    suma = torch.sum(PA*w,dim=1)
    #sumb \sum_{l,s \in B} P_{l, s inB} (mu nu, l s) = sumb_{mu nu \in A}
    #sumb shape (npairs, 10)
    sumb = torch.sum(PB*w,dim=2)
    #reshape back to (npairs 4,4)
    # as will use index add in the following part
    
    #torch.use_deterministic_algorithms(True)
    #print(11111)
    
    # $$$ Why in pm6 (0, maskd[idxi], sumA) but in pm3 (0, maskd[idxj], sumA) ??????
    if(themethod == 'PM6'):
        sumA = torch.zeros(w.shape[0],9,9,dtype=dtype, device=device)
        sumB = torch.zeros_like(sumA)
        sumA[...,(0,0,1,0,1,2,0,1,2,3,0,1,2,3,4,0,1,2,3,4,5,0,1,2,3,4,5,6,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,8),(0,1,1,2,2,2,3,3,3,3,4,4,4,4,4,5,5,5,5,5,5,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8,8)] = suma
        sumB[...,(0,0,1,0,1,2,0,1,2,3,0,1,2,3,4,0,1,2,3,4,5,0,1,2,3,4,5,6,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,8),(0,1,1,2,2,2,3,3,3,3,4,4,4,4,4,5,5,5,5,5,5,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8,8)] = sumb

        
        F_[0].index_add_(0, maskd[idxi], sumA)
        F_[1].index_add_(0, maskd[idxi], sumA)

        #\sum_B
        #F_.index_add_(1, maskd[idxi], sumB)
        F_[0].index_add_(0, maskd[idxj], sumB)
        F_[1].index_add_(0, maskd[idxj], sumB)

    else:
        sumA = torch.zeros(w.shape[0],4,4,dtype=dtype, device=device)
        sumB = torch.zeros_like(sumA)
        sumA[...,(0,0,1,0,1,2,0,1,2,3),(0,1,1,2,2,2,3,3,3,3)] = suma
        sumB[...,(0,0,1,0,1,2,0,1,2,3),(0,1,1,2,2,2,3,3,3,3)] = sumb
    
        #F^A_{mu, nu} = Hcore + \sum^A + \sum_{B} \sum_{l, s \in B} P_{l,s \in B} * (mu nu, l s)
        # $$$ index_add_ below could be done in a more efficient way
        #\sum_A
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
    
    if(themethod == 'PM6'):
        sum_ = torch.zeros(2, w.shape[0],9,9,dtype=dtype, device=device)
    else:
        sum_ = torch.zeros(2, w.shape[0],4,4,dtype=dtype, device=device)

    # (ss ), (px s), (px px), (py s), (py px), (py py), (pz s), (pz px), (pz py), (pz pz)
    #   0,     1         2       3       4         5       6      7         8        9
    if(themethod == 'PM6'):
        ind = torch.tensor([[0,1,3,6,10,15,21,28,36],
                            [1,2,4,7,11,16,22,29,37],
                            [3,4,5,8,12,17,23,30,38],
                            [6,7,8,9,13,18,24,31,39],
                            [10,11,12,13,14,19,25,32,40],
                            [15,16,17,18,19,20,26,33,41],
                            [21,22,23,24,25,26,27,34,42],
                            [28,29,30,31,32,33,34,35,43],
                            [36,37,38,39,40,41,42,43,44]],dtype=torch.int64, device=device)
    else:
        ind = torch.tensor([[0,1,3,6],
                            [1,2,4,7],
                            [3,4,5,8],
                            [6,7,8,9]],dtype=torch.int64, device=device)
        
    # Pp =P[mask], P_{mu \in A, lambda \in B}            
    if(themethod == 'PM6'):
        for i in range(9):
            for j in range(9):
                #\sum_{nu \in A} \sum_{sigma \in B} P_{nu, sigma} * (mu nu, lambda, sigma)
                #sum_[...,i,j] = torch.sum(Pp*torch.transpose(w[...,ind[j],:][...,:,ind[i]],1,2),dim=(1,2))
                sum_[...,i,j] = torch.sum(-PAlpha_[:,mask]*torch.transpose(w[...,ind[j],:][...,:,ind[i]],1,2),dim=(2,3))
    else:
        for i in range(4):
            for j in range(4):
                #\sum_{nu \in A} \sum_{sigma \in B} P_{nu, sigma} * (mu nu, lambda, sigma)
                sum_[...,i,j] = torch.sum(-PAlpha_[:,mask]*w[...,ind[i],:][...,:,ind[j]],dim=(2,3))
            
    F_.index_add_(1,mask,sum_)
    
    
    #torch.use_deterministic_algorithms(False)
    
    ####################
    
    if(themethod == 'PM6'):
        F0_ = F_.reshape(2, nmol,molsize,molsize,9,9) \
                 .transpose(3,4) \
                 .reshape(2, nmol, 9*molsize, 9*molsize).transpose(0,1)
    else:
        F0_ = F_.reshape(2, nmol, molsize, molsize, 4, 4) \
                 .transpose(3,4) \
                 .reshape(2, nmol, 4*molsize, 4*molsize).transpose(0,1)
        
    F0_.add_(F0_.triu(1).transpose(2,3))

    return F0_