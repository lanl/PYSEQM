# Not sure why we even have this file but I'll keep
# it in case it's needed in other packages using pyseqm like sedacs

import torch

# it is better to define mask as the same way defining maskd
# as it will be better to do summation using the representation of P in

def G(nmol, molsize, P0, M, maskd, mask, idxi, idxj, w, W, gss, gpp, gsp, gp2, hsp, themethod, zetas, zetap, zetad, Z, F0SD, G2SD):
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
    if(themethod == 'PM6'):
        P = P0.reshape((nmol,molsize,9,molsize,9)) \
              .transpose(2,3).reshape(nmol*molsize*molsize,9,9)
    else:
        P = P0.reshape((nmol,molsize,4,molsize,4)) \
              .transpose(2,3).reshape(nmol*molsize*molsize,4,4)

    #at this moment,  P has the same shape as M, as it is more convenient
    # to use here
    # while for diagonalization, may have to reshape
    # for the diagonal block, the summation over ortitals on the same atom in Fock matrix
    F = torch.zeros(M.shape, dtype=M.dtype, device=M.device)

    Pptot = P[...,1,1]+P[...,2,2]+P[...,3,3]
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
    if(themethod == 'PM6'):
        #(s,s)
        TMP = torch.zeros_like(M)
        size = nmol*molsize*molsize
        TMP[maskd,0,0] = 0.5*P[maskd,0,0]*gss + Pptot[maskd]*(gsp-0.5*hsp)
        for i in range(1,4):
        #(p,p)
            TMP[maskd,i,i] = P[maskd,0,0]*(gsp-0.5*hsp) + 0.5*P[maskd,i,i]*gpp \
                            + (Pptot[maskd] - P[maskd,i,i]) * (1.25*gp2-0.25*gpp)
        #(s,p) = (p,s) upper triangle
            TMP[maskd,0,i] = P[maskd,0,i]*(1.5*hsp - 0.5*gsp)
        #(p,p*)
        for i,j in [(1,2),(1,3),(2,3)]:
            TMP[maskd,i,j] = P[maskd,i,j]* (0.75*gpp - 1.25*gp2)
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
        F.add_(TMP)
        #del TMP
        
        
        #W = calc_integral(zetas, zetap, zetad, Z, size, maskd, P, F0SD, G2SD)
        PTMP = P.clone()
        TMP = torch.zeros(size,9,9, device = P0.device )
        FLocal = torch.zeros(size,45)
        tril_indices = torch.tril_indices(row=9, col=9, offset=0)
        tril1_indices = [[1,2,2,3,3,3,4,4,4,4,5,5,5,5,5,6,6,6,6,6,6,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8],
                         [0,0,1,0,1,2,0,1,2,3,0,1,2,3,4,0,1,2,3,4,5,0,1,2,3,4,5,6,0,1,2,3,4,5,6,7]]
        #print(tril1_indices)
        #print(tril_indices)
        #sys.exit()
        #triu1_indices = torch.triu_indices(row=9, col=9, offset=1)
        PTMP[...,tril1_indices[0],tril1_indices[1]] = 2.0*PTMP[...,tril1_indices[0],tril1_indices[1]]
        #PTMP[...,triu1_indices[0],triu1_indices[1]] = 2.0*PTMP[...,triu1_indices[0],triu1_indices[1]]
        #while ( i < 9):
        #    j = 0
        #    while(j < 9):
        #        PTMP[...,i,j] = P0[...,j,i]
        #        if (i != j):
        #            PTMP[...,i,j] = 2.0*PTMP[...,i,j]
        #        j = j + 1
        #    i = i +1

        Pnew = torch.zeros(size,45) 

        
        Pnew = PTMP[...,tril_indices[0],tril_indices[1]] 

        
        FLocal[...,0]=W[...,0]*Pnew[...,14]+W[...,1]*Pnew[...,20]+W[...,2]*Pnew[...,27]+W[...,3]*Pnew[...,35]+W[...,4]*Pnew[...,44]
        FLocal[...,1]=W[...,5]*Pnew[...,11]+W[...,6]*Pnew[...,18]+W[...,7]*Pnew[...,22]+W[...,8]*Pnew[...,38]
        FLocal[...,2]=W[...,9]*Pnew[...,10]+W[...,10]*Pnew[...,14]+W[...,11]*Pnew[...,20]+W[...,12]*Pnew[...,21] \
        +W[...,13]*Pnew[...,25]+W[...,14]*Pnew[...,27]+W[...,15]*Pnew[...,35]+W[...,16]*Pnew[...,44]
        FLocal[...,3]=W[...,17]*Pnew[...,12]+W[...,18]*Pnew[...,23]+W[...,19]*Pnew[...,31]+W[...,20]*Pnew[...,37]
        FLocal[...,4]=W[...,21]*Pnew[...,33]+W[...,22]*Pnew[...,36]+W[...,23]*Pnew[...,42]
        FLocal[...,5]=W[...,24]*Pnew[...,10]+W[...,25]*Pnew[...,14]+W[...,26]*Pnew[...,20]+W[...,27]*Pnew[...,21] \
        +W[...,28]*Pnew[...,25]+W[...,29]*Pnew[...,27]+W[...,30]*Pnew[...,35]+W[...,31]*Pnew[...,44]
        FLocal[...,6]=W[...,32]*Pnew[...,16]+W[...,33]*Pnew[...,24]+W[...,34]*Pnew[...,30]
        FLocal[...,7]=W[...,35]*Pnew[...,15]+W[...,36]*Pnew[...,19]+W[...,37]*Pnew[...,26]+W[...,38]*Pnew[...,43]
        FLocal[...,8]=W[...,39]*Pnew[...,28]+W[...,40]*Pnew[...,32]+W[...,41]*Pnew[...,34]+W[...,42]*Pnew[...,41]
        FLocal[...,9]=W[...,43]*Pnew[...,14]+W[...,44]*Pnew[...,20]+W[...,45]*Pnew[...,21]+W[...,46]*Pnew[...,27] \
        +W[...,47]*Pnew[...,35]+W[...,48]*Pnew[...,44]
        FLocal[...,10]=W[...,49]*Pnew[...,2]+W[...,50]*Pnew[...,5]+W[...,51]*Pnew[...,10] \
        +W[...,52]*Pnew[...,20]+W[...,53]*Pnew[...,25]+W[...,54]*Pnew[...,35]
        FLocal[...,11]=W[...,55]*Pnew[...,1]+W[...,56]*Pnew[...,11]+W[...,57]*Pnew[...,18]+W[...,58]*Pnew[...,22]+W[...,59]*Pnew[...,38]
        FLocal[...,12]=W[...,60]*Pnew[...,3]+W[...,61]*Pnew[...,12]+W[...,62]*Pnew[...,23]+W[...,63]*Pnew[...,31]+W[...,64]*Pnew[...,37]
        FLocal[...,13]=W[...,65]*Pnew[...,13]+W[...,66]*Pnew[...,16]+W[...,67]*Pnew[...,30]
        FLocal[...,14]=W[...,68]*Pnew[...,0]+W[...,69]*Pnew[...,2]+W[...,70]*Pnew[...,5]+W[...,71]*Pnew[...,9]+W[...,72]*Pnew[...,14] \
        +W[...,73]*Pnew[...,20]+W[...,74]*Pnew[...,21]+W[...,75]*Pnew[...,27]+W[...,76]*Pnew[...,35]+W[...,77]*Pnew[...,44]
        FLocal[...,15]=W[...,78]*Pnew[...,7]+W[...,79]*Pnew[...,15]+W[...,80]*Pnew[...,19]+W[...,81]*Pnew[...,26]+W[...,82]*Pnew[...,43]
        FLocal[...,16]=W[...,83]*Pnew[...,6]+W[...,84]*Pnew[...,13]+W[...,85]*Pnew[...,16]+W[...,86]*Pnew[...,24]+W[...,87]*Pnew[...,30]
        FLocal[...,17]=W[...,88]*Pnew[...,17]+W[...,89]*Pnew[...,29]+W[...,90]*Pnew[...,39]
        FLocal[...,18]=W[...,91]*Pnew[...,1]+W[...,92]*Pnew[...,11]+W[...,93]*Pnew[...,18]+W[...,94]*Pnew[...,22]+W[...,95]*Pnew[...,38]
        FLocal[...,19]=W[...,96]*Pnew[...,7]+W[...,97]*Pnew[...,15]+W[...,98]*Pnew[...,19]+W[...,99]*Pnew[...,26]+W[...,100]*Pnew[...,43]
        FLocal[...,20]=W[...,101]*Pnew[...,0]+W[...,102]*Pnew[...,2]+W[...,103]*Pnew[...,5]+W[...,104]*Pnew[...,9]+W[...,105]*Pnew[...,10] \
        +W[...,106]*Pnew[...,14]+W[...,107]*Pnew[...,20]+W[...,108]*Pnew[...,21]+W[...,109]*Pnew[...,25]+W[...,110]*Pnew[...,27] \
        +W[...,111]*Pnew[...,35]+W[...,112]*Pnew[...,44]
        FLocal[...,21]=W[...,113]*Pnew[...,2]+W[...,114]*Pnew[...,5]+W[...,115]*Pnew[...,9]+W[...,116]*Pnew[...,14] \
        +W[...,117]*Pnew[...,20] \
        +W[...,118]*Pnew[...,21]+W[...,119]*Pnew[...,27]+W[...,120]*Pnew[...,35]+W[...,121]*Pnew[...,44]
        FLocal[...,22]=W[...,122]*Pnew[...,1]+W[...,123]*Pnew[...,11]+W[...,124]*Pnew[...,18] \
        +W[...,125]*Pnew[...,22]+W[...,126]*Pnew[...,38]
        FLocal[...,23]=W[...,127]*Pnew[...,3]+W[...,128]*Pnew[...,12]+W[...,129]*Pnew[...,23] \
        +W[...,130]*Pnew[...,31]+W[...,131]*Pnew[...,37]
        FLocal[...,24]=W[...,132]*Pnew[...,6]+W[...,133]*Pnew[...,16]+W[...,134]*Pnew[...,24]+W[...,135]*Pnew[...,30]
        FLocal[...,25]=W[...,136]*Pnew[...,2]+W[...,137]*Pnew[...,5]+W[...,138]*Pnew[...,10] \
        +W[...,139]*Pnew[...,20]+W[...,140]*Pnew[...,25]+W[...,141]*Pnew[...,35]
        FLocal[...,26]=W[...,142]*Pnew[...,7]+W[...,143]*Pnew[...,15]+W[...,144]*Pnew[...,19] \
        +W[...,145]*Pnew[...,26]+W[...,146]*Pnew[...,43]
        FLocal[...,27]=W[...,147]*Pnew[...,0]+W[...,148]*Pnew[...,2]+W[...,149]*Pnew[...,5] \
        +W[...,150]*Pnew[...,9]+W[...,151]*Pnew[...,14]+W[...,152]*Pnew[...,20]+W[...,153]*Pnew[...,21] \
        +W[...,154]*Pnew[...,27]+W[...,155]*Pnew[...,35]+W[...,156]*Pnew[...,44]
        FLocal[...,28]=W[...,157]*Pnew[...,8]+W[...,158]*Pnew[...,28]+W[...,159]*Pnew[...,32] \
        +W[...,160]*Pnew[...,34]+W[...,161]*Pnew[...,41]
        FLocal[...,29]=W[...,162]*Pnew[...,17]+W[...,163]*Pnew[...,29]+W[...,164]*Pnew[...,39]
        FLocal[...,30]=W[...,165]*Pnew[...,6]+W[...,166]*Pnew[...,13]+W[...,167]*Pnew[...,16] \
        +W[...,168]*Pnew[...,24]+W[...,169]*Pnew[...,30]
        FLocal[...,31]=W[...,170]*Pnew[...,3]+W[...,171]*Pnew[...,12]+W[...,172]*Pnew[...,23] \
        +W[...,173]*Pnew[...,31]+W[...,174]*Pnew[...,37]
        FLocal[...,32]=FLocal[...,32]+W[...,175]*Pnew[...,8]+W[...,176]*Pnew[...,28]+W[...,177]*Pnew[...,32] \
        +W[...,178]*Pnew[...,34]+W[...,179]*Pnew[...,41]
        FLocal[...,33]=W[...,180]*Pnew[...,4]+W[...,181]*Pnew[...,33]+W[...,182]*Pnew[...,36]+W[...,183]*Pnew[...,42]
        FLocal[...,34]=W[...,184]*Pnew[...,8]+W[...,185]*Pnew[...,28]+W[...,186]*Pnew[...,32] \
        +W[...,187]*Pnew[...,34]+W[...,188]*Pnew[...,41]
        FLocal[...,35]=W[...,189]*Pnew[...,0]+W[...,190]*Pnew[...,2]+W[...,191]*Pnew[...,5]+W[...,192]*Pnew[...,9] \
        +W[...,193]*Pnew[...,10]+W[...,194]*Pnew[...,14]+W[...,195]*Pnew[...,20]+W[...,196]*Pnew[...,21] \
        +W[...,197]*Pnew[...,25]+W[...,198]*Pnew[...,27]+W[...,199]*Pnew[...,35]+W[...,200]*Pnew[...,44]
        FLocal[...,36]=W[...,201]*Pnew[...,4]+W[...,202]*Pnew[...,33]+W[...,203]*Pnew[...,36]+W[...,204]*Pnew[...,42]
        FLocal[...,37]=W[...,205]*Pnew[...,3]+W[...,206]*Pnew[...,12]+W[...,207]*Pnew[...,23] \
        +W[...,208]*Pnew[...,31]+W[...,209]*Pnew[...,37]
        FLocal[...,38]=W[...,210]*Pnew[...,1]+W[...,211]*Pnew[...,11]+W[...,212]*Pnew[...,18] \
        +W[...,213]*Pnew[...,22]+W[...,214]*Pnew[...,38]
        FLocal[...,39]=W[...,215]*Pnew[...,17]+W[...,216]*Pnew[...,29]+W[...,217]*Pnew[...,39]
        FLocal[...,40]=FLocal[...,40]+W[...,218]*Pnew[...,40]
        FLocal[...,41]=W[...,219]*Pnew[...,8]+W[...,220]*Pnew[...,28]+W[...,221]*Pnew[...,32] \
        +W[...,222]*Pnew[...,34]+W[...,223]*Pnew[...,41]
        FLocal[...,42]=W[...,224]*Pnew[...,4]+W[...,225]*Pnew[...,33]+W[...,226]*Pnew[...,36]+W[...,227]*Pnew[...,42]
        FLocal[...,43]=W[...,228]*Pnew[...,7]+W[...,229]*Pnew[...,15]+W[...,230]*Pnew[...,19] \
        +W[...,231]*Pnew[...,26]+W[...,232]*Pnew[...,43]
        FLocal[...,44]=W[...,233]*Pnew[...,0]+W[...,234]*Pnew[...,2]+W[...,235]*Pnew[...,5]+W[...,236]*Pnew[...,9]+W[...,237]*Pnew[...,14] \
        +W[...,238]*Pnew[...,20]+W[...,239]*Pnew[...,21]+W[...,240]*Pnew[...,27]+W[...,241]*Pnew[...,35]+W[...,242]*Pnew[...,44]
        TMP[...,tril_indices[1],tril_indices[0]] = FLocal
        F.add_(TMP)
        #del TMP_d, TMP2, PTMP, Pnew, Pptot
        

    else:
        ### http://openmopac.net/manual/1c2e.html
        #(s,s)
        TMP = torch.zeros_like(M)
        TMP[maskd,0,0] = 0.5*P[maskd,0,0]*gss + Pptot[maskd]*(gsp-0.5*hsp)
        for i in range(1,4):
            #(p,p)
            TMP[maskd,i,i] = P[maskd,0,0]*(gsp-0.5*hsp) + 0.5*P[maskd,i,i]*gpp \
                            + (Pptot[maskd] - P[maskd,i,i]) * (1.25*gp2-0.25*gpp)
            #(s,p) = (p,s) upper triangle
            TMP[maskd,0,i] = P[maskd,0,i]*(1.5*hsp - 0.5*gsp)
        #(p,p*)
        for i,j in [(1,2),(1,3),(2,3)]:
            TMP[maskd,i,j] = P[maskd,i,j]* (0.75*gpp - 1.25*gp2)

        F.add_(TMP)

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
    if(themethod == 'PM6'):
        sumA = torch.zeros(w.shape[0],9,9,dtype=dtype, device=device)
        sumB = torch.zeros_like(sumA)
        sumA[...,(0,0,1,0,1,2,0,1,2,3,0,1,2,3,4,0,1,2,3,4,5,0,1,2,3,4,5,6,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,8),(0,1,1,2,2,2,3,3,3,3,4,4,4,4,4,5,5,5,5,5,5,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8,8)] = suma
        sumB[...,(0,0,1,0,1,2,0,1,2,3,0,1,2,3,4,0,1,2,3,4,5,0,1,2,3,4,5,6,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,8),(0,1,1,2,2,2,3,3,3,3,4,4,4,4,4,5,5,5,5,5,5,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8,8)] = sumb

        F.index_add_(0,maskd[idxi],sumA) 
        F.index_add_(0,maskd[idxj],sumB)

    else:
        sumA = torch.zeros(w.shape[0],4,4,dtype=dtype, device=device)
        sumB = torch.zeros_like(sumA)
        sumA[...,(0,0,1,0,1,2,0,1,2,3),(0,1,1,2,2,2,3,3,3,3)] = suma
        sumB[...,(0,0,1,0,1,2,0,1,2,3),(0,1,1,2,2,2,3,3,3,3)] = sumb
        #F^A_{mu, nu} = Hcore + \sum^A + \sum_{B} \sum_{l, s \in B} P_{l,s \in B} * (mu nu, l s)
        #\sum_B
        F.index_add_(0,maskd[idxi],sumB)
        #\sum_A
        F.index_add_(0,maskd[idxj],sumA)



    # off diagonal block part, check KAB in forck2.f
    # mu, nu in A
    # lambda, sigma in B
    # F_mu_lambda = Hcore - 0.5* \sum_{nu \in A} \sum_{sigma in B} P_{nu, sigma} * (mu nu, lambda, sigma)
    if(themethod == 'PM6'):
        sum = torch.zeros(w.shape[0],9,9,dtype=dtype, device=device)
    else:
        sum = torch.zeros(w.shape[0],4,4,dtype=dtype, device=device)
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
    Pp = -0.5*P[mask]
    if(themethod == 'PM6'):
        for i in range(9):
            for j in range(9):
                #\sum_{nu \in A} \sum_{sigma \in B} P_{nu, sigma} * (mu nu, lambda, sigma)
                sum[...,i,j] = torch.sum(Pp*torch.transpose(w[...,ind[j],:][...,:,ind[i]],1,2),dim=(1,2))

    else:
        for i in range(4):
            for j in range(4):
                #\sum_{nu \in A} \sum_{sigma \in B} P_{nu, sigma} * (mu nu, lambda, sigma)
                sum[...,i,j] = torch.sum(Pp*w[...,ind[i],:][...,:,ind[j]],dim=(1,2))
    #
    F.index_add_(0,mask,sum)

    # if(themethod == 'PM6'):
    #     F0 = F.reshape(nmol,molsize,molsize,9,9) \
    #              .transpose(2,3) \
    #              .reshape(nmol, 9*molsize, 9*molsize)
    # else:
    #     F0 = F.reshape(nmol,molsize,molsize,4,4) \
    #              .transpose(2,3) \
    #              .reshape(nmol, 4*molsize, 4*molsize)
    F0.add_(F0.triu(1).transpose(1,2))

    return F0
