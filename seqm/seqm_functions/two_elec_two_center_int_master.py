import torch
from .two_elec_two_center_int_local_frame import two_elec_two_center_int_local_frame as TETCILF
from .two_elec_two_center_int_local_frame_d_orbitals_master3 import two_elec_two_center_int_local_frame_d_orbitals as TETCILFDO
from .cal_par import *
from .constants import ev
import sys
import math
import numpy
#import scipy.special
from .parameters import  PWCCT
from .RotationMatrixD import *
#two electron two center integrals
def two_elec_two_center_int(const,idxi, idxj, ni, nj, xij, rij, Z, zetas, zetap, zetad, zs, zp, zd,  gss, gpp, gp2, hsp, F0SD, G2SD, rho_core, alpha, chi, themethod):
    """
    two electron two center integrals in molecule frame
    """
    #t = time.time()
    dtype = xij.dtype
    device = xij.device
    #two electron two center integrals
    # ni, nj, xij, rij for each pair
    # Z, zetas, zetap, gss, gpp, gp2, hsp for each atom

    #rotate(ni,nj,xij,rij,tore,da,db, qa,qb, rho0a,rho0b, rho1a,rho1b, rho2a,rho2b, cutoff=1.0e10):
    # ni, nj, xij, rij, da, db, qa, qb, rho0a, rho0b ... rho2b, shape (napirs,)
    # tore: dictionary type tensor tore[1]=1,
    #       valence shell charge for H, tore[6]=4, valence shell charge for C
    tore = const.tore
    qn = const.qn
    qnd = const.qnD_int
    hpp = 0.5*(gpp-gp2)
    hppd =  0.5*(gpp-gp2)
    ##t0 = time.time()
    j = 0
    for i in hpp:
       if(float(i) < 0.1):
           hpp[j]= 0.1
       j = j + 1
    qn0=qn[Z]
    qnd0=qn[Z]
    #Z==0 is for padding zero
    isH = Z==1  # Hydrogen
    isX = Z>2   # Heavy atom
    #### populations elements with d-oribitals
    ###Need extra rho to hold additive terms
    rho_0=torch.zeros_like(qn0)
    rho_1=torch.zeros_like(qn0)
    rho_2=torch.zeros_like(qn0)
    rho_2d=torch.zeros_like(qn0)



    dd=torch.zeros_like(qn0)
    qq=torch.zeros_like(qn0)
    ###Aditional charge seperations we can have with d-oribitals
    dp=torch.zeros_like(qn0)
    ds=torch.zeros_like(qn0)
    dorbdorb=torch.zeros_like(qn0)
    ###
    rho1 = additive_term_rho1.apply
    rho2 = additive_term_rho2.apply

    dd[isX], qq[isX] = dd_qq(qn0[isX],zetas[isX], zetap[isX])

###Correct coefficients?
### 1/5, 4/15, 3/49, 1 or values from paper?

    dsAdditiveTerm=torch.zeros_like(qn0)
    dpAdditiveTerm=torch.zeros_like(qn0)
    ddAdditiveTerm=torch.zeros_like(qn0)
    dd0AdditiveTerm=torch.zeros_like(qn0)
    AIJ52    = torch.zeros_like(qn0)
    AIJ43    = torch.zeros_like(qn0)
    AIJ63    = torch.zeros_like(qn0)
##    AIJ22 = torch.zeros_like(qn0)

    dd4 = torch.zeros_like(qn0)
    dp3 = torch.zeros_like(qn0)
    i = 0
    for j in Z:
        if (j > 20 and j <30) or (j > 38 and j <48) or (j > 70 and j <80) or j == 57:
#            dsAdditiveTerm[i]   = 1/5*GetSlaterCondonParameter(2,qn0[i],zs[i],qn0[i]-1,zd[i],qn0[i],zs[i],qn0[i]-1,zd[i])
            dpAdditiveTerm[i]   = (4/15)*GetSlaterCondonParameter(1,qn0[i],zp[i],qn0[i]-1,zd[i],qn0[i],zp[i],qn0[i]-1,zd[i])
            if (  G2SD[i] > 10**-9):
                dsAdditiveTerm[i] = 1/5*G2SD[i]
            else:
                dsAdditiveTerm[i]   = 1/5*GetSlaterCondonParameter(2,qn0[i],zs[i],qn0[i]-1,zd[i],qn0[i],zs[i],qn0[i]-1,zd[i])
            ddAdditiveTerm[i]   = (4/49)*GetSlaterCondonParameter(2,qn0[i]-1,zd[i],qn0[i]-1,zd[i],qn0[i]-1,zd[i],qn0[i]-1,zd[i])
            dd0AdditiveTerm[i]  = GetSlaterCondonParameter(0,qn0[i]-1,zd[i],qn0[i]-1,zd[i],qn0[i]-1,zd[i],qn0[i]-1,zd[i])
            dd4[i] = GetSlaterCondonParameter(4,qn0[i]-1,zd[i],qn0[i]-1,zd[i],qn0[i]-1,zd[i],qn0[i]-1,zd[i])
            dp3[i] = 27/245*GetSlaterCondonParameter(3,qn0[i],zp[i],qn0[i]-1,zd[i],qn0[i],zp[i],qn0[i]-1,zd[i])
            AIJ52[i]    = AIJL(zetap[i],zetad[i],qn0[i],qn0[i]-1,1)
            AIJ43[i]    = AIJL(zetas[i],zetad[i],qn0[i],qn0[i]-1,2)
            AIJ63[i]    = AIJL(zetad[i],zetad[i],qn0[i]-1,qn0[i]-1,2)

                                   
        elif (j > 32 and j <36) or (j > 50 and j <54):
            dsAdditiveTerm[i]   = 1/5*GetSlaterCondonParameter(2,qn0[i],zs[i],qn0[i],zd[i],qn0[i],zs[i],qn0[i],zd[i])
            dpAdditiveTerm[i]   = (4/15)*GetSlaterCondonParameter(1,qn0[i],zp[i],qn0[i],zd[i],qn0[i],zp[i],qn0[i],zd[i])
            ddAdditiveTerm[i]   = (4/49)*GetSlaterCondonParameter(2,qn0[i],zd[i],qn0[i],zd[i],qn0[i],zd[i],qn0[i],zd[i])
            dd0AdditiveTerm[i]  = GetSlaterCondonParameter(0,qn0[i],zd[i],qn0[i],zd[i],qn0[i],zd[i],qn0[i],zd[i])
            dd4[i] = GetSlaterCondonParameter(4,qn0[i],zd[i],qn0[i],zd[i],qn0[i],zd[i],qn0[i],zd[i])
            dp3[i] = 27/245*GetSlaterCondonParameter(3,qn0[i],zp[i],qn0[i],zd[i],qn0[i],zp[i],qn0[i],zd[i])
            AIJ52[i]    = AIJL(zetap[i],zetad[i],qn0[i],qn0[i],1)
            AIJ43[i]    = AIJL(zetas[i],zetad[i],qn0[i],qn0[i],2)
            AIJ63[i]    = AIJL(zetad[i],zetad[i],qn0[i],qn0[i],2)
##        AIJ22[i] = AIJL(zetas[i],zetap[i],qn0[i],qn0[i],1)
        i = i + 1
            

#    print(time.time() - t)
    dp= AIJ52/math.sqrt(5)
    isY = qnd[Z] > 0
    D = torch.sqrt(AIJ43*math.sqrt(1.0000/15.0000))*math.sqrt(2.0000)
    ds = D
    DS = torch.zeros_like(qn0)
    DS[isY] = POIJ(2,D[isY],dsAdditiveTerm[isY])

    DD0 = torch.zeros_like(qn0)
    FG = dd0AdditiveTerm + ddAdditiveTerm + 4/49*dd4
    FG1 = dd0AdditiveTerm + 0.5*ddAdditiveTerm - 24/441*dd4
    FG2 = dd0AdditiveTerm - ddAdditiveTerm + 6/441*dd4
    DD0[isY] = POIJ(0,1.00000,0.20000*(FG[isY]+2.00000*FG1[isY]+2.00000*FG2[isY]))

    D = AIJ52/math.sqrt(5.0000)
    FG=dpAdditiveTerm+dp3
    FG1=3/49*245/27*dp3
    DP = torch.zeros_like(qn0)
    DP[isY] = POIJ(1,D[isY],FG[isY]-1.8000*FG1[isY])
    ##print(AIJ63, "YER")
    D        = AIJ63/7.0000
    D        = torch.sqrt(2.0000*D)
    dorbdorb = D
    FG= 3/4*ddAdditiveTerm+20/441*dd4
    FG1=35/441*dd4
    DD = torch.zeros_like(qn0)
    DD[isY] = POIJ(2,D[isY],FG[isY]-(20.0000/35.0000)*FG1[isY])
#    y = torch.sqrt((2*qn0+1)*(2*qn0+2)/20.00000) / zetap
    ##print (y)
#    x = POIJ(2,y*math.sqrt(2),hpp[isY])
    ##print (x)

    rho_0 = 0.5*ev/gss
    ##print(isX,isY)
    
    rho_1[isX] = rho1(hsp[isX],dd[isX])
    rho_2[isX] = rho2(hpp[isX],qq[isX])
    rho_2d[isX] = POIJ(2,qq[isX]*math.sqrt(2),hppd[isX])
    ##rho_2[isX] = POIJ(2,qq[isX]*math.sqrt(2),hpp[isX])        

    
    w, e1b, e2a = \
        rotate(ni, nj, xij, rij, \
               tore, dd[idxi],dd[idxj], \
               qq[idxi], qq[idxj], \
               dp[idxi],dp[idxj], \
               ds[idxi],ds[idxj], \
               dorbdorb[idxi],dorbdorb[idxj], \
               rho_0[idxi],rho_0[idxj], \
               rho_1[idxi],rho_1[idxj], \
               rho_2[idxi],rho_2[idxj], \
               DD0[idxi]*isY[idxi],DD0[idxj]*isY[idxj], \
               DP[idxi]*isY[idxi],DP[idxj]*isY[idxj], \
               ##rho_3[idxi],rho_3[idxj], \
               ##rho_4[idxi],rho_4[idxj], \
               DS[idxi]*isY[idxi],DS[idxj]*isY[idxj], \
               ##rho_6[idxi],rho_6[idxj], \
               DD[idxi]*isY[idxi],DD[idxj]*isY[idxj], \
               alpha, themethod,rho_2d[idxi],rho_2d[idxj],rho_core[idxi],rho_core[idxj])

    rho0aTMP = rho_0[idxi].clone()
    rho0bTMP = rho_0[idxj].clone()
    A = (rho_core[idxi] != 0.000)
    B = (rho_core[idxj] != 0.000)

    rho0aTMP[A] =rho_core[idxi][A]
    rho0bTMP[B] =rho_core[idxj][B]
    
    return w, e1b, e2a,rho0aTMP,rho0bTMP

#rotate: rotate the two electron two center integrals from local frame to molecule frame
def rotate(ni,nj,xij,rij,
           tore,da,db, qa,
           qb, dpa, dpb, dsa, 
           dsb, dda, ddb, rho0a,rho0b, rho1a,rho1b, rho2a,rho2b, rho3a,rho3b, rho4a,rho4b, rho5a,rho5b, rho6a,rho6b, diadia, themethod, rho2ad,rho2bd, rho_corea, rho_coreb, cutoff=1.0e10):
    """
    rotate the two elecron two center integrals from local frame to molecule frame
    """
    dtype =  xij.dtype
    device = xij.device
#    t0 =    time.time() 
    # in mopac, xij = xi - xj for rotate, rather than the general one, xj-xi
    # thus put minus sign on xij
    # ni, nj, xij, rij, da, db, qa, qb, rho0a, rho0b ... rho2b, shape (napirs,)
    # tore: dictionary type tensor tore[1]=1,
    #       valence shell charge for H, tore[6]=4, valence shell charge for C

    #   ROTATE CALCULATES THE TWO-PARTICLE INTERACTIONS.
    #
    #   ON INPUT  NI     = ATOMIC NUMBER OF FIRST ATOM.
    #             NJ     = ATOMIC NUMBER OF SECOND ATOM.
    #
    # ON OUTPUT W      = ARRAY OF TWO-ELECTRON REPULSION INTEGRALS.
    #           E1B,E2A= ARRAY OF ELECTRON-NUCLEAR ATTRACTION INTEGRALS,
    #                    E1B = ELECTRON ON ATOM NI ATTRACTING NUCLEUS OF NJ.
    # for the local frame, refer to TwoElectronTwoCenterIntegralsLocalFrame.py
    # repp.f
    # xij, unit vector from atom i to atom j, shape (npairs, 3)
    # ni>=nj, shape (npairs, )
    # rij, distance between atom i and atom j in atomic unit, shape (npairs,)
    # tore, da, db, qa, qb, rho0, rho1, rho2 ==> refer to TwoElectronTwoCenterIntegralsLocalFrame.py
    # cutoff, not used
    #***************************************************************************
    #the inner cutoff in rotate.f is sqrt(2e-5) Angstrom = 0.00447 Angstrom
    #default inner cutoff in data_loader is 0.001 Angstrom
    # +++++
    # in this module, xij is the unit vector for the direction of atom i -> j
    # update data_loader soon
    # +++++
    #***************************************************************************
    #xij = xij0/torch.norm(xij,dim=1,keepdim=True)
    YH = (((ni > 12) & (ni <18)) | ((ni > 20) & (ni <30)) | ((ni > 32) & (ni <36)) | ((ni > 38) & (ni <48)) | ((ni > 50) & (ni <54)) | ((ni > 70) & (ni <80)) | (ni ==57))  & (nj==1)
    YX = (((ni > 12) & (ni <18)) | ((ni > 20) & (ni <30)) | ((ni > 32) & (ni <36)) | ((ni > 38) & (ni <48)) | ((ni > 50) & (ni <54)) | ((ni > 70) & (ni <80)) | (ni ==57)) & \
         ((nj <= 12) | ((nj >= 18) & (nj <=20)) | ((nj >= 30) & (nj <= 32)) | ((nj >= 36) & (nj <= 38)) | ((nj >= 48) & (nj <= 50)) | ((nj >= 54) & (nj <= 56)) | ((nj >= 80) & (nj <= 83))) & (nj !=1)
    YY = (((ni > 12) & (ni <18)) | ((ni > 20) & (ni <30)) | ((ni > 32) & (ni <36)) | ((ni > 38) & (ni <48)) | ((ni > 50) & (ni <54)) | ((ni > 70) & (ni <80)) | (ni ==57)) &\
         (((nj > 12) & (nj <18)) | ((nj > 20) & (nj <30)) | ((nj > 32) & (nj <36)) | ((nj > 38) & (nj <48)) | ((nj > 50) & (nj <54)) | ((nj > 70) & (nj <80)) | (nj ==57))

    #enuc is not computed at this moment
#    if(themethod == "PM6"):
#        HH = (ni==1) & (nj==1)
#        XH =((ni <= 12) | ((ni >= 18) & (ni <=20)) | ((ni >= 30) & (ni <= 32)) | ((ni >= 36) & (ni <= 38)) | ((ni >= 48) & (ni <= 50)) | ((ni >= 54) & (ni <= 56)) | ((ni >= 80) & (ni <= 83))) & (ni !=1) & (nj==1)
#        XX =((ni <= 12) | ((ni >= 18) & (ni <=20)) | ((ni >= 30) & (ni <= 32)) | ((ni >= 36) & (ni <= 38)) | ((ni >= 48) & (ni <= 50)) | ((ni >= 54) & (ni <= 56)) | ((ni >= 80) & (ni <= 83))) & (ni !=1) & \
#            ((nj <= 12) | ((nj >= 18) & (nj <=20)) | ((nj >= 30) & (nj <= 32)) | ((nj >= 36) & (nj <= 38)) | ((nj >= 48) & (nj <= 50)) | ((nj >= 54) & (nj <= 56)) | ((nj >= 80) & (nj <= 83))) & (nj !=1)

 #   else:
    HH = (ni==1) & (nj==1)
    XH = (ni>1) & (nj==1) 
    XX = (ni>1) & (nj>1) 

    wHH, riXH, ri, coreHH, coreXH, core = \
           TETCILF(ni,nj,rij, tore, \
                da, db, qa,qb, rho0a,rho0b, rho1a,rho1b, rho2a,rho2b,themethod)
    riXHPM6 = riXH.clone()
    riPM6a = ri.clone()
    riPM6b = ri.clone()
    if(rho_corea.sum() > 0 or rho_coreb.sum() > 0 ):
       rho0aTMP = rho0a.clone()
       rho0bTMP = rho0b.clone()
       A = (rho_corea != 0.000)
       B = (rho_coreb != 0.000)

       rho0aTMP[A] =rho_corea[A]
       rho0bTMP[B] =rho_coreb[B]
       trash, riXHPM6, riPM6b, notneeded, dont, need = \
           TETCILF(ni,nj,rij, tore, \
                da, db, qa,qb, rho0a,rho0bTMP, rho1a,rho1b, rho2a,rho2b,themethod)


       trash, riXHPM6, riPM6a, notneeded, dont, need = \
           TETCILF(ni,nj,rij, tore, \
                da, db, qa,qb, rho0aTMP,rho0b, rho1a,rho1b, rho2a,rho2b,themethod)


    ###############################33
    # X-H hevay atom - Hydrogen
    xXH=-xij[XH]
    yXH=torch.zeros(xXH.shape[0],2,dtype=dtype, device=device)
    zXH=torch.zeros_like(xXH)
    """
    pytorch new version doesn't support modify z and z depends on a , a depends on z
    zXH[...,3-1] =  torch.sqrt(1.0-xXH[...,3-1]**2)
    cond1XH = zXH[...,3-1]>1.0e-5
    #cond1 = torch.abs(xXH[...,3-1])>0.99999999, but z[3] is used in dominator, so change to
    #cond1 = zXH[...,3-1]>0.0
    #abs(x[3])>0.99999999 ==> z[3]<1.4142e-4
    #don't use th.where, use indexing to get rid of singularity
    aXH = 1.0/zXH[cond1XH,3-1]

    zXH[...,1-1] = 1.0
    zXH[cond1XH,1-1] = -aXH*xXH[cond1XH,1-1]*xXH[cond1XH,3-1]

    #zXH[...,2-1]=0.0
    zXH[cond1XH,2-1] = -aXH*xXH[cond1XH,2-1]*xXH[cond1XH,3-1]
    """
    #modify the code
    #xXH.register_hook(print)
    #zXH2 = torch.sqrt(1.0-xXH[...,3-1]**2)
    zXH2 = torch.zeros_like(xXH[...,2])
    cond_xXH2 = torch.abs(xXH[...,3-1])<1.0
    zXH2[cond_xXH2] = torch.sqrt(1.0-xXH[cond_xXH2,3-1]**2)
    cond1XH = zXH2>1.0e-5
    aXH = 1.0/zXH2[cond1XH]
    zXH0 =  torch.ones_like(zXH2)
    zXH0[cond1XH] = -aXH*xXH[cond1XH,1-1]*xXH[cond1XH,3-1]
    zXH1 = torch.zeros_like(zXH2)
    zXH1[cond1XH] = -aXH*xXH[cond1XH,2-1]*xXH[cond1XH,3-1]
    zXH = torch.stack((zXH0, zXH1, zXH2), dim=1)


    #yXH[...,1-1]=0.0
    yXH[cond1XH,0] =  aXH*xXH[cond1XH,2-1] * \
                      torch.where( xXH[cond1XH,1-1]>=0.0, \
                                   torch.tensor(-1.0,dtype=dtype, device=device), \
                                   torch.tensor(1.0,dtype=dtype, device=device) )
    #yXH[cond1XH,1-1] = -aXH*xXH[cond1XH,2-1]
    #yXH[xXH[...,1-1]<0.0,1-1] *= -1.0
    #yXH[xXH[...,1-1]<0.0,1-1].mul_(-1.0)

    yXH[...,2-1]=1.0
    yXH[cond1XH,2-1] = torch.abs(aXH * xXH[cond1XH,1-1])
    #y[3] is not used

    xx11XH = xXH[...,1-1]**2
    xx21XH = xXH[...,2-1]*xXH[...,1-1]
    xx22XH = xXH[...,2-1]**2
    xx31XH = xXH[...,3-1]*xXH[...,1-1]
    xx32XH = xXH[...,3-1]*xXH[...,2-1]
    xx33XH = xXH[...,3-1]**2
    zz31XH = zXH[...,3-1]*zXH[...,1-1]
    zz32XH = zXH[...,3-1]*zXH[...,2-1]
    zz33XH = zXH[...,3-1]**2
    yyzz11XH = yXH[...,1-1]**2 + zXH[...,1-1]**2
    yyzz21XH = yXH[...,2-1]*yXH[...,1-1] + zXH[...,2-1]*zXH[...,1-1]
    yyzz22XH = yXH[...,2-1]**2 + zXH[...,2-1]**2

    wXH = torch.zeros(riXH.shape[0],10,dtype=dtype, device=device)
    # (s s/s s)
    wXH[...,1-1] = riXH[...,1-1]
    ## (px s/s s)
    #w[2] = ri[2]*x[1]
    wXH[...,2-1] = riXH[...,2-1]*xXH[...,1-1]
    ## (px px/s s)
    #w[3] = ri[3]*xx11 + ri[4]*yyzz11
    wXH[...,3-1] = riXH[...,3-1]*xx11XH + riXH[...,4-1]*yyzz11XH
    ## (py s/s s)
    #w[4] = ri[2]*x[2]
    wXH[...,4-1] = riXH[...,2-1]*xXH[...,2-1]
    ## (py px/s s)
    #w[5] = ri[3]*xx21 + ri[4]*yyzz21
    wXH[...,5-1] = riXH[...,3-1]*xx21XH + riXH[...,4-1]*yyzz21XH
    ## (py py/s s)
    #w[6] = ri[3]*xx22 + ri[4]*yyzz22
    wXH[...,6-1] = riXH[...,3-1]*xx22XH + riXH[...,4-1]*yyzz22XH
    ## (pz s/ss)
    #w[7] = ri[2]*x[3]
    wXH[...,7-1] = riXH[...,2-1]*xXH[...,3-1]
    ## (pz px/s s)
    #w[8] = ri[3]*xx31 + ri[4]*zz31
    wXH[...,8-1] = riXH[...,3-1]*xx31XH + riXH[...,4-1]*zz31XH
    ## (pz py/s s)
    #w[9] = ri[3]*xx32 + ri[4]*zz32
    wXH[...,9-1] = riXH[...,3-1]*xx32XH + riXH[...,4-1]*zz32XH
    ## (pz pz/s s)
    #w[10] = ri[3]*xx33 + ri[4]*zz33
    wXH[...,10-1] = riXH[...,3-1]*xx33XH + riXH[...,4-1]*zz33XH

    ##############################################
    # X-X heavy atom - heavy atom
    K = XX | YX | YY
    x=-xij[K]
    y=torch.zeros(x.shape[0],2,dtype=dtype, device=device)
    #z=torch.zeros_like(x)
    #cond1 = torch.abs(x[...,3-1])>0.99999999
    #x[...,3-1] = torch.where(x[...,3-1]>0.99999999, torch.tensor([1.0],dtype=dtype, x[...,3-1]))
    #x[...,3-1] = torch.where(x[...,3-1]<-0.99999999, torch.tensor([-1.0],dtype=dtype, x[...,3-1]))
    #z[...,3-1] = torch,where(cond1, torch.tensor([0.0],dtype=dtype),
    #                           torch.sqrt(1.0-x[...,3-1]**2))
    """
    #not working in the new version as a depends on z
    z[...,3-1] =  torch.sqrt(1.0-x[...,3-1]**2)
    cond1XX = z[...,3-1]>1.0e-5
    #cond1 = torch.abs(x[...,3-1])>0.99999999, but z[3] is used in dominator, so change to
    #cond1 = z[...,3-1]>0.0
    #abs(x[3])>0.99999999 ==> z[3]<1.4142e-4
    #don't use th.where, use indexing to get rid of singularity
    a = 1.0/z[cond1XX,3-1]
    z[...,1-1] = 1.0
    z[cond1XX,1-1] = -a*x[cond1XX,1-1]*x[cond1XX,3-1]

    #z[...,2-1]=0.0
    z[cond1XX,2-1] = -a*x[cond1XX,2-1]*x[cond1XX,3-1]
    """
    #modify the code

    #z2 = torch.sqrt(1.0-x[...,3-1]**2)
    cond_x2 = torch.abs(x[...,3-1])<1.0
    z2 = torch.zeros_like(x[...,2])
    z2[cond_x2] = torch.sqrt(1.0-x[cond_x2,3-1]**2)
    cond1XX = z2>1.0e-5
    a = 1.0/z2[cond1XX]
    z0 = torch.ones_like(z2)
    z0[cond1XX] = -a*x[cond1XX,1-1]*x[cond1XX,3-1]
    z1 = torch.zeros_like(z2)
    z1[cond1XX] = -a*x[cond1XX,2-1]*x[cond1XX,3-1]

    z = torch.stack((z0,z1,z2),dim=1)

    #y[...,1-1]=0.0
    #y[cond1XX,0] =  a*x[cond1XX,2-1] * \
    #                  torch.where( x[cond1XX,1-1]>=0.0, \
    #                               torch.tensor(-1.0,dtype=dtype, device=device), \
    #                               torch.tensor(1.0,dtype=dtype, device=device) )
    #y[cond1XX,1-1] = -a*x[cond1XX,2-1]
    #y[x[...,1-1]<0.0,1-1] *= -1.0

    y[...,1-1]=0.0
    cond1XX_X1g0 = cond1XX & ( x[...,1-1]>=0.0 )
    cond1XX_X1l0 = cond1XX & ( x[...,1-1]<0.0 )
    y[cond1XX_X1g0,1-1] = -(1.0/z2[cond1XX_X1g0])*x[cond1XX_X1g0,2-1]
    y[cond1XX_X1l0,1-1] = (1.0/z2[cond1XX_X1l0])*x[cond1XX_X1l0,2-1]



    y[...,2-1]=1.0
    y[cond1XX,2-1] = torch.abs(a * x[cond1XX,1-1])
    #y[3] is not used


    xx11 = x[...,1-1]**2
    xx21 = x[...,2-1]*x[...,1-1]
    xx22 = x[...,2-1]**2
    xx31 = x[...,3-1]*x[...,1-1]
    xx32 = x[...,3-1]*x[...,2-1]
    xx33 = x[...,3-1]**2
    yy11 = y[...,1-1]**2
    yy21 = y[...,2-1]*y[...,1-1]
    yy22 = y[...,2-1]**2
    zz11 = z[...,1-1]**2
    zz21 = z[...,2-1]*z[...,1-1]
    zz22 = z[...,2-1]**2
    zz31 = z[...,3-1]*z[...,1-1]
    zz32 = z[...,3-1]*z[...,2-1]
    zz33 = z[...,3-1]**2
    yyzz11 = yy11+zz11
    yyzz21 = yy21+zz21
    yyzz22 = yy22+zz22
    xy11 = 2.0*x[...,1-1]*y[...,1-1]
    xy21 = x[...,1-1]*y[...,2-1]+x[...,2-1]*y[...,1-1]
    xy22 = 2.0*x[...,2-1]*y[...,2-1]
    xy31 = x[...,3-1]*y[...,1-1]
    xy32 = x[...,3-1]*y[...,2-1]
    xz11 = 2.0*x[...,1-1]*z[...,1-1]
    xz21 = x[...,1-1]*z[...,2-1]+x[...,2-1]*z[...,1-1]
    xz22 = 2.0*x[...,2-1]*z[...,2-1]
    xz31 = x[...,1-1]*z[...,3-1]+x[...,3-1]*z[...,1-1]
    xz32 = x[...,2-1]*z[...,3-1]+x[...,3-1]*z[...,2-1]
    xz33 = 2.0*x[...,3-1]*z[...,3-1]
    yz11 = 2.0*y[...,1-1]*z[...,1-1]
    yz21 = y[...,1-1]*z[...,2-1]+y[...,2-1]*z[...,1-1]
    yz22 = 2.0*y[...,2-1]*z[...,2-1]
    yz31 = y[...,1-1]*z[...,3-1]
    yz32 = y[...,2-1]*z[...,3-1]
    
    w = torch.zeros(ri.shape[0],100,dtype=dtype, device=device)



    ##(S S/S S)
    #w[1] = ri[1]
    w[...,1-1] = ri[...,1-1]
    ## (s s/px s)
    #w[2] = ri[5]*x[1]
    w[...,2-1] = ri[...,5-1]*x[...,1-1]
    ## (s s/px px)
    #w[3] = ri[11]*xx11 + ri[12]*yyzz11
    w[...,3-1] = ri[...,11-1]*xx11 + ri[...,12-1]*yyzz11
    ## (s s/py s)
    #w[4] = ri[5]*x[2]
    w[...,4-1] = ri[...,5-1]*x[...,2-1]
    ## (s s/py px)
    #w[5] = ri[11]*xx21 + ri[12]*yyzz21
    w[...,5-1] = ri[...,11-1]*xx21 + ri[...,12-1]*yyzz21
    ## (s s/py py)
    #w[6] = ri[11]*xx22 + ri[12]*yyzz22
    w[...,6-1] = ri[...,11-1]*xx22 + ri[...,12-1]*yyzz22
    ## (s s/pz s)
    #w[7] = ri[5]*x[3]
    w[...,7-1] = ri[...,5-1]*x[...,3-1]
    ## (s s/pz px)
    #w[8] = ri[11]*xx31 + ri[12]*zz31
    w[...,8-1] = ri[...,11-1]*xx31 + ri[...,12-1]*zz31
    ## (s s/pz py)
    #w[9] = ri[11]*xx32 + ri[12]*zz32
    w[...,9-1] = ri[...,11-1]*xx32 + ri[...,12-1]*zz32
    ## (s s/pz pz)
    #w[10] = ri[11]*xx33 + ri[12]*zz33
    w[...,10-1] = ri[...,11-1]*xx33 + ri[...,12-1]*zz33
    ## (px s/s s)
    #w[11] = ri[2]*x[1]
    w[...,11-1] = ri[...,2-1]*x[...,1-1]
    # (px s/px s)
    w[...,12-1] = ri[...,6-1]*xx11 + ri[...,7-1]*yyzz11
    # (px s/px px)
    w[...,13-1] = x[...,1-1] * (ri[...,13-1]*xx11 + ri[...,14-1]*yyzz11) \
          + ri[...,15-1] * (y[...,1-1]*xy11 + z[...,1-1]*xz11)
    # (px s/py s)
    w[...,14-1] = ri[...,6-1]*xx21 + ri[...,7-1]*yyzz21
    # (px s/py px)
    w[...,15-1] = x[...,1-1] * (ri[...,13-1]*xx21 + ri[...,14-1]*yyzz21) \
          + ri[...,15-1] * (y[...,1-1]*xy21 + z[...,1-1]*xz21)
    # (px s/py py)
    w[...,16-1] = x[...,1-1] * (ri[...,13-1]*xx22 + ri[...,14-1]*yyzz22) \
          + ri[...,15-1] * (y[...,1-1]*xy22 + z[...,1-1]*xz22)
    # (px s/pz s)
    w[...,17-1] = ri[...,6-1]*xx31 + ri[...,7-1]*zz31
    # (px s/pz px)
    w[...,18-1] = x[...,1-1] * (ri[...,13-1]*xx31 + ri[...,14-1]*zz31) \
          + ri[...,15-1] * (y[...,1-1]*xy31 + z[...,1-1]*xz31)
    # (px s/pz py)
    w[...,19-1] = x[...,1-1] * (ri[...,13-1]*xx32 + ri[...,14-1]*zz32) \
          + ri[...,15-1] * (y[...,1-1]*xy32 + z[...,1-1]*xz32)
    # (px s/pz pz)
    w[...,20-1] = x[...,1-1] * (ri[...,13-1]*xx33 + ri[...,14-1]*zz33) \
          + ri[...,15-1] * (     z[...,1-1]*xz33)
    # (px px/s s)
    w[...,21-1] = ri[...,3-1]*xx11 + ri[...,4-1]*yyzz11
    # (px px/px s)
    w[...,22-1] = x[...,1-1] * (ri[...,8-1]*xx11 + ri[...,9-1]*yyzz11) \
          + ri[...,10-1] * (y[...,1-1]*xy11 + z[...,1-1]*xz11)
    # (px px/px px)
    w[...,23-1] =  \
       (ri[...,16-1]*xx11 + ri[...,17-1]*yyzz11 ) * xx11 + ri[...,18-1]*xx11*yyzz11 \
       + ri[...,19-1] * (yy11*yy11 + zz11*zz11) \
       + ri[...,20-1] * (xy11*xy11 + xz11*xz11) \
       + ri[...,21-1] * (yy11*zz11 + zz11*yy11) \
       + ri[...,22-1]*yz11*yz11
    # (px px/py s)
    w[...,24-1] = x[...,2-1] * (ri[...,8-1]*xx11 + ri[...,9-1]*yyzz11) \
          + ri[...,10-1] * (y[...,2-1]*xy11 + z[...,2-1]*xz11)
    # (px px/py px)
    w[...,25-1] =  \
       (ri[...,16-1]*xx11 + ri[...,17-1]*yyzz11 ) * xx21 + ri[...,18-1]*xx11*yyzz21 \
       + ri[...,19-1] * (yy11*yy21 + zz11*zz21) \
       + ri[...,20-1] * (xy11*xy21 + xz11*xz21) \
       + ri[...,21-1] * (yy11*zz21 + zz11*yy21) \
       + ri[...,22-1]*yz11*yz21
    # (px px/py py)
    w[...,26-1] =  \
       (ri[...,16-1]*xx11 + ri[...,17-1]*yyzz11 ) * xx22 + ri[...,18-1]*xx11*yyzz22 \
       + ri[...,19-1] * (yy11*yy22 + zz11*zz22) \
       + ri[...,20-1] * (xy11*xy22 + xz11*xz22) \
       + ri[...,21-1] * (yy11*zz22 + zz11*yy22) \
       + ri[...,22-1]*yz11*yz22
    # (px px/pz s)
    w[...,27-1] = x[...,3-1] * (ri[...,8-1]*xx11 + ri[...,9-1]*yyzz11) \
          + ri[...,10-1] * z[...,3-1]*xz11
    # (px px/pz px)
    w[...,28-1] =  \
       (ri[...,16-1]*xx11 + ri[...,17-1]*yyzz11 ) * xx31 \
       + (ri[...,18-1]*xx11 + ri[...,19-1]*zz11 + ri[...,21-1]*yy11 ) * zz31 \
       + ri[...,20-1] * (xy11*xy31 + xz11*xz31) \
       + ri[...,22-1]*yz11*yz31
    # (px px/pz py)
    w[...,29-1] =  \
       (ri[...,16-1]*xx11 + ri[...,17-1]*yyzz11 ) * xx32 \
       + (ri[...,18-1]*xx11 + ri[...,19-1]*zz11 + ri[...,21-1]*yy11 ) * zz32 \
       + ri[...,20-1] * (xy11*xy32 + xz11*xz32) \
       + ri[...,22-1]*yz11*yz32
    # (px px/pz pz)
    w[...,30-1] =  \
       (ri[...,16-1]*xx11 + ri[...,17-1]*yyzz11 ) * xx33 \
       + (ri[...,18-1]*xx11 + ri[...,19-1]*zz11 + ri[...,21-1]*yy11 ) * zz33 \
       + ri[...,20-1]*xz11*xz33
    # (py s/s s)
    w[...,31-1] = ri[...,2-1]*x[...,2-1]
    # (py s/px s)
    w[...,32-1] = ri[...,6-1]*xx21 + ri[...,7-1]*yyzz21
    # (py s/px px)
    w[...,33-1] = x[...,2-1] * (ri[...,13-1]*xx11 + ri[...,14-1]*yyzz11) \
          + ri[...,15-1] * (y[...,2-1]*xy11 + z[...,2-1]*xz11)
    # (py s/py s)
    w[...,34-1] = ri[...,6-1]*xx22 + ri[...,7-1]*yyzz22
    # (py s/py px)
    w[...,35-1] = x[...,2-1] * (ri[...,13-1]*xx21 + ri[...,14-1]*yyzz21) \
          + ri[...,15-1] * (y[...,2-1]*xy21 + z[...,2-1]*xz21)
    # (py s/py py)
    w[...,36-1] = x[...,2-1] * (ri[...,13-1]*xx22 + ri[...,14-1]*yyzz22) \
          + ri[...,15-1] * (y[...,2-1]*xy22 + z[...,2-1]*xz22)
    # (py s/pz s)
    w[...,37-1] = ri[...,6-1]*xx32 + ri[...,7-1]*zz32
    # (py s/pz px)
    w[...,38-1] = x[...,2-1] * (ri[...,13-1]*xx31 + ri[...,14-1]*zz31) \
          + ri[...,15-1] * (y[...,2-1]*xy31 + z[...,2-1]*xz31)
    # (py s/pz py)
    w[...,39-1] = x[...,2-1] * (ri[...,13-1]*xx32 + ri[...,14-1]*zz32) \
          + ri[...,15-1] * (y[...,2-1]*xy32 + z[...,2-1]*xz32)
    # (py s/pz pz)
    w[...,40-1] = x[...,2-1] * (ri[...,13-1]*xx33 + ri[...,14-1]*zz33) \
          + ri[...,15-1] * z[...,2-1]*xz33
    # (py px/s s)
    w[...,41-1] = ri[...,3-1]*xx21 + ri[...,4-1]*yyzz21
    # (py px/px s)
    w[...,42-1] = x[...,1-1] * (ri[...,8-1]*xx21 + ri[...,9-1]*yyzz21) \
          + ri[...,10-1] * (y[...,1-1]*xy21 + z[...,1-1]*xz21)
    # (py px/px px)
    w[...,43-1] =  \
       (ri[...,16-1]*xx21 + ri[...,17-1]*yyzz21 ) * xx11 + ri[...,18-1]*xx21*yyzz11 \
       + ri[...,19-1] * (yy21*yy11 + zz21*zz11) \
       + ri[...,20-1] * (xy21*xy11 + xz21*xz11) \
       + ri[...,21-1] * (yy21*zz11 + zz21*yy11) \
       + ri[...,22-1]*yz21*yz11
    # (py px/py s)
    w[...,44-1] = x[...,2-1] * (ri[...,8-1]*xx21 + ri[...,9-1]*yyzz21) \
          + ri[...,10-1] * (y[...,2-1]*xy21 + z[...,2-1]*xz21)
    # (py px/py px)
    w[...,45-1] =  \
       (ri[...,16-1]*xx21 + ri[...,17-1]*yyzz21 ) * xx21 + ri[...,18-1]*xx21*yyzz21 \
       + ri[...,19-1] * (yy21*yy21 + zz21*zz21) \
       + ri[...,20-1] * (xy21*xy21 + xz21*xz21) \
       + ri[...,21-1] * (yy21*zz21 + zz21*yy21) \
       + ri[...,22-1]*yz21*yz21
    # (py px/py py)
    w[...,46-1] =  \
       (ri[...,16-1]*xx21 + ri[...,17-1]*yyzz21 ) * xx22 + ri[...,18-1]*xx21*yyzz22 \
       + ri[...,19-1] * (yy21*yy22 + zz21*zz22) \
       + ri[...,20-1] * (xy21*xy22 + xz21*xz22) \
       + ri[...,21-1] * (yy21*zz22 + zz21*yy22) \
       + ri[...,22-1]*yz21*yz22
    # (py px/pz s)
    w[...,47-1] = x[...,3-1] * (ri[...,8-1]*xx21 + ri[...,9-1]*yyzz21) \
          + ri[...,10-1] * z[...,3-1]*xz21
    # (py px/pz px)
    w[...,48-1] =  \
       (ri[...,16-1]*xx21 + ri[...,17-1]*yyzz21 ) * xx31 \
       + (ri[...,18-1]*xx21 + ri[...,19-1]*zz21 + ri[...,21-1]*yy21 ) * zz31 \
       + ri[...,20-1] * (xy21*xy31 + xz21*xz31) \
       + ri[...,22-1]*yz21*yz31
    # (py px/pz py)
    w[...,49-1] =  \
       (ri[...,16-1]*xx21 + ri[...,17-1]*yyzz21 ) * xx32 \
       + (ri[...,18-1]*xx21 + ri[...,19-1]*zz21 + ri[...,21-1]*yy21 ) * zz32 \
       + ri[...,20-1] * (xy21*xy32 + xz21*xz32) \
       + ri[...,22-1]*yz21*yz32
    # (py px/pz pz)
    w[...,50-1] =  \
       (ri[...,16-1]*xx21 + ri[...,17-1]*yyzz21 ) * xx33 \
       + (ri[...,18-1]*xx21 + ri[...,19-1]*zz21 + ri[...,21-1]*yy21 ) * zz33 \
       + ri[...,20-1]*xz21*xz33
    # (py py/s s)
    w[...,51-1] = ri[...,3-1]*xx22 + ri[...,4-1]*yyzz22
    # (py py/px s)
    w[...,52-1] = x[...,1-1] * (ri[...,8-1]*xx22 + ri[...,9-1]*yyzz22) \
          + ri[...,10-1] * (y[...,1-1]*xy22 + z[...,1-1]*xz22)
    # (py py/px px)
    w[...,53-1] =  \
       (ri[...,16-1]*xx22 + ri[...,17-1]*yyzz22 ) * xx11 + ri[...,18-1]*xx22*yyzz11 \
       + ri[...,19-1] * (yy22*yy11 + zz22*zz11) \
       + ri[...,20-1] * (xy22*xy11 + xz22*xz11) \
       + ri[...,21-1] * (yy22*zz11 + zz22*yy11) \
       + ri[...,22-1]*yz22*yz11
    # (py py/py s)
    w[...,54-1] = x[...,2-1] * (ri[...,8-1]*xx22 + ri[...,9-1]*yyzz22) \
          + ri[...,10-1] * (y[...,2-1]*xy22 + z[...,2-1]*xz22)
    # (py py/py px)
    w[...,55-1] =  \
       (ri[...,16-1]*xx22 + ri[...,17-1]*yyzz22 ) * xx21 + ri[...,18-1]*xx22*yyzz21 \
       + ri[...,19-1] * (yy22*yy21 + zz22*zz21) \
       + ri[...,20-1] * (xy22*xy21 + xz22*xz21) \
       + ri[...,21-1] * (yy22*zz21 + zz22*yy21) \
       + ri[...,22-1]*yz22*yz21
    # (py py/py py)
    w[...,56-1] =  \
       (ri[...,16-1]*xx22 + ri[...,17-1]*yyzz22 ) * xx22 + ri[...,18-1]*xx22*yyzz22 \
       + ri[...,19-1] * (yy22*yy22 + zz22*zz22) \
       + ri[...,20-1] * (xy22*xy22 + xz22*xz22) \
       + ri[...,21-1] * (yy22*zz22 + zz22*yy22) \
       + ri[...,22-1]*yz22*yz22
    # (py py/pz s)
    w[...,57-1] = x[...,3-1] * (ri[...,8-1]*xx22 + ri[...,9-1]*yyzz22) \
          + ri[...,10-1] *  z[...,3-1]*xz22
    # (py py/pz px)
    w[...,58-1] =  \
       (ri[...,16-1]*xx22 + ri[...,17-1]*yyzz22 ) * xx31 \
       + (ri[...,18-1]*xx22 + ri[...,19-1]*zz22 + ri[...,21-1]*yy22 ) * zz31 \
       + ri[...,20-1] * (xy22*xy31 + xz22*xz31) \
       + ri[...,22-1]*yz22*yz31
    # (py py/pz py)
    w[...,59-1] =  \
       (ri[...,16-1]*xx22 + ri[...,17-1]*yyzz22 ) * xx32 \
       + (ri[...,18-1]*xx22 + ri[...,19-1]*zz22 + ri[...,21-1]*yy22 ) * zz32 \
       + ri[...,20-1] * (xy22*xy32 + xz22*xz32) \
       + ri[...,22-1]*yz22*yz32
    # (py py/pz pz)
    w[...,60-1] =  \
       (ri[...,16-1]*xx22 + ri[...,17-1]*yyzz22 ) * xx33 \
       + (ri[...,18-1]*xx22 + ri[...,19-1]*zz22 + ri[...,21-1]*yy22 ) * zz33 \
       + ri[...,20-1]*xz22*xz33
    # (pz s/ss)
    w[...,61-1] = ri[...,2-1]*x[...,3-1]
    # (pz s/px s)
    w[...,62-1] = ri[...,6-1]*xx31 + ri[...,7-1]*zz31
    # (pz s/px px)
    w[...,63-1] = x[...,3-1] * (ri[...,13-1]*xx11 + ri[...,14-1]*yyzz11) \
          + ri[...,15-1] * z[...,3-1]*xz11
    # (pz s/py s)
    w[...,64-1] = ri[...,6-1]*xx32 + ri[...,7-1]*zz32
    # (pz s/py px)
    w[...,65-1] = x[...,3-1] * (ri[...,13-1]*xx21 + ri[...,14-1]*yyzz21) \
          + ri[...,15-1] * z[...,3-1]*xz21
    # (pz s/py py)
    w[...,66-1] = x[...,3-1] * (ri[...,13-1]*xx22 + ri[...,14-1]*yyzz22) \
          + ri[...,15-1] * z[...,3-1]*xz22
    # (pz s/pz s)
    w[...,67-1] = ri[...,6-1]*xx33 + ri[...,7-1]*zz33
    # (pz s/pz px)
    w[...,68-1] = x[...,3-1] * (ri[...,13-1]*xx31 + ri[...,14-1]*zz31) \
          + ri[...,15-1] *  z[...,3-1]*xz31
    # (pz s/pz py)
    w[...,69-1] = x[...,3-1] * (ri[...,13-1]*xx32 + ri[...,14-1]*zz32) \
          + ri[...,15-1] * z[...,3-1]*xz32
    # (pz s/pz pz)
    w[...,70-1] = x[...,3-1] * (ri[...,13-1]*xx33 + ri[...,14-1]*zz33) \
          + ri[...,15-1] * z[...,3-1]*xz33
    # (pz px/s s)
    w[...,71-1] = ri[...,3-1]*xx31 + ri[...,4-1]*zz31
    # (pz px/px s)
    w[...,72-1] = x[...,1-1] * (ri[...,8-1]*xx31 + ri[...,9-1]*zz31) \
          + ri[...,10-1] * (y[...,1-1]*xy31 + z[...,1-1]*xz31)
    # (pz px/px px)
    w[...,73-1] =  \
       (ri[...,16-1]*xx31 + ri[...,17-1]*zz31 ) * xx11 + ri[...,18-1]*xx31*yyzz11 \
       + ri[...,19-1]*zz31*zz11 \
       + ri[...,20-1] * (xy31*xy11 + xz31*xz11) \
       + ri[...,21-1]*zz31*yy11 \
       + ri[...,22-1]*yz31*yz11
    # (pz px/py s)
    w[...,74-1] = x[...,2-1] * (ri[...,8-1]*xx31 + ri[...,9-1]*zz31) \
          + ri[...,10-1] * (y[...,2-1]*xy31 + z[...,2-1]*xz31)
    # (pz px/py px)
    w[...,75-1] =  \
       (ri[...,16-1]*xx31 + ri[...,17-1]*zz31 ) * xx21 + ri[...,18-1]*xx31*yyzz21 \
       + ri[...,19-1]*zz31*zz21 \
       + ri[...,20-1] * (xy31*xy21 + xz31*xz21) \
       + ri[...,21-1]*zz31*yy21 \
       + ri[...,22-1]*yz31*yz21
    # (pz px/py py)
    w[...,76-1] =  \
       (ri[...,16-1]*xx31 + ri[...,17-1]*zz31 ) * xx22 + ri[...,18-1]*xx31*yyzz22 \
       + ri[...,19-1]*zz31*zz22 \
       + ri[...,20-1] * (xy31*xy22 + xz31*xz22) \
       + ri[...,21-1]*zz31*yy22 \
       + ri[...,22-1]*yz31*yz22
    # (pz px/pz s)
    w[...,77-1] = x[...,3-1] * (ri[...,8-1]*xx31 + ri[...,9-1]*zz31) \
          + ri[...,10-1] * z[...,3-1]*xz31
    # (pz px/pz px)
    w[...,78-1] =  \
       (ri[...,16-1]*xx31 + ri[...,17-1]*zz31 ) * xx31 \
       + (ri[...,18-1]*xx31 + ri[...,19-1]*zz31 ) * zz31 \
       + ri[...,20-1] * (xy31*xy31 + xz31*xz31) \
       + ri[...,22-1]*yz31*yz31
    # (pz px/pz py)
    w[...,79-1] =  \
       (ri[...,16-1]*xx31 + ri[...,17-1]*zz31 ) * xx32 \
       + (ri[...,18-1]*xx31 + ri[...,19-1]*zz31 ) * zz32 \
       + ri[...,20-1] * (xy31*xy32 + xz31*xz32) \
       + ri[...,22-1]*yz31*yz32
    # (pz px/pz pz)
    w[...,80-1] =  \
       (ri[...,16-1]*xx31 + ri[...,17-1]*zz31 ) * xx33 \
       + (ri[...,18-1]*xx31 + ri[...,19-1]*zz31 ) * zz33 \
       + ri[...,20-1]*xz31*xz33
    # (pz py/s s)
    w[...,81-1] = ri[...,3-1]*xx32 + ri[...,4-1]*zz32
    # (pz py/px s)
    w[...,82-1] = x[...,1-1] * (ri[...,8-1]*xx32 + ri[...,9-1]*zz32) \
          + ri[...,10-1] * (y[...,1-1]*xy32 + z[...,1-1]*xz32)
    # (pz py/px px)
    w[...,83-1] =  \
       (ri[...,16-1]*xx32 + ri[...,17-1]*zz32 ) * xx11 + ri[...,18-1]*xx32*yyzz11 \
       + ri[...,19-1]*zz32*zz11 \
       + ri[...,20-1] * (xy32*xy11 + xz32*xz11) \
       + ri[...,21-1]*zz32*yy11 \
       + ri[...,22-1]*yz32*yz11
    # (pz py/py s)
    w[...,84-1] = x[...,2-1] * (ri[...,8-1]*xx32 + ri[...,9-1]*zz32) \
          + ri[...,10-1] * (y[...,2-1]*xy32 + z[...,2-1]*xz32)
    # (pz py/py px)
    w[...,85-1] =  \
       (ri[...,16-1]*xx32 + ri[...,17-1]*zz32 ) * xx21 + ri[...,18-1]*xx32*yyzz21 \
       + ri[...,19-1]*zz32*zz21 \
       + ri[...,20-1] * (xy32*xy21 + xz32*xz21) \
       + ri[...,21-1]*zz32*yy21 \
       + ri[...,22-1]*yz32*yz21
    # (pz py/py py)
    w[...,86-1] =  \
       (ri[...,16-1]*xx32 + ri[...,17-1]*zz32 ) * xx22 + ri[...,18-1]*xx32*yyzz22 \
       + ri[...,19-1]*zz32*zz22 \
       + ri[...,20-1] * (xy32*xy22 + xz32*xz22) \
       + ri[...,21-1]*zz32*yy22 \
       + ri[...,22-1]*yz32*yz22
    # (pz py/pz s)
    w[...,87-1] = x[...,3-1] * (ri[...,8-1]*xx32 + ri[...,9-1]*zz32) \
          + ri[...,10-1] * z[...,3-1]*xz32
    # (pz py/pz px)
    w[...,88-1] =  \
       (ri[...,16-1]*xx32 + ri[...,17-1]*zz32 ) * xx31 \
       + (ri[...,18-1]*xx32 + ri[...,19-1]*zz32 ) * zz31 \
       + ri[...,20-1] * (xy32*xy31 + xz32*xz31) \
       + ri[...,22-1]*yz32*yz31
    # (pz py/pz py)
    w[...,89-1] =  \
       (ri[...,16-1]*xx32 + ri[...,17-1]*zz32 ) * xx32 \
       + (ri[...,18-1]*xx32 + ri[...,19-1]*zz32 ) * zz32 \
       + ri[...,20-1] * (xy32*xy32 + xz32*xz32) \
       + ri[...,22-1]*yz32*yz32
    # (pz py/pz pz)
    w[...,90-1] =  \
       (ri[...,16-1]*xx32 + ri[...,17-1]*zz32 ) * xx33 \
       + (ri[...,18-1]*xx32 + ri[...,19-1]*zz32 ) * zz33 \
       + ri[...,20-1]*xz32*xz33
    # (pz pz/s s)
    w[...,91-1] = ri[...,3-1]*xx33 + ri[...,4-1]*zz33
    # (pz pz/px s)
    w[...,92-1] = x[...,1-1] * (ri[...,8-1]*xx33 + ri[...,9-1]*zz33) \
          + ri[...,10-1] * z[...,1-1]*xz33
    # (pz pz/px px)
    w[...,93-1] =  \
       (ri[...,16-1]*xx33 + ri[...,17-1]*zz33 ) * xx11 + ri[...,18-1]*xx33*yyzz11 \
       + ri[...,19-1]*zz33*zz11 \
       + ri[...,20-1]*xz33*xz11 \
       + ri[...,21-1]*zz33*yy11
    # (pz pz/py s)
    w[...,94-1] = x[...,2-1] * (ri[...,8-1]*xx33 + ri[...,9-1]*zz33) \
          + ri[...,10-1] * z[...,2-1]*xz33
    # (pz pz/py px)
    w[...,95-1] =  \
       (ri[...,16-1]*xx33 + ri[...,17-1]*zz33 ) * xx21 + ri[...,18-1]*xx33*yyzz21 \
       + ri[...,19-1]*zz33*zz21 \
       + ri[...,20-1]*xz33*xz21 \
       + ri[...,21-1]*zz33*yy21
    # (pz pz/py py)
    w[...,96-1] =  \
       (ri[...,16-1]*xx33 + ri[...,17-1]*zz33 ) * xx22 + ri[...,18-1]*xx33*yyzz22 \
       + ri[...,19-1]*zz33*zz22 \
       + ri[...,20-1]*xz33*xz22 \
       + ri[...,21-1]*zz33*yy22
    # (pz pz/pz s)
    w[...,97-1] = x[...,3-1] * (ri[...,8-1]*xx33 + ri[...,9-1]*zz33) \
          + ri[...,10-1] * z[...,3-1]*xz33
    # (pz pz/pz px)
    w[...,98-1] =  \
       (ri[...,16-1]*xx33 + ri[...,17-1]*zz33 ) * xx31 \
       + (ri[...,18-1]*xx33 + ri[...,19-1]*zz33 ) * zz31 \
       + ri[...,20-1]*xz33*xz31
    # (pz pz/pz py)
    w[...,99-1] =  \
       (ri[...,16-1]*xx33 + ri[...,17-1]*zz33 ) * xx32 \
       + (ri[...,18-1]*xx33 + ri[...,19-1]*zz33 ) * zz32 \
       + ri[...,20-1]*xz33*xz32
    # (pz pz/pz pz)
    w[...,100-1] =  \
       (ri[...,16-1]*xx33 + ri[...,17-1]*zz33 ) * xx33 \
       + (ri[...,18-1]*xx33 + ri[...,19-1]*zz33 ) * zz33 \
       + ri[...,20-1]*xz33*xz33
    #
    """
    #
    css1,csp1,cpps1,cppp1 = core[1:,1]
    css2,csp2,cpps2,cppp2 = core[1:,2]
    e1b[1] = -css1
    #if(natorb(ni).eq.4) then
    if ni>1:
        # currently only s and p orbitals
        e1b[2] = -csp1 *x[1]
        e1b[3] = -cpps1*xx11-cppp1*yyzz11
        e1b[4] = -csp1 *x[2]
        e1b[5] = -cpps1*xx21-cppp1*yyzz21
        e1b[6] = -cpps1*xx22-cppp1*yyzz22
        e1b[7] = -csp1 *x[3]
        e1b[8] = -cpps1*xx31-cppp1*zz31
        e1b[9] = -cpps1*xx32-cppp1*zz32
        e1b[10] = -cpps1*xx33-cppp1*zz33

    e2a[1] = -css2
    #if(natorb(nj).eq.4) then
    if nj>1:
        e2a[2] = -csp2 *x[1]
        e2a[3] = -cpps2*xx11-cppp2*yyzz11
        e2a[4] = -csp2 *x[2]
        e2a[5] = -cpps2*xx21-cppp2*yyzz21
        e2a[6] = -cpps2*xx22-cppp2*yyzz22
        e2a[7] = -csp2 *x[3]
        e2a[8] = -cpps2*xx31-cppp2*zz31
        e2a[9] = -cpps2*xx32-cppp2*zz32
        e2a[10] = -cpps2*xx33-cppp2*zz33
    """
    """
    e1bHH = -coreHH[...,0]
    e2aHH = -coreHH[...,1]

    e1bXH = torch.zeros(coreXH.shape[0],10,dtype=dtype, device=device)
    e1bXH[...,1-1] = -coreXH[...,0]
    e1bXH[...,2-1] = -coreXH[...,1]*xXH[...,1-1]
    e1bXH[...,3-1] = -coreXH[...,2]*xx11XH - coreXH[...,3]*yyzz11XH
    e1bXH[...,4-1] = -coreXH[...,1]*xXH[...,2-1]
    e1bXH[...,5-1] = -coreXH[...,2]*xx21XH - coreXH[...,3]*yyzz21XH
    e1bXH[...,6-1] = -coreXH[...,2]*xx22XH - coreXH[...,3]*yyzz22XH
    e1bXH[...,7-1] = -coreXH[...,1]*xXH[...,3-1]
    e1bXH[...,8-1] = -coreXH[...,2]*xx31XH - coreXH[...,3]*zz31XH
    e1bXH[...,9-1] = -coreXH[...,2]*xx32XH - coreXH[...,3]*zz32XH
    e1bXH[...,10-1] = -coreXH[...,2]*xx33XH - coreXH[...,3]*zz33XH
    e2aXH = -coreXH[...,4]

    e1b = torch.zeros(core.shape[0],10,dtype=dtype, device=device)
    e2a = torch.zeros_like(e1b)
    e1b[...,1-1] = -core[...,0]
    e1b[...,2-1] = -core[...,1]*x[...,1-1]
    e1b[...,3-1] = -core[...,2]*xx11 - core[...,3]*yyzz11
    e1b[...,4-1] = -core[...,1]*x[...,2-1]
    e1b[...,5-1] = -core[...,2]*xx21 - core[...,3]*yyzz21
    e1b[...,6-1] = -core[...,2]*xx22 - core[...,3]*yyzz22
    e1b[...,7-1] = -core[...,1]*x[...,3-1]
    e1b[...,8-1] = -core[...,2]*xx31 - core[...,3]*zz31
    e1b[...,9-1] = -core[...,2]*xx32 - core[...,3]*zz32
    e1b[...,10-1] = -core[...,2]*xx33 - core[...,3]*zz33

    e2a[...,1-1] = -core[...,4]
    e2a[...,2-1] = -core[...,5]*x[...,1-1]
    e2a[...,3-1] = -core[...,6]*xx11 - core[...,7]*yyzz11
    e2a[...,4-1] = -core[...,5]*x[...,2-1]
    e2a[...,5-1] = -core[...,6]*xx21 - core[...,7]*yyzz21
    e2a[...,6-1] = -core[...,6]*xx22 - core[...,7]*yyzz22
    e2a[...,7-1] = -core[...,5]*x[...,3-1]
    e2a[...,8-1] = -core[...,6]*xx31 - core[...,7]*zz31
    e2a[...,9-1] = -core[...,6]*xx32 - core[...,7]*zz32
    e2a[...,10-1] = -core[...,6]*xx33 - core[...,7]*zz33

    return wHH, e1bHH, e2aHH, wXH, e1bXH, e2aXH, w, e1b, e2a
    """
    #combine w, e1b, e2a




    # as index_add_ is used later, which is slow, so
    # change e1b, e2a to shape (npairs, 4,4), only need to do index_add once
    """
    e1b = torch.zeros(rij.shape[0],10,dtype=dtype, device=device)
    e2a = torch.zeros_like(e1b)

    e1b[HH,0] = -coreHH[...,0]
    e2a[HH,0] = -coreHH[...,1]

    #e1bXH = torch.zeros(coreXH.shape[0],10,dtype=dtype, device=device)
    e1b[XH,1-1] = -coreXH[...,0]
    e1b[XH,2-1] = -coreXH[...,1]*xXH[...,1-1]
    e1b[XH,3-1] = -coreXH[...,2]*xx11XH - coreXH[...,3]*yyzz11XH
    e1b[XH,4-1] = -coreXH[...,1]*xXH[...,2-1]
    e1b[XH,5-1] = -coreXH[...,2]*xx21XH - coreXH[...,3]*yyzz21XH
    e1b[XH,6-1] = -coreXH[...,2]*xx22XH - coreXH[...,3]*yyzz22XH
    e1b[XH,7-1] = -coreXH[...,1]*xXH[...,3-1]
    e1b[XH,8-1] = -coreXH[...,2]*xx31XH - coreXH[...,3]*zz31XH
    e1b[XH,9-1] = -coreXH[...,2]*xx32XH - coreXH[...,3]*zz32XH
    e1b[XH,10-1] = -coreXH[...,2]*xx33XH - coreXH[...,3]*zz33XH
    e2a[XH,0] = -coreXH[...,4]

    #e1b = torch.zeros(core.shape[0],10,dtype=dtype, device=device)
    #e2a = torch.zeros_like(e1b)
    e1b[XX,1-1] = -core[...,0]
    e1b[XX,2-1] = -core[...,1]*x[...,1-1]
    e1b[XX,3-1] = -core[...,2]*xx11 - core[...,3]*yyzz11
    e1b[XX,4-1] = -core[...,1]*x[...,2-1]
    e1b[XX,5-1] = -core[...,2]*xx21 - core[...,3]*yyzz21
    e1b[XX,6-1] = -core[...,2]*xx22 - core[...,3]*yyzz22
    e1b[XX,7-1] = -core[...,1]*x[...,3-1]
    e1b[XX,8-1] = -core[...,2]*xx31 - core[...,3]*zz31
    e1b[XX,9-1] = -core[...,2]*xx32 - core[...,3]*zz32
    e1b[XX,10-1] = -core[...,2]*xx33 - core[...,3]*zz33

    e2a[XX,1-1] = -core[...,4]
    e2a[XX,2-1] = -core[...,5]*x[...,1-1]
    e2a[XX,3-1] = -core[...,6]*xx11 - core[...,7]*yyzz11
    e2a[XX,4-1] = -core[...,5]*x[...,2-1]
    e2a[XX,5-1] = -core[...,6]*xx21 - core[...,7]*yyzz21
    e2a[XX,6-1] = -core[...,6]*xx22 - core[...,7]*yyzz22
    e2a[XX,7-1] = -core[...,5]*x[...,3-1]
    e2a[XX,8-1] = -core[...,6]*xx31 - core[...,7]*zz31
    e2a[XX,9-1] = -core[...,6]*xx32 - core[...,7]*zz32
    e2a[XX,10-1] = -core[...,6]*xx33 - core[...,7]*zz33
    """

    e1b = torch.zeros((rij.shape[0],4,4),dtype=dtype, device=device)
    e2a = torch.zeros_like(e1b)

    e1b[HH,0,0] = -coreHH[...,0]
    e2a[HH,0,0] = -coreHH[...,1]

    #e1bXH = torch.zeros(coreXH.shape[0],10,dtype=dtype, device=device)
    e1b[XH,0,0] = -coreXH[...,0]
    e1b[XH,0,1] = -coreXH[...,1]*xXH[...,1-1]
    e1b[XH,1,1] = -coreXH[...,2]*xx11XH - coreXH[...,3]*yyzz11XH
    e1b[XH,0,2] = -coreXH[...,1]*xXH[...,2-1]
    e1b[XH,1,2] = -coreXH[...,2]*xx21XH - coreXH[...,3]*yyzz21XH
    e1b[XH,2,2] = -coreXH[...,2]*xx22XH - coreXH[...,3]*yyzz22XH
    e1b[XH,0,3] = -coreXH[...,1]*xXH[...,3-1]
    e1b[XH,1,3] = -coreXH[...,2]*xx31XH - coreXH[...,3]*zz31XH
    e1b[XH,2,3] = -coreXH[...,2]*xx32XH - coreXH[...,3]*zz32XH
    e1b[XH,3,3] = -coreXH[...,2]*xx33XH - coreXH[...,3]*zz33XH
    e2a[XH,0,0] = -coreXH[...,4]
    #e1b = torch.zeros(core.shape[0],10,dtype=dtype, device=device)
    #e2a = torch.zeros_like(e1b)
    e1b[XX,0,0] = -core[...,0]
    e1b[XX,0,1] = -core[...,1]*x[...,1-1]
    e1b[XX,1,1] = -core[...,2]*xx11 - core[...,3]*yyzz11
    e1b[XX,0,2] = -core[...,1]*x[...,2-1]
    e1b[XX,1,2] = -core[...,2]*xx21 - core[...,3]*yyzz21
    e1b[XX,2,2] = -core[...,2]*xx22 - core[...,3]*yyzz22
    e1b[XX,0,3] = -core[...,1]*x[...,3-1]
    e1b[XX,1,3] = -core[...,2]*xx31 - core[...,3]*zz31
    e1b[XX,2,3] = -core[...,2]*xx32 - core[...,3]*zz32
    e1b[XX,3,3] = -core[...,2]*xx33 - core[...,3]*zz33

    e2a[XX,0,0] = -core[...,4]
    e2a[XX,0,1] = -core[...,5]*x[...,1-1]
    e2a[XX,1,1] = -core[...,6]*xx11 - core[...,7]*yyzz11
    e2a[XX,0,2] = -core[...,5]*x[...,2-1]
    e2a[XX,1,2] = -core[...,6]*xx21 - core[...,7]*yyzz21
    e2a[XX,2,2] = -core[...,6]*xx22 - core[...,7]*yyzz22
    e2a[XX,0,3] = -core[...,5]*x[...,3-1]
    e2a[XX,1,3] = -core[...,6]*xx31 - core[...,7]*zz31
    e2a[XX,2,3] = -core[...,6]*xx32 - core[...,7]*zz32
    e2a[XX,3,3] = -core[...,6]*xx33 - core[...,7]*zz33
#    t0 = time.time()
    if themethod == "PM6":
#        t0 = time.time()
        dRotationMatrix = GenerateRotationMatrix(xij)
        riYH, riYX, riYY, coreYH, coreYX, coreYY = \
                TETCILFDO(ni,nj,rij, tore, \
                da, db, qa,qb, dpa, dpb, dsa, dsb, dda, ddb, \
                rho0a,rho0b, rho1a,rho1b, rho2ad,rho2bd, rho3a,rho3b, rho4a,rho4b, rho5a,rho5b, rho6a,rho6b, diadia, themethod, dRotationMatrix,ri,riXH,rho_corea,rho_coreb, riXHPM6,riPM6a, riPM6b)


        e1bD = torch.zeros((rij.shape[0],9,9),dtype=dtype, device=device)
        e2aD = torch.zeros_like(e1bD)
        ########## HH ##########
#        e1bD[HH,0,0] = e1b[HH,0,0]
#        e2aD[HH,0,0] = e2a[HH,0,0] 
 
        ########## XH ##########
#        e1bD[XH,0,0] = e1b[XH,0,0]
#        e1bD[XH,0,1] = e1b[XH,0,1]
#        e1bD[XH,1,1] = e1b[XH,1,1]
#        e1bD[XH,0,2] = e1b[XH,0,2]
#        e1bD[XH,1,2] = e1b[XH,1,2]
#        e1bD[XH,2,2] = e1b[XH,2,2]
#        e1bD[XH,0,3] = e1b[XH,0,3]
#        e1bD[XH,1,3] = e1b[XH,1,3]
#        e1bD[XH,2,3] = e1b[XH,2,3]
#        e1bD[XH,3,3] = e1b[XH,3,3]
#        e2aD[XH,0,0] = e2a[XH,0,0]
        e1bD[ HH | XH | XX, :4, :4] = e2a
        e2aD[ HH | XH | XX, :4, :4] = e1b

        ########## YH ##########
        e1bD[YH,0,0] = -coreYH[...,0]
        e2aD[YH,0,0] = -coreYH[...,1]
        e2aD[YH,0,1] = -coreYH[...,2]
        e2aD[YH,1,1] = -coreYH[...,3]
        e2aD[YH,0,2] = -coreYH[...,4]
        e2aD[YH,1,2] = -coreYH[...,5]
        e2aD[YH,2,2] = -coreYH[...,6]
        e2aD[YH,0,3] = -coreYH[...,7]
        e2aD[YH,1,3] = -coreYH[...,8]
        e2aD[YH,2,3] = -coreYH[...,9]
        e2aD[YH,3,3] = -coreYH[...,10]
        e2aD[YH,0,4] = -coreYH[...,11]
        e2aD[YH,1,4] = -coreYH[...,12]
        e2aD[YH,2,4] = -coreYH[...,13]
        e2aD[YH,3,4] = -coreYH[...,14]
        e2aD[YH,4,4] = -coreYH[...,15]
        e2aD[YH,0,5] = -coreYH[...,16]
        e2aD[YH,1,5] = -coreYH[...,17]
        e2aD[YH,2,5] = -coreYH[...,18]
        e2aD[YH,3,5] = -coreYH[...,19]
        e2aD[YH,4,5] = -coreYH[...,20]
        e2aD[YH,5,5] = -coreYH[...,21]
        e2aD[YH,0,6] = -coreYH[...,22]
        e2aD[YH,1,6] = -coreYH[...,23]
        e2aD[YH,2,6] = -coreYH[...,24]
        e2aD[YH,3,6] = -coreYH[...,25]
        e2aD[YH,4,6] = -coreYH[...,26]
        e2aD[YH,5,6] = -coreYH[...,27]
        e2aD[YH,6,6] = -coreYH[...,28]
        e2aD[YH,0,7] = -coreYH[...,29]
        e2aD[YH,1,7] = -coreYH[...,30]
        e2aD[YH,2,7] = -coreYH[...,31]
        e2aD[YH,3,7] = -coreYH[...,32]
        e2aD[YH,4,7] = -coreYH[...,33]
        e2aD[YH,5,7] = -coreYH[...,34]
        e2aD[YH,6,7] = -coreYH[...,35]
        e2aD[YH,7,7] = -coreYH[...,36]
        e2aD[YH,0,8] = -coreYH[...,37]
        e2aD[YH,1,8] = -coreYH[...,38]
        e2aD[YH,2,8] = -coreYH[...,39]
        e2aD[YH,3,8] = -coreYH[...,40]
        e2aD[YH,4,8] = -coreYH[...,41]
        e2aD[YH,5,8] = -coreYH[...,42]
        e2aD[YH,6,8] = -coreYH[...,43]
        e2aD[YH,7,8] = -coreYH[...,44]
        e2aD[YH,8,8] = -coreYH[...,45]

        ########## YX ##########
        e1bD[YX,0,0] = -coreYX[...,0]
        e1bD[YX,0,1] = -coreYX[...,1]
        e1bD[YX,1,1] = -coreYX[...,2]
        e1bD[YX,0,2] = -coreYX[...,3]
        e1bD[YX,1,2] = -coreYX[...,4]
        e1bD[YX,2,2] = -coreYX[...,5]
        e1bD[YX,0,3] = -coreYX[...,6]
        e1bD[YX,1,3] = -coreYX[...,7]
        e1bD[YX,2,3] = -coreYX[...,8]
        e1bD[YX,3,3] = -coreYX[...,9]


        e2aD[YX,0,0] = -coreYX[...,10]
        e2aD[YX,0,1] = -coreYX[...,11]
        e2aD[YX,1,1] = -coreYX[...,12]
        e2aD[YX,0,2] = -coreYX[...,13]
        e2aD[YX,1,2] = -coreYX[...,14]
        e2aD[YX,2,2] = -coreYX[...,15]
        e2aD[YX,0,3] = -coreYX[...,16]
        e2aD[YX,1,3] = -coreYX[...,17]
        e2aD[YX,2,3] = -coreYX[...,18]
        e2aD[YX,3,3] = -coreYX[...,19]
        e2aD[YX,0,4] = -coreYX[...,20]
        e2aD[YX,1,4] = -coreYX[...,21]
        e2aD[YX,2,4] = -coreYX[...,22]
        e2aD[YX,3,4] = -coreYX[...,23]
        e2aD[YX,4,4] = -coreYX[...,24]
        e2aD[YX,0,5] = -coreYX[...,25]
        e2aD[YX,1,5] = -coreYX[...,26]
        e2aD[YX,2,5] = -coreYX[...,27]
        e2aD[YX,3,5] = -coreYX[...,28]
        e2aD[YX,4,5] = -coreYX[...,29]
        e2aD[YX,5,5] = -coreYX[...,30]
        e2aD[YX,0,6] = -coreYX[...,31]
        e2aD[YX,1,6] = -coreYX[...,32]
        e2aD[YX,2,6] = -coreYX[...,33]
        e2aD[YX,3,6] = -coreYX[...,34]
        e2aD[YX,4,6] = -coreYX[...,35]
        e2aD[YX,5,6] = -coreYX[...,36]
        e2aD[YX,6,6] = -coreYX[...,37]
        e2aD[YX,0,7] = -coreYX[...,38]
        e2aD[YX,1,7] = -coreYX[...,39]
        e2aD[YX,2,7] = -coreYX[...,40]
        e2aD[YX,3,7] = -coreYX[...,41]
        e2aD[YX,4,7] = -coreYX[...,42]
        e2aD[YX,5,7] = -coreYX[...,43]
        e2aD[YX,6,7] = -coreYX[...,44]
        e2aD[YX,7,7] = -coreYX[...,45]
        e2aD[YX,0,8] = -coreYX[...,46]
        e2aD[YX,1,8] = -coreYX[...,47]
        e2aD[YX,2,8] = -coreYX[...,48]
        e2aD[YX,3,8] = -coreYX[...,49]
        e2aD[YX,4,8] = -coreYX[...,50]
        e2aD[YX,5,8] = -coreYX[...,51]
        e2aD[YX,6,8] = -coreYX[...,52]
        e2aD[YX,7,8] = -coreYX[...,53]
        e2aD[YX,8,8] = -coreYX[...,54]
        #print(e2aD)
        #print(e1bD)

        ########## YY ##########
        e2aD[YY,0,0] = -coreYY[...,0]
        e2aD[YY,0,1] = -coreYY[...,1]
        e2aD[YY,1,1] = -coreYY[...,2]
        e2aD[YY,0,2] = -coreYY[...,3]
        e2aD[YY,1,2] = -coreYY[...,4]
        e2aD[YY,2,2] = -coreYY[...,5]
        e2aD[YY,0,3] = -coreYY[...,6]
        e2aD[YY,1,3] = -coreYY[...,7]
        e2aD[YY,2,3] = -coreYY[...,8]
        e2aD[YY,3,3] = -coreYY[...,9]
        e2aD[YY,0,4] = -coreYY[...,10]
        e2aD[YY,1,4] = -coreYY[...,11]
        e2aD[YY,2,4] = -coreYY[...,12]
        e2aD[YY,3,4] = -coreYY[...,13]
        e2aD[YY,4,4] = -coreYY[...,14]
        e2aD[YY,0,5] = -coreYY[...,15]
        e2aD[YY,1,5] = -coreYY[...,16]
        e2aD[YY,2,5] = -coreYY[...,17]
        e2aD[YY,3,5] = -coreYY[...,18]
        e2aD[YY,4,5] = -coreYY[...,19]
        e2aD[YY,5,5] = -coreYY[...,20]
        e2aD[YY,0,6] = -coreYY[...,21]
        e2aD[YY,1,6] = -coreYY[...,22]
        e2aD[YY,2,6] = -coreYY[...,23]
        e2aD[YY,3,6] = -coreYY[...,24]
        e2aD[YY,4,6] = -coreYY[...,25]
        e2aD[YY,5,6] = -coreYY[...,26]
        e2aD[YY,6,6] = -coreYY[...,27]
        e2aD[YY,0,7] = -coreYY[...,28]
        e2aD[YY,1,7] = -coreYY[...,29]
        e2aD[YY,2,7] = -coreYY[...,30]
        e2aD[YY,3,7] = -coreYY[...,31]
        e2aD[YY,4,7] = -coreYY[...,32]
        e2aD[YY,5,7] = -coreYY[...,33]
        e2aD[YY,6,7] = -coreYY[...,34]
        e2aD[YY,7,7] = -coreYY[...,35]
        e2aD[YY,0,8] = -coreYY[...,36]
        e2aD[YY,1,8] = -coreYY[...,37]
        e2aD[YY,2,8] = -coreYY[...,38]
        e2aD[YY,3,8] = -coreYY[...,39]
        e2aD[YY,4,8] = -coreYY[...,40]
        e2aD[YY,5,8] = -coreYY[...,41]
        e2aD[YY,6,8] = -coreYY[...,42]
        e2aD[YY,7,8] = -coreYY[...,43]
        e2aD[YY,8,8] = -coreYY[...,44]

        e1bD[YY,0,0] = -coreYY[...,45]
        e1bD[YY,0,1] = -coreYY[...,46]
        e1bD[YY,1,1] = -coreYY[...,47]
        e1bD[YY,0,2] = -coreYY[...,48]
        e1bD[YY,1,2] = -coreYY[...,49]
        e1bD[YY,2,2] = -coreYY[...,50]
        e1bD[YY,0,3] = -coreYY[...,51]
        e1bD[YY,1,3] = -coreYY[...,52]
        e1bD[YY,2,3] = -coreYY[...,53]
        e1bD[YY,3,3] = -coreYY[...,54]
        e1bD[YY,0,4] = -coreYY[...,55]
        e1bD[YY,1,4] = -coreYY[...,56]
        e1bD[YY,2,4] = -coreYY[...,57]
        e1bD[YY,3,4] = -coreYY[...,58]
        e1bD[YY,4,4] = -coreYY[...,59]
        e1bD[YY,0,5] = -coreYY[...,60]
        e1bD[YY,1,5] = -coreYY[...,61]
        e1bD[YY,2,5] = -coreYY[...,62]
        e1bD[YY,3,5] = -coreYY[...,63]
        e1bD[YY,4,5] = -coreYY[...,64]
        e1bD[YY,5,5] = -coreYY[...,65]
        e1bD[YY,0,6] = -coreYY[...,66]
        e1bD[YY,1,6] = -coreYY[...,67]
        e1bD[YY,2,6] = -coreYY[...,68]
        e1bD[YY,3,6] = -coreYY[...,69]
        e1bD[YY,4,6] = -coreYY[...,70]
        e1bD[YY,5,6] = -coreYY[...,71]
        e1bD[YY,6,6] = -coreYY[...,72]
        e1bD[YY,0,7] = -coreYY[...,73]
        e1bD[YY,1,7] = -coreYY[...,74]
        e1bD[YY,2,7] = -coreYY[...,75]
        e1bD[YY,3,7] = -coreYY[...,76]
        e1bD[YY,4,7] = -coreYY[...,77]
        e1bD[YY,5,7] = -coreYY[...,78]
        e1bD[YY,6,7] = -coreYY[...,79]
        e1bD[YY,7,7] = -coreYY[...,80]
        e1bD[YY,0,8] = -coreYY[...,81]
        e1bD[YY,1,8] = -coreYY[...,82]
        e1bD[YY,2,8] = -coreYY[...,83]
        e1bD[YY,3,8] = -coreYY[...,84]
        e1bD[YY,4,8] = -coreYY[...,85]
        e1bD[YY,5,8] = -coreYY[...,86]
        e1bD[YY,6,8] = -coreYY[...,87]
        e1bD[YY,7,8] = -coreYY[...,88]
        e1bD[YY,8,8] = -coreYY[...,89]


        wc  = torch.zeros(rij.shape[0],45,45,dtype=dtype, device=device)

        wc[YH,0,:45] = riYH
        wc[YX,:10,:45] = riYX.reshape((-1,10,45))
        wc[YY] = riYY.reshape((-1,45,45))
        #wc[YY] = torch.transpose(riYY.reshape((-1,45,45)),1,2)
        #print('wc[YY]')
        #print(wc[YY])


        wc[HH,0,0] = wHH
        wc[XH,:10,0] = wXH
        wc[XX,:10,:10] = w.reshape((-1,10,10))

        KK = YH | YX | YY

        wcRotated = wc.clone()
        wcRotated[~KK,...,...] = torch.transpose(wc[~KK,...,...],1,2)
        wcRotated[KK,...,...] = Rotate2Center2Electron(wc[KK,...,...],dRotationMatrix[KK,...,...])

        return wcRotated, e1bD, e2aD
    
    wc  = torch.zeros(rij.shape[0],10,10,dtype=dtype, device=device)
    wc[HH,0,0] = wHH
    wc[XH,:,0] = wXH
    wc[XX] = w.reshape((-1,10,10))

    return wc, e1b, e2a


def GetSlaterCondonParameter(K,NA,EA,NB,EB,NC,EC,ND,ED):
    

#     CALCULATE THE RADIAL PART OF ONE-CENTER TWO-ELECTRON INTEGRALS
#     (SLATER-CONDON PARAMETER).
#     K     - TYPE OF INTEGRAL, CAN BE EQUAL TO 0,1,2,3,4 IN SPD-BASIS
#     NA,NB - PRINCIPLE QUANTUM NUMBER OF AO, ELECTRON 1
#     EA,EB - EXPONENTS OF AO, ELECTRON 1
#     NC,ND - PRINCIPLE QUANTUM NUMBER OF AO, ELECTRON 2
#     EC,ED - EXPONENTS OF AO, ELECTRON 2
      NA = int(NA)
      NB = int(NB)
      NC = int(NC)
      ND = int(ND)




      AEA    = math.log(EA)
      AEB    = math.log(EB)
      AEC    = math.log(EC)
      AED    = math.log(ED)
      NAB    = NA+NB
      NCD    = NC+ND
      ECD    = EC+ED
      EAB    = EA+EB
      E      = ECD+EAB
      N      = NAB+NCD
      AE     = math.log(E)
      A2     = math.log(2)
      ACD    = math.log(ECD)
      AAB    = math.log(EAB)
      C      = math.exp(math.log(math.factorial(N-1))+NA*AEA+NB*AEB+NC*AEC+ND*AED 
                   +0.5*(AEA+AEB+AEC+AED)+A2*(N+2) 
                   -0.5*(math.log(math.factorial(2*NA))+math.log(math.factorial(2*NB))   
                   +math.log(math.factorial(2*NC))+math.log(math.factorial(2*ND)))-AE*N)
      C      = C*ev
      S0     = 1/E
      S1     = 0
      S2     = 0
      M      = NCD-K
      I      = 1
        
      while(I <= M):
          S0     = S0*E/ECD
          S1     = S1+S0*(binom(NCD-K-1,I-1)-binom(NCD+K+1-1,I-1))/binom(N-1,I-1)
          I = I + 1
      M1     = M+1
      M2     = NCD+K+1
      I      = M1
      while(I <= M2):
          S0     = S0*E/ECD
          S2     = S2+S0*binom(M2-1,I-1)/binom(N-1,I-1)
          I = I + 1
      S3     = math.exp(AE*N-ACD*M2-AAB*(NAB-K))/binom(N-1,M2-1)
      slaterCondon = C*(S1-S2+S3)
      return slaterCondon

def GetSlaterCondonParameter_test(K,NA,EA,NB,EB,NC,EC,ND,ED):
    

#     CALCULATE THE RADIAL PART OF ONE-CENTER TWO-ELECTRON INTEGRALS
#     (SLATER-CONDON PARAMETER).
#     K     - TYPE OF INTEGRAL, CAN BE EQUAL TO 0,1,2,3,4 IN SPD-BASIS
#     NA,NB - PRINCIPLE QUANTUM NUMBER OF AO, ELECTRON 1
#     EA,EB - EXPONENTS OF AO, ELECTRON 1
#     NC,ND - PRINCIPLE QUANTUM NUMBER OF AO, ELECTRON 2
#     EC,ED - EXPONENTS OF AO, ELECTRON 2
      NA = int(NA)
      NB = int(NB)
      NC = int(NC)
      ND = int(ND)




      AEA    = math.log(EA)
      AEB    = math.log(EB)
      AEC    = math.log(EC)
      AED    = math.log(ED)
      NAB    = NA+NB
      NCD    = NC+ND
      ECD    = EC+ED
      EAB    = EA+EB
      E      = ECD+EAB
      N      = NAB+NCD
      AE     = math.log(E)
      A2     = math.log(2)
      ACD    = math.log(ECD)
      AAB    = math.log(EAB)
      C      = math.exp(math.log(math.factorial(N-1))+NA*AEA+NB*AEB+NC*AEC+ND*AED 
                   +0.5*(AEA+AEB+AEC+AED)+A2*(N+2) 
                   -0.5*(math.log(math.factorial(2*NA))+math.log(math.factorial(2*NB))   
                   +math.log(math.factorial(2*NC))+math.log(math.factorial(2*ND)))-AE*N)
      C      = C*ev
      S0     = 1/E
      S1     = 0
      S2     = 0
      M      = NCD-K
      I      = 1
      #print('NAAA\n',NA ,NB ,NC ,ND ,AEA,AEB,AEC,AED,NAB,NCD,ECD,EAB,E  ,N  ,AE ,A2 ,ACD,AAB,C ,S0 ,S1 ,S2 ,M)
        
      while(I <= M):
          S0     = S0*E/ECD
          S1     = S1+S0*(binom(NCD-K-1,I-1)-binom(NCD+K+1-1,I-1))/binom(N-1,I-1)
          I = I + 1
      M1     = M+1
      M2     = NCD+K+1
      I      = M1
      while(I <= M2):
          S0     = S0*E/ECD
          S2     = S2+S0*binom(M2-1,I-1)/binom(N-1,I-1)
          I = I + 1
      S3     = math.exp(AE*N-ACD*M2-AAB*(NAB-K))/binom(N-1,M2-1)
      slaterCondon = C*(S1-S2+S3)
      return slaterCondon

def binom(a, b):

      n = math.factorial(a)
      m = math.factorial(b)
      o = math.factorial(a-b)
      k = n/(m*(o))

      return k 

