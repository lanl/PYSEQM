import torch
from torch import sqrt
from .constants import ev
import sys
import numpy
import math
from .RotationMatrixD import RotateCore
import time
#this version try to split the pairs H-H, X-H, and X-X
# ~40% pairs are X-X, and each take 22 in ri
# ~40% are X-H, each take 4 in ri
# <20% are H-H, each only take 1 in ri
#spliting these will save memory, while make the sctructure complexated
# and the overlapy matrix code diat.py is implemented as treating all the pairs
# the same regarding the storage in memory

# as the rotate.f treat these pairs differently
# will generate ri for each type of pairs, and do the same thing on rotate.f
# then combine them together to be in the same shape


#chech repp.f
def two_elec_two_center_int_local_frame_d_orbitals(ni,nj,r0d, tore, da0d,db0d, qa0d,qb0d, dpa0d, dpb0d, dsa0d, dsb0d, dda0d, ddb0d, rho0ad,rho0bd, rho1ad,rho1bd, rho2ad,rho2bd, rho3ad,rho3bd, rho4ad,rho4bd, rho5ad,rho5bd, rho6ad,rho6bd, dd, themethod,rotationMatrix, ri, riXH,drho_corea,drho_coreb, riXHPM6, riPM6a, riPM6b  ):
    """
    two electron two center integrals in local frame for each pair
    """
##    t = time.time()
    dtype = r0d.dtype
    device = r0d.device
    # ni, nj, r0, da0, db0, qa0, qb0, rho0a, rho0b ... rho2b, shape (napirs,)
    # tore: dictionary type tensor tore[1]=1,
    #       valence shell charge for H, tore[6]=4, valence shell charge for C
    #rho0=0.5/am
    #rho1=0.5/ad
    #rho2=0.5/aq
    #ni, nj, atomic number of the first and second ones
    #r0 : INTERATOMIC distance rij, in atomic unit
    #ri: array of two-electron repulsion integrals, in unit of ev
    #core: 4 by 2 array of electron-core attraction integrals

    #C *** THIS ROUTINE COMPUTES THE TWO-CENTRE REPULSION INTEGRALS AND THE
    #C *** NUCLEAR ATTRACTION INTEGRALS.
    #C *** THE TWO-CENTRE REPULSION INTEGRALS (OVER LOCAL COORDINATES) ARE
    #C *** STORED AS FOLLOWS (WHERE P-SIGMA = O,  AND P-PI = P AND P* )
    #C     (SS/SS)=1,   (SO/SS)=2,   (OO/SS)=3,   (PP/SS)=4,   (SS/OS)=5,
    #C     (SO/SO)=6,   (SP/SP)=7,   (OO/SO)=8,   (PP/SO)=9,   (PO/SP)=10,
    #C     (SS/OO)=11,  (SS/PP)=12,  (SO/OO)=13,  (SO/PP)=14,  (SP/OP)=15,
    #C     (OO/OO)=16,  (PP/OO)=17,  (OO/PP)=18,  (PP/PP)=19,  (PO/PO)=20,
    #C     (PP/P*P*)=21,   (P*P/P*P)=22.
    #C *** THE STORAGE OF THE NUCLEAR ATTRACTION INTEGRALS  CORE(KL/IJ) IS
    #C     (SS/)=1,   (SO/)=2,   (OO/)=3,   (PP/)=4
    #C     WHERE IJ=1 IF THE ORBITALS CENTRED ON ATOM I,  =2 IF ON ATOM J.
    # da, db, dipole charge separation
    # qa, qb, qutrupole charge separation
    # tore valence shell charge: H -> +1, O -> +6
    # rho0, rho1, rho2: additive terms
    # ni, nj, r, shape: (nparis,)
    # ri : (npairs, 22)
    # core : (npairs, 4, 2)
    #  tore: make it like a dict
    # tore[1] =  1 for H
    # tore[8] = 6 for O
    # da, db, qa, qb, shape (npairs,)
    # rho0-2, shape (npairs,)
    ev1 = ev/2.0
    ev2 = ev/4.0
    ev3 = ev/8.0
    ev4 = ev/16.0

    HH = (ni==1) & (nj==1)
    XH = (ni>1) & (nj==1) 
    XX = (ni>1) & (nj>1) 
    YH = (((ni > 12) & (ni <18)) | ((ni > 20) & (ni <30)) | ((ni > 32) & (ni <36)) | ((ni > 38) & (ni <48)) | ((ni > 50) & (ni <54)) | ((ni > 70) & (ni <80)) | (ni ==57))  & (nj==1)
  
    YX = (ni>1) & (nj>1) & (((ni > 12) & (ni <18)) | ((ni > 20) & (ni <30)) | ((ni > 32) & (ni <36)) | ((ni > 38) & (ni <48)) | ((ni > 50) & (ni <54)) | ((ni > 70) & (ni <80)) | (ni ==57)) & \
         ((nj <= 12) | ((nj >= 18) & (nj <=20)) | ((nj >= 30) & (nj <= 32)) | ((nj >= 36) & (nj <= 38)) | ((nj >= 48) & (nj <= 50)) | ((nj >= 54) & (nj <= 56)) | ((nj >= 80) & (nj <= 83)))

    YY = (((ni > 12) & (ni <18)) | ((ni > 20) & (ni <30)) | ((ni > 32) & (ni <36)) | ((ni > 38) & (ni <48)) | ((ni > 50) & (ni <54)) | ((ni > 70) & (ni <80)) | (ni ==57)) &\
         (((nj > 12) & (nj <18)) | ((nj > 20) & (nj <30)) | ((nj > 32) & (nj <36)) | ((nj > 38) & (nj <48)) | ((nj > 50) & (nj <54)) | ((nj > 70) & (nj <80)) | (nj ==57))


    Ycore = (drho_corea[YY] > 0.000) | (drho_coreb[YY] > 0.000)
    if(YH.sum() > 0 ):

        riYH =  torch.zeros(YH.sum(),45,dtype=dtype, device=device)
    
        r0 = r0d[YH]
        da0 = da0d[YH]
        qa0 = qa0d[YH]
        dpa0 = dpa0d[YH]
        dsa0 = dsa0d[YH]
        dda0 = dda0d[YH]
        rho0a = rho0ad[YH]
        rho0b = rho0bd[YH]
        rho1a = rho1ad[YH]
        rho2a = rho2ad[YH]
        rho3a = rho3ad[YH]
        rho4a = rho4ad[YH]
        rho5a = rho5ad[YH]
        rho6a = rho6ad[YH]
    
    
        rhoSS = (rho0a+rho0b)**2
        rhoSDD0 = (rho3a+rho0b)**2 
        rhoSSP = (rho1a+rho0b)**2
        rhoSPD = (rho4a+rho0b)**2
        rhoSPP = (rho2a+rho0b)**2
        ##rhoSSD = (rho3a+rho6b)**2
        rhoSSD = (rho5a+rho0b)**2
        rhoSDD = (rho6a+rho0b)**2

    
        ########    00|00    ########
        qq = ev/torch.sqrt(r0**2+rhoSS)
        #############################
        DDq_Sq = ev/torch.sqrt(r0**2+rhoSDD0)

        ########    00|10    ########
    
        #############################
        DPUz_Sq = (ev1/torch.sqrt((r0+dpa0)**2+rhoSPD)-ev1/torch.sqrt((r0-dpa0)**2+rhoSPD))
        SPUz_Sq = (ev1/torch.sqrt((r0+da0)**2+rhoSSP)-ev1/torch.sqrt((r0-da0)**2+rhoSSP))
    


        #############################
    
        DDQtzx_onehalfQtxy_Sq = ev2/torch.sqrt((r0-dda0)**2+rhoSDD)+ev2/torch.sqrt((r0+dda0)**2+rhoSDD)-ev1/torch.sqrt((r0)**2+(dda0)**2+rhoSDD)
        DSQtzx_onehalfQtxy_Sq = ev2/torch.sqrt((r0-dsa0)**2+rhoSSD)+ev2/torch.sqrt((r0+dsa0)**2+rhoSSD)-ev1/torch.sqrt((r0)**2+(dsa0)**2+rhoSSD)
        PPQtzx_onehalfQtxy_Sq = ev2/torch.sqrt((r0-qa0)**2+rhoSPP)+ev2/torch.sqrt((r0+qa0)**2+rhoSPP)-ev1/torch.sqrt((r0)**2+(da0)**2+rhoSPP)




        ############################## D|S ##############################
        
        #####DxyDxy|SS#####
        riYH[...,44]=DDq_Sq*1.000000*1.000000+DDQtzx_onehalfQtxy_Sq*-1.333333*1.000000

        #####PxDxz|SS#####
        riYH[...,17]=DPUz_Sq*1.000000*1.000000

        #####DxzDxz|SS#####
        riYH[...,20]=DDq_Sq*1.000000*1.000000+DDQtzx_onehalfQtxy_Sq*0.666667*1.000000

        #####SDz2|SS#####
        riYH[...,10]=DSQtzx_onehalfQtxy_Sq*1.154701*1.000000

        #####PzDz2|SS#####
        riYH[...,11]=DPUz_Sq*1.154701*1.000000

        #####Dz2Dz2|SS#####
        riYH[...,14]=DDq_Sq*1.000000*1.000000+DDQtzx_onehalfQtxy_Sq*1.333333*1.000000

        #####PyDyz|SS#####
        riYH[...,24]=DPUz_Sq*1.000000*1.000000

        #####DyzDyz|SS#####
        riYH[...,27]=DDq_Sq*1.000000*1.000000+DDQtzx_onehalfQtxy_Sq*0.666667*1.000000

        #####Dx2-y2Dx2-y2|SS#####
        riYH[...,35]=DDq_Sq*1.000000*1.000000+DDQtzx_onehalfQtxy_Sq*-1.333333*1.000000

        coreYH = torch.zeros(YH.sum(),46,dtype=dtype, device=device)
        coreYHLocal = torch.zeros(YH.sum(),46,dtype=dtype, device=device)

        YYH = (((ni[XH] > 12) & (ni[XH] <18)) | ((ni[XH] > 20) & (ni[XH] <30)) | ((ni[XH] > 32) & (ni[XH] <36)) | ((ni[XH] > 38) & (ni[XH] <48)) | ((ni[XH] > 50) & (ni[XH] <54)) | ((ni[XH] > 70) & (ni[XH] <80)) | (ni[XH] ==57))  & (nj[XH]==1)


        coreYHLocal[...,0] = tore[ni[YH]]*riXHPM6[YYH,1-1]

        coreYHLocal[...,1] = tore[nj[YH]]*riXH[YYH,1-1]  
        coreYHLocal[...,3] = tore[nj[YH]]*riXH[YYH,4-1]
        coreYHLocal[...,7] = tore[nj[YH]]*riXH[YYH,2-1]
        coreYHLocal[...,10] = tore[nj[YH]]*riXH[YYH,3-1]
        coreYHLocal[...,15] = tore[nj[YH]]*riYH[...,44]
        coreYHLocal[...,17] = tore[nj[YH]]*riYH[...,17]
        coreYHLocal[...,21] = tore[nj[YH]]*riYH[...,20]
        coreYHLocal[...,22] = tore[nj[YH]]*riYH[...,10]
        coreYHLocal[...,25] = tore[nj[YH]]*riYH[...,11]
        coreYHLocal[...,28] = tore[nj[YH]]*riYH[...,14]
        coreYH[...,0] = coreYHLocal[...,0]
        coreYH[...,1:] = RotateCore(coreYHLocal[...,1:],rotationMatrix[YH,...,...],3)
    else:
        coreYH = torch.zeros(YH.sum(),46,dtype=dtype, device=device)
        riYH =  torch.zeros(YH.sum(),45,dtype=dtype, device=device)
    if(YX.sum() > 0 ):

        riYX =  torch.zeros(YX.sum(),450,dtype=dtype, device=device)
    
        r0 = r0d[YX]
        da0 = da0d[YX]
        db0 = db0d[YX]
        qa0 = qa0d[YX]
        qb0 = qb0d[YX]
        dpa0 = dpa0d[YX]
        dsa0 = dsa0d[YX]
        dda0 = dda0d[YX]
        rho0a = rho0ad[YX]
        rho0b = rho0bd[YX]
        rho1a = rho1ad[YX]
        rho1b = rho1bd[YX]
        rho2a = rho2ad[YX]
        rho2b = rho2bd[YX]
        rho3a = rho3ad[YX]
        rho4a = rho4ad[YX]
        rho5a = rho5ad[YX]
        rho6a = rho6ad[YX]
    
        rhoSS = (rho0a+rho0b)**2
        rhoSDD0 = (rho3a+rho0b)**2 
        rhoSSP = (rho1a+rho0b)**2
        rhoSPD = (rho4a+rho0b)**2
        rhoSPP = (rho2a+rho0b)**2
        rhoSSD = (rho5a+rho0b)**2
        rhoSDD = (rho6a+rho0b)**2

        rhoDD0SP = (rho3a+rho1b)**2

        rhoDD0PP = (rho3a+rho2b)**2

        rhoSPSP = (rho1a+rho1b)**2
        rhoSPPD = (rho4a+rho1b)**2

        rhoSPPP = (rho2a+rho1b)**2
        rhoSPSD = (rho5a+rho1b)**2
        rhoSPDD = (rho6a+rho1b)**2
        rhoPDPP = (rho4a+rho2b)**2

        rhoPPPP = (rho2a+rho2b)**2
        rhoPPDD = (rho6a+rho2b)**2
        rhoPPSD = (rho5a+rho2b)**2


        ########    00|00    ########
        qq = ev/torch.sqrt(r0**2+rhoSS)
        #############################
        DDq_Sq = ev/torch.sqrt(r0**2+rhoSDD0)



        ########    00|10    ########
        Sq_SPUz = -(ev1/torch.sqrt((r0+db0)**2+rhoSSP)-ev1/torch.sqrt((r0-db0)**2+rhoSSP))
    
        DDq_SPUz = -(ev1/torch.sqrt((r0+db0)**2+rhoDD0SP)-ev1/torch.sqrt((r0-db0)**2+rhoDD0SP))
        ##UPPER
        #############################
        DPUz_Sq = (ev1/torch.sqrt((r0+dpa0)**2+rhoSPD)-ev1/torch.sqrt((r0-dpa0)**2+rhoSPD))
        SPUz_Sq = (ev1/torch.sqrt((r0+da0)**2+rhoSSP)-ev1/torch.sqrt((r0-da0)**2+rhoSSP))



        ########    00|20    ########
        qb0 = qb0*math.sqrt(2)


        Sq_PPQtzx_onehalfQtxy = ev2/torch.sqrt((r0-(2)*qb0)**2+rhoSPP)+ev2/torch.sqrt((r0+(2)*qb0)**2+rhoSPP)-ev1/torch.sqrt((r0)**2+((0)*db0)**2+rhoSPP)

        DDq_PPQtzx_onehalfQtxy = ev2/torch.sqrt((r0-qb0)**2+rhoDD0PP)+ev2/torch.sqrt((r0+qb0)**2+rhoDD0PP)-ev1/torch.sqrt((r0)**2+(qb0)**2+rhoDD0PP)
        #############################
        DDQtzx_onehalfQtxy_Sq = ev2/torch.sqrt((r0-dda0)**2+rhoSDD)+ev2/torch.sqrt((r0+dda0)**2+rhoSDD)-ev1/torch.sqrt((r0)**2+(dda0)**2+rhoSDD)
        DSQtzx_onehalfQtxy_Sq = ev2/torch.sqrt((r0-dsa0)**2+rhoSSD)+ev2/torch.sqrt((r0+dsa0)**2+rhoSSD)-ev1/torch.sqrt((r0)**2+(dsa0)**2+rhoSSD)



        PPQtzx_onehalfQtxy_Sq = ev2/torch.sqrt((r0-qa0)**2+rhoSPP)+ev2/torch.sqrt((r0+qa0)**2+rhoSPP)-ev1/torch.sqrt((r0)**2+(da0)**2+rhoSPP)
 
        qb0 = qb0/math.sqrt(2)


        ########    11|11    ########
        SPUpi_SPUpi = ev1/torch.sqrt(r0**2+(da0-qb0)**2+rhoSPSP)-ev1/torch.sqrt(r0**2+(da0+qb0)**2+rhoSPSP)
        #############################
        DPUpi_SPUpi = ev1/torch.sqrt(r0**2+(db0-dpa0)**2+rhoSPPD)-ev1/torch.sqrt(r0**2+(db0+dpa0)**2+rhoSPPD)
    
        ########    10|10    ########
        SPUz_SPUz =  ev2/torch.sqrt((r0+da0-db0)**2+rhoSPSP) - \
                     ev2/torch.sqrt((r0+da0+db0)**2+rhoSPSP) - \
                     ev2/torch.sqrt((r0-da0-db0)**2+rhoSPSP) + \
                     ev2/torch.sqrt((r0-da0+db0)**2+rhoSPSP)
        #############################
        DPUz_SPUz =  ev2/torch.sqrt((r0+db0-dpa0)**2+rhoSPPD) - \
                     ev2/torch.sqrt((r0+db0+dpa0)**2+rhoSPPD) - \
                     ev2/torch.sqrt((r0-db0-dpa0)**2+rhoSPPD) + \
                     ev2/torch.sqrt((r0-db0+dpa0)**2+rhoSPPD)
    
    
        ########    11|21    ########
        dsa0 = dsa0/math.sqrt(2)
        dda0 = dda0/math.sqrt(2)

        SPUpi_PPQpiz = -(-ev2/torch.sqrt((r0-qb0)**2+(da0-qb0)**2+rhoSPPP) + \
                       ev2/torch.sqrt((r0-qb0)**2+(da0+qb0)**2+rhoSPPP) + \
                       ev2/torch.sqrt((r0+qb0)**2+(da0-qb0)**2+rhoSPPP) - \
                       ev2/torch.sqrt((r0+qb0)**2+(da0+qb0)**2+rhoSPPP))
        DPUpi_PPQpiz = -(-ev2/torch.sqrt((r0-qb0)**2+(dpa0-qb0)**2+rhoPDPP) + \
                       ev2/torch.sqrt((r0-qb0)**2+(dpa0+qb0)**2+rhoPDPP) + \
                       ev2/torch.sqrt((r0+qb0)**2+(dpa0-qb0)**2+rhoPDPP) - \
                       ev2/torch.sqrt((r0+qb0)**2+(dpa0+qb0)**2+rhoPDPP))
        ### UPPER
        #############################
        PPQpiz_SPUpi = (-ev2/torch.sqrt((r0-qa0)**2+(db0-qa0)**2+rhoSPPP) + \
                       ev2/torch.sqrt((r0-qa0)**2+(db0+qa0)**2+rhoSPPP) + \
                       ev2/torch.sqrt((r0+qa0)**2+(db0-qa0)**2+rhoSPPP) - \
                       ev2/torch.sqrt((r0+qa0)**2+(db0+qa0)**2+rhoSPPP))
        DSQpiz_SPUpi = (-ev2/torch.sqrt((r0-dsa0)**2+(db0-dsa0)**2+rhoSPSD) + \
                       ev2/torch.sqrt((r0-dsa0)**2+(db0+dsa0)**2+rhoSPSD) + \
                       ev2/torch.sqrt((r0+dsa0)**2+(db0-dsa0)**2+rhoSPSD) - \
                       ev2/torch.sqrt((r0+dsa0)**2+(db0+dsa0)**2+rhoSPSD))
        DDQpiz_SPUpi = (-ev2/torch.sqrt((r0-dda0)**2+(db0-dda0)**2+rhoSPDD) + \
                       ev2/torch.sqrt((r0-dda0)**2+(db0+dda0)**2+rhoSPDD) + \
                       ev2/torch.sqrt((r0+dda0)**2+(db0-dda0)**2+rhoSPDD) - \
                       ev2/torch.sqrt((r0+dda0)**2+(db0+dda0)**2+rhoSPDD))
        dsa0 = dsa0*math.sqrt(2)
        dda0 = dda0*math.sqrt(2)
        #db0 = db0*math.sqrt(2)
 

        ########    10|20    ########
        qa0 = qa0*math.sqrt(2)
        qb0 = qb0*math.sqrt(2)



        SPUz_PPQtzx_onehalfQtxy = (-ev3/torch.sqrt((r0-da0-math.sqrt(1)*qb0)**2+rhoSPPP) - \
                                   ev3/torch.sqrt((r0-da0+math.sqrt(1)*qb0)**2+rhoSPPP) + \
                                   ev3/torch.sqrt((r0+da0-math.sqrt(1)*qb0)**2+rhoSPPP) + \
                                   ev3/torch.sqrt((r0+da0+math.sqrt(1)*qb0)**2+rhoSPPP) + \
                                   ev2/torch.sqrt((r0-da0)**2+(math.sqrt(1)*qb0)**2+rhoSPPP) - \
                                   ev2/torch.sqrt((r0+da0)**2+(math.sqrt(1)*qb0)**2+rhoSPPP))
    
        DPUz_PPQtzx_onehalfQtxy = (-ev3/torch.sqrt((r0-dpa0-math.sqrt(1)*qb0)**2+rhoPDPP) - \
                                   ev3/torch.sqrt((r0-dpa0+math.sqrt(1)*qb0)**2+rhoPDPP) + \
                                   ev3/torch.sqrt((r0+dpa0-math.sqrt(1)*qb0)**2+rhoPDPP) + \
                                   ev3/torch.sqrt((r0+dpa0+math.sqrt(1)*qb0)**2+rhoPDPP) + \
                                   ev2/torch.sqrt((r0-dpa0)**2+(math.sqrt(1)*qb0)**2+rhoPDPP) - \
                                   ev2/torch.sqrt((r0+dpa0)**2+(math.sqrt(1)*qb0)**2+rhoPDPP))
        ##UPPER
        #############################
        PPQtzx_onehalfQtxy_SPUz = -(-ev3/torch.sqrt((r0-db0-math.sqrt(1)*qa0)**2+rhoSPPP) - \
                                   ev3/torch.sqrt((r0-db0+math.sqrt(1)*qa0)**2+rhoSPPP) + \
                                   ev3/torch.sqrt((r0+db0-math.sqrt(1)*qa0)**2+rhoSPPP) + \
                                   ev3/torch.sqrt((r0+db0+math.sqrt(1)*qa0)**2+rhoSPPP) + \
                                   ev2/torch.sqrt((r0-db0)**2+(math.sqrt(1)*qa0)**2+rhoSPPP) - \
                                   ev2/torch.sqrt((r0+db0)**2+(math.sqrt(1)*qa0)**2+rhoSPPP))
        DSQtzx_onehalfQtxy_SPUz = -(-ev3/torch.sqrt((r0-db0-math.sqrt(1)*dsa0)**2+rhoSPSD) - \
                                   ev3/torch.sqrt((r0-db0+math.sqrt(1)*dsa0)**2+rhoSPSD) + \
                                   ev3/torch.sqrt((r0+db0-math.sqrt(1)*dsa0)**2+rhoSPSD) + \
                                   ev3/torch.sqrt((r0+db0+math.sqrt(1)*dsa0)**2+rhoSPSD) + \
                                   ev2/torch.sqrt((r0-db0)**2+(math.sqrt(1)*dsa0)**2+rhoSPSD) - \
                                   ev2/torch.sqrt((r0+db0)**2+(math.sqrt(1)*dsa0)**2+rhoSPSD))
        DDQtzx_onehalfQtxy_SPUz = -(-ev3/torch.sqrt((r0-db0-math.sqrt(1)*dda0)**2+rhoSPDD) - \
                                   ev3/torch.sqrt((r0-db0+math.sqrt(1)*dda0)**2+rhoSPDD) + \
                                   ev3/torch.sqrt((r0+db0-math.sqrt(1)*dda0)**2+rhoSPDD) + \
                                   ev3/torch.sqrt((r0+db0+math.sqrt(1)*dda0)**2+rhoSPDD) + \
                                   ev2/torch.sqrt((r0-db0)**2+(math.sqrt(1)*dda0)**2+rhoSPDD) - \
                                   ev2/torch.sqrt((r0+db0)**2+(math.sqrt(1)*dda0)**2+rhoSPDD))
        qa0 = qa0/math.sqrt(2)
        qb0 = qb0/math.sqrt(2)

    
        ########    20|20    ########
        #qa0 = qa0*math.sqrt(2)
        qb0 = qb0*math.sqrt(2)
        #dda0 = dda0*math.sqrt(2)
        #dsa0 = dsa0*math.sqrt(2)

        PPQtzx_onehalfQtxy_PPQtzx_onehalfQtxy =(ev3/torch.sqrt(r0**2+(qa0-qb0)**2+rhoPPPP) + \
                                                ev3/torch.sqrt(r0**2+(qa0+qb0)**2+rhoPPPP) - \
                                                ev3/torch.sqrt((r0+qb0)**2+qa0**2+rhoPPPP) - \
                                                ev3/torch.sqrt((r0-qb0)**2+qa0**2+rhoPPPP) + \
                                                ev4/torch.sqrt((r0-qb0+qa0)**2+rhoPPPP) + \
                                                ev4/torch.sqrt((r0+qb0-qa0)**2+rhoPPPP) + \
                                                ev4/torch.sqrt((r0+qb0+qa0)**2+rhoPPPP) + \
                                                ev4/torch.sqrt((r0-qb0-qa0)**2+rhoPPPP) - \
                                                ev3/torch.sqrt((r0-qa0)**2+qb0**2+rhoPPPP) - \
                                                ev3/torch.sqrt((r0+qa0)**2+qb0**2+rhoPPPP))- \
                                              0.25*(ev2/torch.sqrt(r0**2+(qa0-qb0)**2+rhoPPPP) +\
                                                 ev2/torch.sqrt(r0**2+(qa0+qb0)**2+rhoPPPP) -\
                                                 ev1/torch.sqrt(r0**2+qa0**2+qb0**2+rhoPPPP))




        #############################
        
        DSQtzx_onehalfQtxy_PPQtzx_onehalfQtxy = (ev3/torch.sqrt(r0**2+(qb0-dsa0)**2+rhoPPSD) + \
                                                ev3/torch.sqrt(r0**2+(qb0+dsa0)**2+rhoPPSD) - \
                                                ev3/torch.sqrt((r0+dsa0)**2+qb0**2+rhoPPSD) - \
                                                ev3/torch.sqrt((r0-dsa0)**2+qb0**2+rhoPPSD) + \
                                                ev4/torch.sqrt((r0-dsa0+qb0)**2+rhoPPSD) + \
                                                ev4/torch.sqrt((r0+dsa0-qb0)**2+rhoPPSD) + \
                                                ev4/torch.sqrt((r0+dsa0+qb0)**2+rhoPPSD) + \
                                                ev4/torch.sqrt((r0-dsa0-qb0)**2+rhoPPSD) - \
                                                ev3/torch.sqrt((r0-qb0)**2+dsa0**2+rhoPPSD) - \
                                                ev3/torch.sqrt((r0+qb0)**2+dsa0**2+rhoPPSD)) -  \
                                              0.25*(ev2/torch.sqrt(r0**2+(qb0-dsa0)**2+rhoPPSD) +\
                                                 ev2/torch.sqrt(r0**2+(qb0+dsa0)**2+rhoPPSD) -\
                                                 ev1/torch.sqrt(r0**2+qb0**2+dsa0**2+rhoPPSD))

 
        DDQtzx_onehalfQtxy_PPQtzx_onehalfQtxy = (ev3/torch.sqrt(r0**2+(qb0-dda0)**2+rhoPPDD) + \
                                                ev3/torch.sqrt(r0**2+(qb0+dda0)**2+rhoPPDD) - \
                                                ev3/torch.sqrt((r0+dda0)**2+qb0**2+rhoPPDD) - \
                                                ev3/torch.sqrt((r0-dda0)**2+qb0**2+rhoPPDD) + \
                                                ev4/torch.sqrt((r0-dda0+qb0)**2+rhoPPDD) + \
                                                ev4/torch.sqrt((r0+dda0-qb0)**2+rhoPPDD) + \
                                                ev4/torch.sqrt((r0+dda0+qb0)**2+rhoPPDD) + \
                                                ev4/torch.sqrt((r0-dda0-qb0)**2+rhoPPDD) - \
                                                ev3/torch.sqrt((r0-qb0)**2+dda0**2+rhoPPDD) - \
                                                ev3/torch.sqrt((r0+qb0)**2+dda0**2+rhoPPDD))- \
                                              0.25*(ev2/torch.sqrt(r0**2+(qb0-dda0)**2+rhoPPDD) +\
                                                 ev2/torch.sqrt(r0**2+(qb0+dda0)**2+rhoPPDD) -\
                                                 ev1/torch.sqrt(r0**2+qb0**2+dda0**2+rhoPPDD))
 
        qb0 = qb0/math.sqrt(2)

    
        ########    21|21    ########
        dsa0 = dsa0/math.sqrt(2)
        dda0 = dda0/math.sqrt(2) 

        PPQpiz_PPQpiz =  ev3/torch.sqrt((r0+qa0-qb0)**2+(qa0-qb0)**2+rhoPPPP) - \
                         ev3/torch.sqrt((r0+qa0-qb0)**2+(qa0+qb0)**2+rhoPPPP) - \
                         ev3/torch.sqrt((r0+qa0+qb0)**2+(qa0-qb0)**2+rhoPPPP) + \
                         ev3/torch.sqrt((r0+qa0+qb0)**2+(qa0+qb0)**2+rhoPPPP) - \
                         ev3/torch.sqrt((r0-qa0-qb0)**2+(qa0-qb0)**2+rhoPPPP) + \
                         ev3/torch.sqrt((r0-qa0-qb0)**2+(qa0+qb0)**2+rhoPPPP) + \
                         ev3/torch.sqrt((r0-qa0+qb0)**2+(qa0-qb0)**2+rhoPPPP) - \
                         ev3/torch.sqrt((r0-qa0+qb0)**2+(qa0+qb0)**2+rhoPPPP)
    
        #############################
    
        DSQpiz_PPQpiz =  ev3/torch.sqrt((r0+qb0-dsa0)**2+(qb0-dsa0)**2+rhoPPSD) - \
                         ev3/torch.sqrt((r0+qb0-dsa0)**2+(qb0+dsa0)**2+rhoPPSD) - \
                         ev3/torch.sqrt((r0+qb0+dsa0)**2+(qb0-dsa0)**2+rhoPPSD) + \
                         ev3/torch.sqrt((r0+qb0+dsa0)**2+(qb0+dsa0)**2+rhoPPSD) - \
                         ev3/torch.sqrt((r0-qb0-dsa0)**2+(qb0-dsa0)**2+rhoPPSD) + \
                         ev3/torch.sqrt((r0-qb0-dsa0)**2+(qb0+dsa0)**2+rhoPPSD) + \
                         ev3/torch.sqrt((r0-qb0+dsa0)**2+(qb0-dsa0)**2+rhoPPSD) - \
                         ev3/torch.sqrt((r0-qb0+dsa0)**2+(qb0+dsa0)**2+rhoPPSD)
    
        DDQpiz_PPQpiz =  ev3/torch.sqrt((r0+qb0-dda0)**2+(qb0-dda0)**2+rhoPPDD) - \
                         ev3/torch.sqrt((r0+qb0-dda0)**2+(qb0+dda0)**2+rhoPPDD) - \
                         ev3/torch.sqrt((r0+qb0+dda0)**2+(qb0-dda0)**2+rhoPPDD) + \
                         ev3/torch.sqrt((r0+qb0+dda0)**2+(qb0+dda0)**2+rhoPPDD) - \
                         ev3/torch.sqrt((r0-qb0-dda0)**2+(qb0-dda0)**2+rhoPPDD) + \
                         ev3/torch.sqrt((r0-qb0-dda0)**2+(qb0+dda0)**2+rhoPPDD) + \
                         ev3/torch.sqrt((r0-qb0+dda0)**2+(qb0-dda0)**2+rhoPPDD) - \
                         ev3/torch.sqrt((r0-qb0+dda0)**2+(qb0+dda0)**2+rhoPPDD)
        dsa0 = dsa0*math.sqrt(2)
        dda0 = dda0*math.sqrt(2)    
    
        ########    22|22    ########
        qb0 = qb0*math.sqrt(2)

        PPQxy_PPQxy = ev2/torch.sqrt((r0)**2+(qa0-qb0)**2+rhoPPPP) + \
                      ev2/torch.sqrt((r0)**2+(qa0+qb0)**2+rhoPPPP) - \
                      ev1/torch.sqrt((r0)**2+(qa0)**2+(qb0)**2+rhoPPPP)
    
        #############################
        DSQxy_PPQxy = ev2/torch.sqrt((r0)**2+(qb0-dsa0)**2+rhoPPSD) + \
                      ev2/torch.sqrt((r0)**2+(qb0+dsa0)**2+rhoPPSD) - \
                      ev1/torch.sqrt((r0)**2+(qb0)**2+(dsa0)**2+rhoPPSD)
    
        DDQxy_PPQxy = ev2/torch.sqrt((r0)**2+(qb0-dda0)**2+rhoPPDD) + \
                      ev2/torch.sqrt((r0)**2+(qb0+dda0)**2+rhoPPDD) - \
                      ev1/torch.sqrt((r0)**2+(qb0)**2+(dda0)**2+rhoPPDD)

        qb0 = qb0/math.sqrt(2)


        ########    22|2-2    ########
        qb0 = qb0*math.sqrt(2)
        tildaPPQxy_PPQxy = ev2/torch.sqrt((r0)**2+(qa0-qb0)**2+rhoPPPP) + \
                      ev2/torch.sqrt((r0)**2+(qa0+qb0)**2+rhoPPPP) - \
                      ev1/torch.sqrt((r0)**2+(qa0)**2+(qb0)**2+rhoPPPP)

        #############################
        tildaDSQxy_PPQxy = ev2/torch.sqrt((r0)**2+(qb0-dsa0)**2+rhoPPSD) + \
                      ev2/torch.sqrt((r0)**2+(qb0+dsa0)**2+rhoPPSD) - \
                      ev1/torch.sqrt((r0)**2+(qb0)**2+(dsa0)**2+rhoPPSD)

        tildaDDQxy_PPQxy = ev2/torch.sqrt((r0)**2+(qb0-dda0)**2+rhoPPDD) + \
                      ev2/torch.sqrt((r0)**2+(qb0+dda0)**2+rhoPPDD) - \
                      ev1/torch.sqrt((r0)**2+(qb0)**2+(dda0)**2+rhoPPDD)


        qb0 = qb0/math.sqrt(2)



        #####SDxy|PxPy#####
        riYX[...,396]=tildaDSQxy_PPQxy*1.000000*1.000000
        
        #####PxDxy|SPy#####
        riYX[...,308]=DPUpi_SPUpi*1.000000*1.000000
        
        #####PxDxy|PzPy#####
        riYX[...,353]=DPUpi_PPQpiz*1.000000*1.000000
        
        #####PyDxy|SPx#####
        riYX[...,174]=DPUpi_SPUpi*1.000000*1.000000
        
        #####PyDxy|PzPx#####
        riYX[...,219]=DPUpi_PPQpiz*1.000000*1.000000
        
        #####DxyDxy|SS#####
        riYX[...,44]=DDq_Sq*1.000000*1.000000+DDQtzx_onehalfQtxy_Sq*-1.333333*1.000000


        #####DxyDxy|PxPx#####
        riYX[...,269]=DDq_Sq*1.000000*1.000000+DDq_PPQtzx_onehalfQtxy*1.000000*-0.666667+DDQtzx_onehalfQtxy_Sq*-1.333333*1.000000+DDQtzx_onehalfQtxy_PPQtzx_onehalfQtxy*-1.333333*-0.666667
        
        #####DxyDxy|PyPy#####
        riYX[...,449]=DDq_Sq*1.000000*1.000000+DDq_PPQtzx_onehalfQtxy*1.000000*-0.666667+DDQtzx_onehalfQtxy_Sq*-1.333333*1.000000+DDQtzx_onehalfQtxy_PPQtzx_onehalfQtxy*-1.333333*-0.666667
        
        #####DxyDxy|SPz#####
        riYX[...,89]=DDq_SPUz*1.000000*1.000000+DDQtzx_onehalfQtxy_SPUz*-1.333333*1.000000
        
        #####DxyDxy|PzPz#####
        riYX[...,134]=DDq_Sq*1.000000*1.000000+DDq_PPQtzx_onehalfQtxy*1.000000*1.333333+DDQtzx_onehalfQtxy_Sq*-1.333333*1.000000+DDQtzx_onehalfQtxy_PPQtzx_onehalfQtxy*-1.333333*1.333333
        
        #####SDxz|SPx#####
        riYX[...,150]=DSQpiz_SPUpi*1.000000*1.000000
        
        #####SDxz|PzPx#####
        riYX[...,195]=DSQpiz_PPQpiz*1.000000*1.000000
        
        #####PxDxz|SS#####
        riYX[...,17]=DPUz_Sq*1.000000*1.000000
        
        #####PxDxz|PxPx#####
        riYX[...,242]=DPUz_Sq*1.000000*1.000000+DPUz_PPQtzx_onehalfQtxy*1.000000*-0.666667
        
        #####PxDxz|PyPy#####
        riYX[...,422]=DPUz_Sq*1.000000*1.000000+DPUz_PPQtzx_onehalfQtxy*1.000000*-0.666667
        
        #####PxDxz|SPz#####
        riYX[...,62]=DPUz_SPUz*1.000000*1.000000
        
        #####PxDxz|PzPz#####
        riYX[...,107]=DPUz_Sq*1.000000*1.000000+DPUz_PPQtzx_onehalfQtxy*1.000000*1.333333
        
        #####PzDxz|SPx#####
        riYX[...,151]=DPUpi_SPUpi*1.000000*1.000000
        
        #####PzDxz|PzPx#####
        riYX[...,196]=DPUpi_PPQpiz*1.000000*1.000000
        
        #####DxzDxy|SPy#####
        riYX[...,311]=DDQpiz_SPUpi*1.000000*1.000000
        
        #####DxzDxy|PzPy#####
        riYX[...,356]=DDQpiz_PPQpiz*1.000000*1.000000
        
        #####DxzDxz|SS#####
        riYX[...,20]=DDq_Sq*1.000000*1.000000+DDQtzx_onehalfQtxy_Sq*0.666667*1.000000
        
        #####DxzDxz|PxPx#####
        riYX[...,245]=DDq_Sq*1.000000*1.000000+DDq_PPQtzx_onehalfQtxy*1.000000*-0.666667+DDQtzx_onehalfQtxy_Sq*0.666667*1.000000+DDQtzx_onehalfQtxy_PPQtzx_onehalfQtxy*0.666667*-0.666667+DDQxy_PPQxy*1.000000*1.000000
        
        #####DxzDxz|PyPy#####
        riYX[...,425]=DDq_Sq*1.000000*1.000000+DDq_PPQtzx_onehalfQtxy*1.000000*-0.666667+DDQtzx_onehalfQtxy_Sq*0.666667*1.000000+DDQtzx_onehalfQtxy_PPQtzx_onehalfQtxy*0.666667*-0.666667-DDQxy_PPQxy*1.000000*1.000000
        
        #####DxzDxz|SPz#####
        riYX[...,65]=DDq_SPUz*1.000000*1.000000+DDQtzx_onehalfQtxy_SPUz*0.666667*1.000000
        
        #####DxzDxz|PzPz#####
        riYX[...,110]=DDq_Sq*1.000000*1.000000+DDq_PPQtzx_onehalfQtxy*1.000000*1.333333+DDQtzx_onehalfQtxy_Sq*0.666667*1.000000+DDQtzx_onehalfQtxy_PPQtzx_onehalfQtxy*0.666667*1.333333
        
        #####SDz2|SS#####
        riYX[...,10]=DSQtzx_onehalfQtxy_Sq*1.154701*1.000000
        
        #####SDz2|PxPx#####
        riYX[...,235]=DSQtzx_onehalfQtxy_Sq*1.154701*1.000000+DSQtzx_onehalfQtxy_PPQtzx_onehalfQtxy*1.154701*-0.666667
        
        #####SDz2|PyPy#####
        riYX[...,415]=DSQtzx_onehalfQtxy_Sq*1.154701*1.000000+DSQtzx_onehalfQtxy_PPQtzx_onehalfQtxy*1.154701*-0.666667
        
        #####SDz2|SPz#####
        riYX[...,55]=DSQtzx_onehalfQtxy_SPUz*1.154701*1.000000
        
        #####SDz2|PzPz#####
        riYX[...,100]=DSQtzx_onehalfQtxy_Sq*1.154701*1.000000+DSQtzx_onehalfQtxy_PPQtzx_onehalfQtxy*1.154701*1.333333
        
        #####PxDz2|SPx#####
        riYX[...,147]=DPUpi_SPUpi*-0.577350*1.000000
        
        #####PxDz2|PzPx#####
        riYX[...,192]=DPUpi_PPQpiz*-0.577350*1.000000
        
        #####PyDz2|SPy#####
        riYX[...,283]=DPUpi_SPUpi*1.000000*1.000000*-0.577350
        
        #####PyDz2|PzPy#####
        riYX[...,328]=-DPUpi_PPQpiz*1.000000*1.000000*0.57735026918962995
        
        #####PzDz2|SS#####
        riYX[...,11]=DPUz_Sq*1.154701*1.000000
        
        #####PzDz2|PxPx#####
        riYX[...,236]=DPUz_Sq*1.154701*1.000000+DPUz_PPQtzx_onehalfQtxy*1.154701*-0.666667
        
        #####PzDz2|PyPy#####
        riYX[...,416]=DPUz_Sq*1.154701*1.000000+DPUz_PPQtzx_onehalfQtxy*1.154701*-0.666667
        
        #####PzDz2|SPz#####
        riYX[...,56]=DPUz_SPUz*1.154701*1.000000
        
        #####PzDz2|PzPz#####
        riYX[...,101]=DPUz_Sq*1.154701*1.000000+DPUz_PPQtzx_onehalfQtxy*1.154701*1.333333
        
        #####Dz2Dxy|PxPy#####
        riYX[...,400]=tildaDDQxy_PPQxy*1.000000*1.000000*1.1547005383793001
        
        #####Dz2Dxz|SPx#####
        riYX[...,154]=DDQpiz_SPUpi*0.577350*1.000000
        
        #####Dz2Dxz|PzPx#####
        riYX[...,199]=DDQpiz_PPQpiz*0.577350*1.000000
        
        #####Dz2Dz2|SS#####
        riYX[...,14]=DDq_Sq*1.000000*1.000000+DDQtzx_onehalfQtxy_Sq*1.333333*1.000000
        
        #####Dz2Dz2|PxPx#####
        riYX[...,239]=DDq_Sq*1.000000*1.000000+DDq_PPQtzx_onehalfQtxy*1.000000*-0.666667+DDQtzx_onehalfQtxy_Sq*1.333333*1.000000+DDQtzx_onehalfQtxy_PPQtzx_onehalfQtxy*1.333333*-0.666667
        
        #####Dz2Dz2|PyPy#####
        riYX[...,419]=DDq_Sq*1.000000*1.000000+DDq_PPQtzx_onehalfQtxy*1.000000*-0.666667+DDQtzx_onehalfQtxy_Sq*1.333333*1.000000+DDQtzx_onehalfQtxy_PPQtzx_onehalfQtxy*1.333333*-0.666667
        
        #####Dz2Dz2|SPz#####
        riYX[...,59]=DDq_SPUz*1.000000*1.000000+DDQtzx_onehalfQtxy_SPUz*1.333333*1.000000
        
        #####Dz2Dz2|PzPz#####
        riYX[...,104]=DDq_Sq*1.000000*1.000000+DDq_PPQtzx_onehalfQtxy*1.000000*1.333333+DDQtzx_onehalfQtxy_Sq*1.333333*1.000000+DDQtzx_onehalfQtxy_PPQtzx_onehalfQtxy*1.333333*1.333333
        
        #####SDyz|SPy#####
        riYX[...,291]=DSQpiz_SPUpi*1.000000*1.000000
        
        #####SDyz|PzPy#####
        riYX[...,336]=DSQpiz_PPQpiz*1.000000*1.000000
        
        #####PyDyz|SS#####
        riYX[...,24]=DPUz_Sq*1.000000*1.000000
        
        #####PyDyz|PxPx#####
        riYX[...,249]=DPUz_Sq*1.000000*1.000000+DPUz_PPQtzx_onehalfQtxy*1.000000*-0.666667
        
        #####PyDyz|PyPy#####
        riYX[...,429]=DPUz_Sq*1.000000*1.000000+DPUz_PPQtzx_onehalfQtxy*1.000000*-0.666667
        
        #####PyDyz|SPz#####
        riYX[...,69]=DPUz_SPUz*1.000000*1.000000
        
        #####PyDyz|PzPz#####
        riYX[...,114]=DPUz_Sq*1.000000*1.000000+DPUz_PPQtzx_onehalfQtxy*1.000000*1.333333
        
        #####PzDyz|SPy#####
        riYX[...,292]=DPUpi_SPUpi*1.000000*1.000000
        
        #####PzDyz|PzPy#####
        riYX[...,337]=DPUpi_PPQpiz*1.000000*1.000000
        
        #####DyzDxy|SPx#####
        riYX[...,177]=DDQpiz_SPUpi*1.000000*1.000000
        
        #####DyzDxy|PzPx#####
        riYX[...,222]=DDQpiz_PPQpiz*1.000000*1.000000
        
        #####DxzDyz|PxPy#####
        riYX[...,386]=tildaDDQxy_PPQxy*1.000000*1.000000
        
        #####Dz2Dyz|SPy#####
        riYX[...,295]=DDQpiz_SPUpi*1.000000*1.000000*0.57735026918962995
        
        #####Dz2Dyz|PzPy#####
        riYX[...,340]=DDQpiz_PPQpiz*1.000000*1.000000*0.57735026918962995
        
        #####DyzDyz|SS#####
        riYX[...,27]=DDq_Sq*1.000000*1.000000+DDQtzx_onehalfQtxy_Sq*0.666667*1.000000
        
        #####DyzDyz|PxPx#####
        riYX[...,252]=DDq_Sq*1.000000*1.000000+DDq_PPQtzx_onehalfQtxy*1.000000*-0.666667+DDQtzx_onehalfQtxy_Sq*0.666667*1.000000+DDQtzx_onehalfQtxy_PPQtzx_onehalfQtxy*0.666667*-0.666667-DDQxy_PPQxy*1.000000*1.000000
        
        #####DyzDyz|PyPy#####
        riYX[...,432]=DDq_Sq*1.000000*1.000000+DDq_PPQtzx_onehalfQtxy*1.000000*-0.666667+DDQtzx_onehalfQtxy_Sq*0.666667*1.000000+DDQtzx_onehalfQtxy_PPQtzx_onehalfQtxy*0.666667*-0.666667+DDQxy_PPQxy*1.000000*1.000000
        
        #####DyzDyz|SPz#####
        riYX[...,72]=DDq_SPUz*1.000000*1.000000+DDQtzx_onehalfQtxy_SPUz*0.666667*1.000000
        
        #####DyzDyz|PzPz#####
        riYX[...,117]=DDq_Sq*1.000000*1.000000+DDq_PPQtzx_onehalfQtxy*1.000000*1.333333+DDQtzx_onehalfQtxy_Sq*0.666667*1.000000+DDQtzx_onehalfQtxy_PPQtzx_onehalfQtxy*0.666667*1.333333
        
        #####SDx2-y2|PxPx#####
        riYX[...,253]=DSQxy_PPQxy*1.000000*1.000000
        
        #####SDx2-y2|PyPy#####
        riYX[...,433]=-DSQxy_PPQxy*1.000000*1.000000
        
        #####PxDx2-y2|SPx#####
        riYX[...,165]=DPUpi_SPUpi*1.000000*1.000000
        
        #####PxDx2-y2|PzPx#####
        riYX[...,210]=DPUpi_PPQpiz*1.000000*1.000000
        
        #####PyDx2-y2|SPy#####
        riYX[...,301]=-DPUpi_SPUpi*1.000000*1.000000
        
        #####PyDx2-y2|PzPy#####
        riYX[...,346]=-DPUpi_PPQpiz*1.000000*1.000000
        
        #####DxzDx2-y2|SPx#####
        riYX[...,168]=DDQpiz_SPUpi*1.000000*1.000000
        
        #####DxzDx2-y2|PzPx#####
        riYX[...,213]=DDQpiz_PPQpiz*1.000000*1.000000
        
        #####Dz2Dx2-y2|PxPx#####
        riYX[...,257]=DDQxy_PPQxy*1.154701*1.000000
        
        #####Dz2Dx2-y2|PyPy#####
        riYX[...,437]=-DDQxy_PPQxy*1.154701*1.000000
        
        #####DyzDx2-y2|SPy#####
        riYX[...,304]=-DDQpiz_SPUpi*1.000000*1.000000
        
        #####DyzDx2-y2|PzPy#####
        riYX[...,349]=-DDQpiz_PPQpiz*1.000000*1.000000
        
        #####Dx2-y2Dx2-y2|SS#####
        riYX[...,35]=DDq_Sq*1.000000*1.000000+DDQtzx_onehalfQtxy_Sq*-1.333333*1.000000
        
        #####Dx2-y2Dx2-y2|PxPx#####
        riYX[...,260]=DDq_Sq*1.000000*1.000000+DDq_PPQtzx_onehalfQtxy*1.000000*-0.666667+DDQtzx_onehalfQtxy_Sq*-1.333333*1.000000+DDQtzx_onehalfQtxy_PPQtzx_onehalfQtxy*-1.333333*-0.666667
        
        #####Dx2-y2Dx2-y2|PyPy#####
        riYX[...,440]=DDq_Sq*1.000000*1.000000+DDq_PPQtzx_onehalfQtxy*1.000000*-0.666667+DDQtzx_onehalfQtxy_Sq*-1.333333*1.000000+DDQtzx_onehalfQtxy_PPQtzx_onehalfQtxy*-1.333333*-0.666667
        
        #####Dx2-y2Dx2-y2|SPz#####
        riYX[...,80]=DDq_SPUz*1.000000*1.000000+DDQtzx_onehalfQtxy_SPUz*-1.333333*1.000000
        
        #####Dx2-y2Dx2-y2|PzPz#####
        riYX[...,125]=DDq_Sq*1.000000*1.000000+DDq_PPQtzx_onehalfQtxy*1.000000*1.333333+DDQtzx_onehalfQtxy_Sq*-1.333333*1.000000+DDQtzx_onehalfQtxy_PPQtzx_onehalfQtxy*-1.333333*1.333333
        
        

        coreYX = torch.zeros(YX.sum(),55,dtype=dtype, device=device)
        coreYXLocal = torch.zeros(YX.sum(),55,dtype=dtype, device=device)
        YYX = (ni[XX]>1) & (nj[XX]>1) & (((ni[XX] > 12) & (ni[XX] <18)) | ((ni[XX] > 20) & (ni[XX] <30)) | ((ni[XX] > 32) & (ni[XX] <36)) | ((ni[XX] > 38) & (ni[XX] <48)) | ((ni[XX] > 50) & (ni[XX] <54)) | ((ni[XX] > 70) & (ni[XX] <80)) | (ni[XX] ==57)) & \
         ((nj[XX] <= 12) | ((nj[XX] >= 18) & (nj[XX] <=20)) | ((nj[XX] >= 30) & (nj[XX] <= 32)) | ((nj[XX] >= 36) & (nj[XX] <= 38)) | ((nj[XX] >= 48) & (nj[XX] <= 50)) | ((nj[XX] >= 54) & (nj[XX] <= 56)) | ((nj[XX] >= 80) & (nj[XX] <= 83)))

        coreYXLocal[...,0] = tore[ni[YX]]*riPM6a[YYX,1-1]
        coreYXLocal[...,2] = tore[ni[YX]]*riPM6a[YYX,12-1]   ##PP
        coreYXLocal[...,6] = tore[ni[YX]]*riPM6a[YYX,5-1]  ##SP
        coreYXLocal[...,9] = tore[ni[YX]]*riPM6a[YYX,11-1]  ##PP
        coreYXLocal[...,10] = tore[nj[YX]]*riPM6b[YYX,1-1]
        coreYXLocal[...,12] = tore[nj[YX]]*riPM6b[YYX,4-1]
        coreYXLocal[...,16] = tore[nj[YX]]*riPM6b[YYX,2-1]
        coreYXLocal[...,19] = tore[nj[YX]]*riPM6b[YYX,3-1]
        coreYXLocal[...,24] = tore[nj[YX]]*riYX[...,44]
        coreYXLocal[...,26] = tore[nj[YX]]*riYX[...,17]
        coreYXLocal[...,30] = tore[nj[YX]]*riYX[...,20]
        coreYXLocal[...,31] = tore[nj[YX]]*riYX[...,10]
        coreYXLocal[...,34] = tore[nj[YX]]*riYX[...,11]
        coreYXLocal[...,37] = tore[nj[YX]]*riYX[...,14]


        coreYX[...,:10] = RotateCore(coreYXLocal[...,:10],rotationMatrix[YX,...,...],2)
        coreYX[...,10:] = RotateCore(coreYXLocal[...,10:],rotationMatrix[YX,...,...],3)

    else:
        coreYX = torch.zeros(YX.sum(),55,dtype=dtype, device=device)
        riYX =  torch.zeros(YX.sum(),450,dtype=dtype, device=device)

    if(YY.sum() > 0 ):

        riYY =  torch.zeros(YY.sum(),2025,dtype=dtype, device=device)
        
        r0 = r0d[YY] 
        da0 = da0d[YY]
        db0 = db0d[YY]
        qa0 = qa0d[YY]
        qb0 = qb0d[YY]
        dpa0 = dpa0d[YY]
        dpb0 = dpb0d[YY]
        dsa0 = dsa0d[YY]
        dsb0 = dsb0d[YY]

        ######
        ### dda0 and ddb0 probably should be swapped. Don't know why.
        ######
        dda0 = dda0d[YY]
        ddb0 = ddb0d[YY]
        ######



        rho0a = rho0ad[YY]
        rho0b = rho0bd[YY]
        rho1a = rho1ad[YY]
        rho1b = rho1bd[YY]
        rho2a = rho2ad[YY]
        rho2b = rho2bd[YY]
        rho3a = rho3ad[YY]
        rho3b = rho3bd[YY]
        rho4a = rho4ad[YY]
        rho4b = rho4bd[YY]
        rho5a = rho5ad[YY]
        rho5b = rho5bd[YY]
        rho6a = rho6ad[YY]
        rho6b = rho6bd[YY]
    
        rhoSS = (rho0b+rho0a)**2
        rhoSDD0 = (rho0b+rho3a)**2 
        rhoSSP = (rho0b+rho1a)**2
        rhoSPD = (rho0b+rho4a)**2
        rhoSPP = (rho0b+rho2a)**2
        rhoSSD = (rho0b+rho5a)**2
        rhoSDD = (rho0b+rho6a)**2

####
        rhoD0SD = (rho0a+rho3b)**2
        rhoSPS = (rho0a+rho1b)**2
        rhoPDS = (rho0a+rho4b)**2
        rhoPPS = (rho0a+rho2b)**2
        rhoSDS = (rho0a+rho5b)**2
        rhoDDS = (rho0a+rho6b)**2
####
    
        rhoDD0DD0 = (rho3b+rho3a)**2
        rhoDD0SP = (rho3b+rho1a)**2
        rhoDD0PD = (rho3b+rho4a)**2
    
        rhoDD0PP = (rho3b+rho2a)**2
        rhoDD0SD = (rho3b+rho5a)**2
        rhoDD0DD = (rho3b+rho6a)**2

####
        rhoSPDD0 = (rho3a+rho1b)**2
        rhoPDDD0 = (rho3a+rho4b)**2

        rhoPPDD0 = (rho3a+rho2b)**2
        rhoSDDD0 = (rho3a+rho5b)**2
        rhoDDDD0 = (rho3a+rho6b)**2
####
    
        rhoSPSP = (rho1b+rho1a)**2
        rhoSPPD = (rho1b+rho4a)**2
        rhoPDPD = (rho4b+rho4a)**2
####
        rhoPDSP = (rho1a+rho4b)**2
####
    
        rhoSPPP = (rho1b+rho2a)**2
        rhoSPSD = (rho1b+rho5a)**2
        rhoSPDD = (rho1b+rho6a)**2
        rhoPDPP = (rho4b+rho2a)**2
        rhoPDSD = (rho4b+rho5a)**2
        rhoPDDD = (rho4b+rho6a)**2

####
        rhoPPSP = (rho1a+rho2b)**2
        rhoSDSP = (rho1a+rho5b)**2
        rhoDDSP = (rho1a+rho6b)**2
        rhoPPPD = (rho4a+rho2b)**2
        rhoSDPD = (rho4a+rho5b)**2
        rhoDDPD = (rho4a+rho6b)**2
####
    
        rhoPPPP = (rho2b+rho2a)**2
        rhoPPDD = (rho2b+rho6a)**2
        rhoSDDD = (rho5b+rho6a)**2
        rhoSDSD = (rho5b+rho5a)**2
        rhoDDDD = (rho6b+rho6a)**2
        rhoPPSD = (rho2b+rho5a)**2

####
        rhoDDPP = (rho2a+rho6b)**2
        rhoDDSD = (rho5a+rho6b)**2
        rhoSDPP = (rho2a+rho5b)**2
####
    
        
    
        ########    00|00    ########
        qq = ev/torch.sqrt(r0**2+rhoSS)
        Sq_DDq = ev/torch.sqrt(r0**2+rhoD0SD)
        DDq_DDq = ev/torch.sqrt(r0**2+rhoDD0DD0)
        #############################
        DDq_Sq = ev/torch.sqrt(r0**2+rhoSDD0)


        ########    00|10    ########
        Sq_DPUz = -(ev1/torch.sqrt((r0+dpb0)**2+rhoPDS)-ev1/torch.sqrt((r0-dpb0)**2+rhoPDS))
        Sq_SPUz = -(ev1/torch.sqrt((r0+db0)**2+rhoSPS)-ev1/torch.sqrt((r0-db0)**2+rhoSPS))
    
        DDq_DPUz = -(ev1/torch.sqrt((r0+dpb0)**2+rhoPDDD0)-ev1/torch.sqrt((r0-dpb0)**2+rhoPDDD0))
        DDq_SPUz = -(ev1/torch.sqrt((r0+db0)**2+rhoSPDD0)-ev1/torch.sqrt((r0-db0)**2+rhoSPDD0))
    
        #############################
        DPUz_Sq = (ev1/torch.sqrt((r0+dpa0)**2+rhoSPD)-ev1/torch.sqrt((r0-dpa0)**2+rhoSPD))
        SPUz_Sq = (ev1/torch.sqrt((r0+da0)**2+rhoSSP)-ev1/torch.sqrt((r0-da0)**2+rhoSSP))
        DPUz_DDq = (ev1/torch.sqrt((r0+dpa0)**2+rhoDD0PD)-ev1/torch.sqrt((r0-dpa0)**2+rhoDD0PD))
        SPUz_DDq = (ev1/torch.sqrt((r0+da0)**2+rhoDD0SP)-ev1/torch.sqrt((r0-da0)**2+rhoDD0SP))
    
        ########    00|20    ########
        qa0 = qa0*math.sqrt(2)
        qb0 = qb0*math.sqrt(2)
        ### $$$$ possible fix is to comment line below ###
        #dda0 = dda0*math.sqrt(2)
        dsa0 = dsa0*math.sqrt(2)


        Sq_DDQtzx_onehalfQtxy = ev2/torch.sqrt((r0-ddb0)**2+rhoDDS)+ev2/torch.sqrt((r0+ddb0)**2+rhoDDS)-ev1/torch.sqrt((r0)**2+(ddb0)**2+rhoDDS)
        Sq_DSQtzx_onehalfQtxy = ev2/torch.sqrt((r0-dsb0)**2+rhoSDS)+ev2/torch.sqrt((r0+dsb0)**2+rhoSDS)-ev1/torch.sqrt((r0)**2+(dsb0)**2+rhoSDS)
        Sq_PPQtzx_onehalfQtxy = ev2/torch.sqrt((r0-(2)*qb0)**2+rhoPPS)+ev2/torch.sqrt((r0+(2)*qb0)**2+rhoPPS)-ev1/torch.sqrt((r0)**2+((0)*db0)**2+rhoPPS)

        ### $$$$ Possible fix ###
        #DDq_DDQtzx_onehalfQtxy = ev2/torch.sqrt((r0-ddb0)**2+rhoDD0DD)+ev2/torch.sqrt((r0+ddb0)**2+rhoDD0DD)-ev1/torch.sqrt((r0)**2+(ddb0)**2+rhoDD0DD)
        DDq_DDQtzx_onehalfQtxy =  ev2/torch.sqrt((r0-ddb0)**2+rhoDD0DD)-ev1/torch.sqrt((r0)**2+(ddb0)**2+rhoDD0DD)+ev2/torch.sqrt((r0+ddb0)**2+rhoDD0DD)

        DDq_DSQtzx_onehalfQtxy = ev2/torch.sqrt((r0-dsb0)**2+rhoSDDD0)+ev2/torch.sqrt((r0+dsb0)**2+rhoSDDD0)-ev1/torch.sqrt((r0)**2+(dsb0)**2+rhoSDDD0)
        DDq_PPQtzx_onehalfQtxy = ev2/torch.sqrt((r0-qb0)**2+rhoPPDD0)+ev2/torch.sqrt((r0+qb0)**2+rhoPPDD0)-ev1/torch.sqrt((r0)**2+(qb0)**2+rhoPPDD0)
        #############################
    
        DDQtzx_onehalfQtxy_Sq = ev2/torch.sqrt((r0-dda0)**2+rhoSDD)+ev2/torch.sqrt((r0+dda0)**2+rhoSDD)-ev1/torch.sqrt((r0)**2+(dda0)**2+rhoSDD)
        DSQtzx_onehalfQtxy_Sq = ev2/torch.sqrt((r0-dsa0)**2+rhoSSD)+ev2/torch.sqrt((r0+dsa0)**2+rhoSSD)-ev1/torch.sqrt((r0)**2+(dsa0)**2+rhoSSD)
        PPQtzx_onehalfQtxy_Sq = ev2/torch.sqrt((r0-qa0)**2+rhoSPP)+ev2/torch.sqrt((r0+qa0)**2+rhoSPP)-ev1/torch.sqrt((r0)**2+(da0)**2+rhoSPP)
        
        ### $$$$ Possible fix ###
        #DDQtzx_onehalfQtxy_DDq = ev2/torch.sqrt((r0-dda0)**2+rhoDD0DD)+ev2/torch.sqrt((r0+dda0)**2+rhoDD0DD)-ev1/torch.sqrt((r0)**2+(dda0)**2+rhoDD0DD)
        DDQtzx_onehalfQtxy_DDq = ev2/torch.sqrt((r0-dda0)**2+rhoDDDD0)+ev2/torch.sqrt((r0+dda0)**2+rhoDDDD0)-ev1/torch.sqrt((r0)**2+(dda0)**2+rhoDDDD0)

        DSQtzx_onehalfQtxy_DDq = ev2/torch.sqrt((r0-dsa0)**2+rhoDD0SD)+ev2/torch.sqrt((r0+dsa0)**2+rhoDD0SD)-ev1/torch.sqrt((r0)**2+(dsa0)**2+rhoDD0SD)
        PPQtzx_onehalfQtxy_DDq = ev2/torch.sqrt((r0-qa0)**2+rhoDD0PP)+ev2/torch.sqrt((r0+qa0)**2+rhoDD0PP)-ev1/torch.sqrt((r0)**2+(qa0)**2+rhoDD0PP)
    
        qa0 = qa0/math.sqrt(2)
        qb0 = qb0/math.sqrt(2)
        dda0 = dda0/math.sqrt(2)
        dsa0 = dsa0/math.sqrt(2)

    
     
        ########    11|11    ########
        SPUpi_SPUpi = ev1/torch.sqrt(r0**2+(da0-qb0)**2+rhoSPSP)-ev1/torch.sqrt(r0**2+(da0+qb0)**2+rhoSPSP)
        SPUpi_DPUpi = ev1/torch.sqrt(r0**2+(da0-dpb0)**2+rhoPDSP)-ev1/torch.sqrt(r0**2+(da0+dpb0)**2+rhoPDSP)
        DPUpi_DPUpi = ev1/torch.sqrt(r0**2+(dpa0-dpb0)**2+rhoPDPD)-ev1/torch.sqrt(r0**2+(dpa0+dpb0)**2+rhoPDPD)
        #############################
        DPUpi_SPUpi = ev1/torch.sqrt(r0**2+(db0-dpa0)**2+rhoSPPD)-ev1/torch.sqrt(r0**2+(db0+dpa0)**2+rhoSPPD)
    
        ########    10|10    ########
        SPUz_SPUz =  ev2/torch.sqrt((r0+da0-db0)**2+rhoSPSP) - \
                     ev2/torch.sqrt((r0+da0+db0)**2+rhoSPSP) - \
                     ev2/torch.sqrt((r0-da0-db0)**2+rhoSPSP) + \
                     ev2/torch.sqrt((r0-da0+db0)**2+rhoSPSP)
        SPUz_DPUz =  ev2/torch.sqrt((r0+da0-dpb0)**2+rhoPDSP) - \
                     ev2/torch.sqrt((r0+da0+dpb0)**2+rhoPDSP) - \
                     ev2/torch.sqrt((r0-da0-dpb0)**2+rhoPDSP) + \
                     ev2/torch.sqrt((r0-da0+dpb0)**2+rhoPDSP)
        DPUz_DPUz =  ev2/torch.sqrt((r0+dpa0-dpb0)**2+rhoPDPD) - \
                     ev2/torch.sqrt((r0+dpa0+dpb0)**2+rhoPDPD) - \
                     ev2/torch.sqrt((r0-dpa0-dpb0)**2+rhoPDPD) + \
                     ev2/torch.sqrt((r0-dpa0+dpb0)**2+rhoPDPD)
        #############################
        DPUz_SPUz =  ev2/torch.sqrt((r0+db0-dpa0)**2+rhoSPPD) - \
                     ev2/torch.sqrt((r0+db0+dpa0)**2+rhoSPPD) - \
                     ev2/torch.sqrt((r0-db0-dpa0)**2+rhoSPPD) + \
                     ev2/torch.sqrt((r0-db0+dpa0)**2+rhoSPPD)
    
    
        ########    11|21    ########
        dsb0 = dsb0/math.sqrt(2)
        dsa0 = dsa0/math.sqrt(2)
        dda0 = dda0/math.sqrt(2)
        ddb0 = ddb0/math.sqrt(2)


        SPUpi_PPQpiz = -(-ev2/torch.sqrt((r0-qb0)**2+(da0-qb0)**2+rhoPPSP) + \
                       ev2/torch.sqrt((r0-qb0)**2+(da0+qb0)**2+rhoPPSP) + \
                       ev2/torch.sqrt((r0+qb0)**2+(da0-qb0)**2+rhoPPSP) - \
                       ev2/torch.sqrt((r0+qb0)**2+(da0+qb0)**2+rhoPPSP))
        SPUpi_DSQpiz = -(-ev2/torch.sqrt((r0-dsb0)**2+(da0-dsb0)**2+rhoSDSP) + \
                       ev2/torch.sqrt((r0-dsb0)**2+(da0+dsb0)**2+rhoSDSP) + \
                       ev2/torch.sqrt((r0+dsb0)**2+(da0-dsb0)**2+rhoSDSP) - \
                       ev2/torch.sqrt((r0+dsb0)**2+(da0+dsb0)**2+rhoSDSP))
        SPUpi_DDQpiz = -(-ev2/torch.sqrt((r0-ddb0)**2+(da0-ddb0)**2+rhoDDSP) + \
                       ev2/torch.sqrt((r0-ddb0)**2+(da0+ddb0)**2+rhoDDSP) + \
                       ev2/torch.sqrt((r0+ddb0)**2+(da0-ddb0)**2+rhoDDSP) - \
                       ev2/torch.sqrt((r0+ddb0)**2+(da0+ddb0)**2+rhoDDSP))
        DPUpi_PPQpiz = -(-ev2/torch.sqrt((r0-qb0)**2+(dpa0-qb0)**2+rhoPPPD) + \
                       ev2/torch.sqrt((r0-qb0)**2+(dpa0+qb0)**2+rhoPPPD) + \
                       ev2/torch.sqrt((r0+qb0)**2+(dpa0-qb0)**2+rhoPPPD) - \
                       ev2/torch.sqrt((r0+qb0)**2+(dpa0+qb0)**2+rhoPPPD))
        DPUpi_DSQpiz = -(-ev2/torch.sqrt((r0-dsb0)**2+(dpa0-dsb0)**2+rhoSDPD) + \
                       ev2/torch.sqrt((r0-dsb0)**2+(dpa0+dsb0)**2+rhoSDPD) + \
                       ev2/torch.sqrt((r0+dsb0)**2+(dpa0-dsb0)**2+rhoSDPD) - \
                       ev2/torch.sqrt((r0+dsb0)**2+(dpa0+dsb0)**2+rhoSDPD))
        DPUpi_DDQpiz = -(-ev2/torch.sqrt((r0-ddb0)**2+(dpa0-ddb0)**2+rhoDDPD) + \
                       ev2/torch.sqrt((r0-ddb0)**2+(dpa0+ddb0)**2+rhoDDPD) + \
                       ev2/torch.sqrt((r0+ddb0)**2+(dpa0-ddb0)**2+rhoDDPD) - \
                       ev2/torch.sqrt((r0+ddb0)**2+(dpa0+ddb0)**2+rhoDDPD))
        #############################
        PPQpiz_SPUpi = (-ev2/torch.sqrt((r0-qa0)**2+(db0-qa0)**2+rhoSPPP) + \
                       ev2/torch.sqrt((r0-qa0)**2+(db0+qa0)**2+rhoSPPP) + \
                       ev2/torch.sqrt((r0+qa0)**2+(db0-qa0)**2+rhoSPPP) - \
                       ev2/torch.sqrt((r0+qa0)**2+(db0+qa0)**2+rhoSPPP))
        DSQpiz_SPUpi = (-ev2/torch.sqrt((r0-dsa0)**2+(db0-dsa0)**2+rhoSPSD) + \
                       ev2/torch.sqrt((r0-dsa0)**2+(db0+dsa0)**2+rhoSPSD) + \
                       ev2/torch.sqrt((r0+dsa0)**2+(db0-dsa0)**2+rhoSPSD) - \
                       ev2/torch.sqrt((r0+dsa0)**2+(db0+dsa0)**2+rhoSPSD))
        DDQpiz_SPUpi = (-ev2/torch.sqrt((r0-dda0)**2+(db0-dda0)**2+rhoSPDD) + \
                       ev2/torch.sqrt((r0-dda0)**2+(db0+dda0)**2+rhoSPDD) + \
                       ev2/torch.sqrt((r0+dda0)**2+(db0-dda0)**2+rhoSPDD) - \
                       ev2/torch.sqrt((r0+dda0)**2+(db0+dda0)**2+rhoSPDD))
    
        PPQpiz_DPUpi = (-ev2/torch.sqrt((r0-qa0)**2+(dpb0-qa0)**2+rhoPDPP) + \
                       ev2/torch.sqrt((r0-qa0)**2+(dpb0+qa0)**2+rhoPDPP) + \
                       ev2/torch.sqrt((r0+qa0)**2+(dpb0-qa0)**2+rhoPDPP) - \
                       ev2/torch.sqrt((r0+qa0)**2+(dpb0+qa0)**2+rhoPDPP))
        DSQpiz_DPUpi = (-ev2/torch.sqrt((r0-dsa0)**2+(dpb0-dsa0)**2+rhoPDSD) + \
                       ev2/torch.sqrt((r0-dsa0)**2+(dpb0+dsa0)**2+rhoPDSD) + \
                       ev2/torch.sqrt((r0+dsa0)**2+(dpb0-dsa0)**2+rhoPDSD) - \
                       ev2/torch.sqrt((r0+dsa0)**2+(dpb0+dsa0)**2+rhoPDSD))
        DDQpiz_DPUpi = (-ev2/torch.sqrt((r0-dda0)**2+(dpb0-dda0)**2+rhoPDDD) + \
                       ev2/torch.sqrt((r0-dda0)**2+(dpb0+dda0)**2+rhoPDDD) + \
                       ev2/torch.sqrt((r0+dda0)**2+(dpb0-dda0)**2+rhoPDDD) - \
                       ev2/torch.sqrt((r0+dda0)**2+(dpb0+dda0)**2+rhoPDDD))
        dsb0 = dsb0*math.sqrt(2)
        dsa0 = dsa0*math.sqrt(2)
        dda0 = dda0*math.sqrt(2)
        ddb0 = ddb0*math.sqrt(2)
 
    
    
    
    
        ########    10|20    ########
        qa0 = qa0*math.sqrt(2)
        qb0 = qb0*math.sqrt(2)
        ddb0 = ddb0*math.sqrt(2)
        dda0 = dda0*math.sqrt(2)
        SPUz_PPQtzx_onehalfQtxy = -ev3/torch.sqrt((r0-da0-math.sqrt(1)*qb0)**2+rhoPPSP) - \
                                   ev3/torch.sqrt((r0-da0+math.sqrt(1)*qb0)**2+rhoPPSP) + \
                                   ev3/torch.sqrt((r0+da0-math.sqrt(1)*qb0)**2+rhoPPSP) + \
                                   ev3/torch.sqrt((r0+da0+math.sqrt(1)*qb0)**2+rhoPPSP) + \
                                   ev2/torch.sqrt((r0-da0)**2+(math.sqrt(1)*qb0)**2+rhoPPSP) - \
                                   ev2/torch.sqrt((r0+da0)**2+(math.sqrt(1)*qb0)**2+rhoPPSP)
        SPUz_DSQtzx_onehalfQtxy = -ev3/torch.sqrt((r0-da0-math.sqrt(1)*dsb0)**2+rhoSDSP) - \
                                   ev3/torch.sqrt((r0-da0+math.sqrt(1)*dsb0)**2+rhoSDSP) + \
                                   ev3/torch.sqrt((r0+da0-math.sqrt(1)*dsb0)**2+rhoSDSP) + \
                                   ev3/torch.sqrt((r0+da0+math.sqrt(1)*dsb0)**2+rhoSDSP) + \
                                   ev2/torch.sqrt((r0-da0)**2+(math.sqrt(1)*dsb0)**2+rhoSDSP) - \
                                   ev2/torch.sqrt((r0+da0)**2+(math.sqrt(1)*dsb0)**2+rhoSDSP)
        SPUz_DDQtzx_onehalfQtxy = -ev3/torch.sqrt((r0-da0-math.sqrt(1)*ddb0)**2+rhoDDSP) - \
                                   ev3/torch.sqrt((r0-da0+math.sqrt(1)*ddb0)**2+rhoDDSP) + \
                                   ev3/torch.sqrt((r0+da0-math.sqrt(1)*ddb0)**2+rhoDDSP) + \
                                   ev3/torch.sqrt((r0+da0+math.sqrt(1)*ddb0)**2+rhoDDSP) + \
                                   ev2/torch.sqrt((r0-da0)**2+(math.sqrt(1)*ddb0)**2+rhoDDSP) - \
                                   ev2/torch.sqrt((r0+da0)**2+(math.sqrt(1)*ddb0)**2+rhoDDSP)
    
        DPUz_PPQtzx_onehalfQtxy = -ev3/torch.sqrt((r0-dpa0-math.sqrt(1)*qb0)**2+rhoPPPD) - \
                                   ev3/torch.sqrt((r0-dpa0+math.sqrt(1)*qb0)**2+rhoPPPD) + \
                                   ev3/torch.sqrt((r0+dpa0-math.sqrt(1)*qb0)**2+rhoPPPD) + \
                                   ev3/torch.sqrt((r0+dpa0+math.sqrt(1)*qb0)**2+rhoPPPD) + \
                                   ev2/torch.sqrt((r0-dpa0)**2+(math.sqrt(1)*qb0)**2+rhoPPPD) - \
                                   ev2/torch.sqrt((r0+dpa0)**2+(math.sqrt(1)*qb0)**2+rhoPPPD)
        DPUz_DSQtzx_onehalfQtxy = -ev3/torch.sqrt((r0-dpa0-math.sqrt(1)*dsb0)**2+rhoSDPD) - \
                                   ev3/torch.sqrt((r0-dpa0+math.sqrt(1)*dsb0)**2+rhoSDPD) + \
                                   ev3/torch.sqrt((r0+dpa0-math.sqrt(1)*dsb0)**2+rhoSDPD) + \
                                   ev3/torch.sqrt((r0+dpa0+math.sqrt(1)*dsb0)**2+rhoSDPD) + \
                                   ev2/torch.sqrt((r0-dpa0)**2+(math.sqrt(1)*dsb0)**2+rhoSDPD) - \
                                   ev2/torch.sqrt((r0+dpa0)**2+(math.sqrt(1)*dsb0)**2+rhoSDPD)
        DPUz_DDQtzx_onehalfQtxy = -ev3/torch.sqrt((r0-dpa0-math.sqrt(1)*ddb0)**2+rhoDDPD) - \
                                   ev3/torch.sqrt((r0-dpa0+math.sqrt(1)*ddb0)**2+rhoDDPD) + \
                                   ev3/torch.sqrt((r0+dpa0-math.sqrt(1)*ddb0)**2+rhoDDPD) + \
                                   ev3/torch.sqrt((r0+dpa0+math.sqrt(1)*ddb0)**2+rhoDDPD) + \
                                   ev2/torch.sqrt((r0-dpa0)**2+(math.sqrt(1)*ddb0)**2+rhoDDPD) - \
                                   ev2/torch.sqrt((r0+dpa0)**2+(math.sqrt(1)*ddb0)**2+rhoDDPD)
        #############################
        PPQtzx_onehalfQtxy_SPUz = -(-ev3/torch.sqrt((r0-db0-math.sqrt(1)*qa0)**2+rhoSPPP) - \
                                   ev3/torch.sqrt((r0-db0+math.sqrt(1)*qa0)**2+rhoSPPP) + \
                                   ev3/torch.sqrt((r0+db0-math.sqrt(1)*qa0)**2+rhoSPPP) + \
                                   ev3/torch.sqrt((r0+db0+math.sqrt(1)*qa0)**2+rhoSPPP) + \
                                   ev2/torch.sqrt((r0-db0)**2+(math.sqrt(1)*qa0)**2+rhoSPPP) - \
                                   ev2/torch.sqrt((r0+db0)**2+(math.sqrt(1)*qa0)**2+rhoSPPP))
        DSQtzx_onehalfQtxy_SPUz = -(-ev3/torch.sqrt((r0-db0-math.sqrt(1)*dsa0)**2+rhoSPSD) - \
                                   ev3/torch.sqrt((r0-db0+math.sqrt(1)*dsa0)**2+rhoSPSD) + \
                                   ev3/torch.sqrt((r0+db0-math.sqrt(1)*dsa0)**2+rhoSPSD) + \
                                   ev3/torch.sqrt((r0+db0+math.sqrt(1)*dsa0)**2+rhoSPSD) + \
                                   ev2/torch.sqrt((r0-db0)**2+(math.sqrt(1)*dsa0)**2+rhoSPSD) - \
                                   ev2/torch.sqrt((r0+db0)**2+(math.sqrt(1)*dsa0)**2+rhoSPSD))
        DDQtzx_onehalfQtxy_SPUz = -(-ev3/torch.sqrt((r0-db0-math.sqrt(1)*dda0)**2+rhoSPDD) - \
                                   ev3/torch.sqrt((r0-db0+math.sqrt(1)*dda0)**2+rhoSPDD) + \
                                   ev3/torch.sqrt((r0+db0-math.sqrt(1)*dda0)**2+rhoSPDD) + \
                                   ev3/torch.sqrt((r0+db0+math.sqrt(1)*dda0)**2+rhoSPDD) + \
                                   ev2/torch.sqrt((r0-db0)**2+(math.sqrt(1)*dda0)**2+rhoSPDD) - \
                                   ev2/torch.sqrt((r0+db0)**2+(math.sqrt(1)*dda0)**2+rhoSPDD))
        PPQtzx_onehalfQtxy_DPUz = -(-ev3/torch.sqrt((r0-dpb0-math.sqrt(1)*qa0)**2+rhoPDPP) - \
                                   ev3/torch.sqrt((r0-dpb0+math.sqrt(1)*qa0)**2+rhoPDPP) + \
                                   ev3/torch.sqrt((r0+dpb0-math.sqrt(1)*qa0)**2+rhoPDPP) + \
                                   ev3/torch.sqrt((r0+dpb0+math.sqrt(1)*qa0)**2+rhoPDPP) + \
                                   ev2/torch.sqrt((r0-dpb0)**2+(math.sqrt(1)*qa0)**2+rhoPDPP) - \
                                   ev2/torch.sqrt((r0+dpb0)**2+(math.sqrt(1)*qa0)**2+rhoPDPP))
        DSQtzx_onehalfQtxy_DPUz = -(-ev3/torch.sqrt((r0-dpb0-math.sqrt(1)*dsa0)**2+rhoPDSD) - \
                                   ev3/torch.sqrt((r0-dpb0+math.sqrt(1)*dsa0)**2+rhoPDSD) + \
                                   ev3/torch.sqrt((r0+dpb0-math.sqrt(1)*dsa0)**2+rhoPDSD) + \
                                   ev3/torch.sqrt((r0+dpb0+math.sqrt(1)*dsa0)**2+rhoPDSD) + \
                                   ev2/torch.sqrt((r0-dpb0)**2+(math.sqrt(1)*dsa0)**2+rhoPDSD) - \
                                   ev2/torch.sqrt((r0+dpb0)**2+(math.sqrt(1)*dsa0)**2+rhoPDSD))
        DDQtzx_onehalfQtxy_DPUz = -(-ev3/torch.sqrt((r0-dpb0-math.sqrt(1)*dda0)**2+rhoPDDD) - \
                                   ev3/torch.sqrt((r0-dpb0+math.sqrt(1)*dda0)**2+rhoPDDD) + \
                                   ev3/torch.sqrt((r0+dpb0-math.sqrt(1)*dda0)**2+rhoPDDD) + \
                                   ev3/torch.sqrt((r0+dpb0+math.sqrt(1)*dda0)**2+rhoPDDD) + \
                                   ev2/torch.sqrt((r0-dpb0)**2+(math.sqrt(1)*dda0)**2+rhoPDDD) - \
                                   ev2/torch.sqrt((r0+dpb0)**2+(math.sqrt(1)*dda0)**2+rhoPDDD))
        qa0 = qa0/math.sqrt(2)
        qb0 = qb0/math.sqrt(2)
        ddb0 = ddb0/math.sqrt(2)
        dda0 = dda0/math.sqrt(2)


        ########    20|20    ########
        qa0=qa0*math.sqrt(2)
        qb0 = qb0*math.sqrt(2)
        ### $$$$ possible fix is to comment line below ###
        #ddb0 = ddb0*math.sqrt(2)
        dda0 = dda0*math.sqrt(2)
        dsa0 = dsa0*math.sqrt(2)
        dsb0 = dsb0*math.sqrt(2)

        PPQtzx_onehalfQtxy_PPQtzx_onehalfQtxy =(ev3/torch.sqrt(r0**2+(qa0-qb0)**2+rhoPPPP) + \
                                                ev3/torch.sqrt(r0**2+(qa0+qb0)**2+rhoPPPP) - \
                                                ev3/torch.sqrt((r0+qb0)**2+qa0**2+rhoPPPP) - \
                                                ev3/torch.sqrt((r0-qb0)**2+qa0**2+rhoPPPP) + \
                                                ev4/torch.sqrt((r0-qb0+qa0)**2+rhoPPPP) + \
                                                ev4/torch.sqrt((r0+qb0-qa0)**2+rhoPPPP) + \
                                                ev4/torch.sqrt((r0+qb0+qa0)**2+rhoPPPP) + \
                                                ev4/torch.sqrt((r0-qb0-qa0)**2+rhoPPPP) - \
                                                ev3/torch.sqrt((r0-qa0)**2+qb0**2+rhoPPPP) - \
                                                ev3/torch.sqrt((r0+qa0)**2+qb0**2+rhoPPPP))- \
                                              0.25*(ev2/torch.sqrt(r0**2+(qa0-qb0)**2+rhoPPPP) +\
                                                 ev2/torch.sqrt(r0**2+(qa0+qb0)**2+rhoPPPP) -\
                                                 ev1/torch.sqrt(r0**2+qa0**2+qb0**2+rhoPPPP))




        PPQtzx_onehalfQtxy_DSQtzx_onehalfQtxy = (ev3/torch.sqrt(r0**2+(qa0-dsb0)**2+rhoSDPP) + \
                                                ev3/torch.sqrt(r0**2+(qa0+dsb0)**2+rhoSDPP) - \
                                                ev3/torch.sqrt((r0+dsb0)**2+qa0**2+rhoSDPP) - \
                                                ev3/torch.sqrt((r0-dsb0)**2+qa0**2+rhoSDPP) + \
                                                ev4/torch.sqrt((r0-dsb0+qa0)**2+rhoSDPP) + \
                                                ev4/torch.sqrt((r0+dsb0-qa0)**2+rhoSDPP) + \
                                                ev4/torch.sqrt((r0+dsb0+qa0)**2+rhoSDPP) + \
                                                ev4/torch.sqrt((r0-dsb0-qa0)**2+rhoSDPP) - \
                                                ev3/torch.sqrt((r0-qa0)**2+dsb0**2+rhoSDPP) - \
                                                ev3/torch.sqrt((r0+qa0)**2+dsb0**2+rhoSDPP)) -  \
                                              0.25*(ev2/torch.sqrt(r0**2+(qa0-dsb0)**2+rhoSDPP) +\
                                                 ev2/torch.sqrt(r0**2+(qa0+dsb0)**2+rhoSDPP) -\
                                                 ev1/torch.sqrt(r0**2+qa0**2+dsb0**2+rhoSDPP))



        PPQtzx_onehalfQtxy_DDQtzx_onehalfQtxy = (ev3/torch.sqrt(r0**2+(qa0-ddb0)**2+rhoDDPP) + \
                                                ev3/torch.sqrt(r0**2+(qa0+ddb0)**2+rhoDDPP) - \
                                                ev3/torch.sqrt((r0+ddb0)**2+qa0**2+rhoDDPP) - \
                                                ev3/torch.sqrt((r0-ddb0)**2+qa0**2+rhoDDPP) + \
                                                ev4/torch.sqrt((r0-ddb0+qa0)**2+rhoDDPP) + \
                                                ev4/torch.sqrt((r0+ddb0-qa0)**2+rhoDDPP) + \
                                                ev4/torch.sqrt((r0+ddb0+qa0)**2+rhoDDPP) + \
                                                ev4/torch.sqrt((r0-ddb0-qa0)**2+rhoDDPP) - \
                                                ev3/torch.sqrt((r0-qa0)**2+ddb0**2+rhoDDPP) - \
                                                ev3/torch.sqrt((r0+qa0)**2+ddb0**2+rhoDDPP))- \
                                              0.25*(ev2/torch.sqrt(r0**2+(qa0-ddb0)**2+rhoDDPP) +\
                                                 ev2/torch.sqrt(r0**2+(qa0+ddb0)**2+rhoDDPP) -\
                                                 ev1/torch.sqrt(r0**2+qa0**2+ddb0**2+rhoDDPP))




        DSQtzx_onehalfQtxy_DDQtzx_onehalfQtxy = (ev3/torch.sqrt(r0**2+(dsa0-ddb0)**2+rhoDDSD) + \
                                                ev3/torch.sqrt(r0**2+(dsa0+ddb0)**2+rhoDDSD) - \
                                                ev3/torch.sqrt((r0+ddb0)**2+dsa0**2+rhoDDSD) - \
                                                ev3/torch.sqrt((r0-ddb0)**2+dsa0**2+rhoDDSD) + \
                                                ev4/torch.sqrt((r0-ddb0+dsa0)**2+rhoDDSD) + \
                                                ev4/torch.sqrt((r0+ddb0-dsa0)**2+rhoDDSD) + \
                                                ev4/torch.sqrt((r0+ddb0+dsa0)**2+rhoDDSD) + \
                                                ev4/torch.sqrt((r0-ddb0-dsa0)**2+rhoDDSD) - \
                                                ev3/torch.sqrt((r0-dsa0)**2+ddb0**2+rhoDDSD) - \
                                                ev3/torch.sqrt((r0+dsa0)**2+ddb0**2+rhoDDSD))- \
                                              0.25*(ev2/torch.sqrt(r0**2+(dsa0-ddb0)**2+rhoDDSD) +\
                                                 ev2/torch.sqrt(r0**2+(dsa0+ddb0)**2+rhoDDSD) -\
                                                 ev1/torch.sqrt(r0**2+dsa0**2+ddb0**2+rhoDDSD))


        DSQtzx_onehalfQtxy_DSQtzx_onehalfQtxy = (ev3/torch.sqrt(r0**2+(dsa0-dsb0)**2+rhoSDSD) + \
                                                ev3/torch.sqrt(r0**2+(dsa0+dsb0)**2+rhoSDSD) - \
                                                ev3/torch.sqrt((r0+dsb0)**2+dsa0**2+rhoSDSD) - \
                                                ev3/torch.sqrt((r0-dsb0)**2+dsa0**2+rhoSDSD) + \
                                                ev4/torch.sqrt((r0-dsb0+dsa0)**2+rhoSDSD) + \
                                                ev4/torch.sqrt((r0+dsb0-dsa0)**2+rhoSDSD) + \
                                                ev4/torch.sqrt((r0+dsb0+dsa0)**2+rhoSDSD) + \
                                                ev4/torch.sqrt((r0-dsb0-dsa0)**2+rhoSDSD) - \
                                                ev3/torch.sqrt((r0-dsa0)**2+dsb0**2+rhoSDSD) - \
                                                ev3/torch.sqrt((r0+dsa0)**2+dsb0**2+rhoSDSD))- \
                                              0.25*(ev2/torch.sqrt(r0**2+(dsa0-dsb0)**2+rhoSDSD) +\
                                                 ev2/torch.sqrt(r0**2+(dsa0+dsb0)**2+rhoSDSD) -\
                                                 ev1/torch.sqrt(r0**2+dsa0**2+dsb0**2+rhoSDSD))
 
        DDQtzx_onehalfQtxy_DDQtzx_onehalfQtxy = (ev3/torch.sqrt(r0**2+(dda0-ddb0)**2+rhoDDDD) + \
                                                ev3/torch.sqrt(r0**2+(dda0+ddb0)**2+rhoDDDD) - \
                                                ev3/torch.sqrt((r0+ddb0)**2+dda0**2+rhoDDDD) - \
                                                ev3/torch.sqrt((r0-ddb0)**2+dda0**2+rhoDDDD) + \
                                                ev4/torch.sqrt((r0-ddb0+dda0)**2+rhoDDDD) + \
                                                ev4/torch.sqrt((r0+ddb0-dda0)**2+rhoDDDD) + \
                                                ev4/torch.sqrt((r0+ddb0+dda0)**2+rhoDDDD) + \
                                                ev4/torch.sqrt((r0-ddb0-dda0)**2+rhoDDDD) - \
                                                ev3/torch.sqrt((r0-dda0)**2+ddb0**2+rhoDDDD) - \
                                                ev3/torch.sqrt((r0+dda0)**2+ddb0**2+rhoDDDD))- \
                                              0.25*(ev2/torch.sqrt(r0**2+(dda0-ddb0)**2+rhoDDDD) +\
                                                 ev2/torch.sqrt(r0**2+(dda0+ddb0)**2+rhoDDDD) -\
                                                 ev1/torch.sqrt(r0**2+dda0**2+ddb0**2+rhoDDDD))


        #############################


        DSQtzx_onehalfQtxy_PPQtzx_onehalfQtxy = (ev3/torch.sqrt(r0**2+(qa0-dsb0)**2+rhoPPSD) + \
                                                ev3/torch.sqrt(r0**2+(qa0+dsb0)**2+rhoPPSD) - \
                                                ev3/torch.sqrt((r0+dsb0)**2+qa0**2+rhoPPSD) - \
                                                ev3/torch.sqrt((r0-dsb0)**2+qa0**2+rhoPPSD) + \
                                                ev4/torch.sqrt((r0-dsb0+qa0)**2+rhoPPSD) + \
                                                ev4/torch.sqrt((r0+dsb0-qa0)**2+rhoPPSD) + \
                                                ev4/torch.sqrt((r0+dsb0+qa0)**2+rhoPPSD) + \
                                                ev4/torch.sqrt((r0-dsb0-qa0)**2+rhoPPSD) - \
                                                ev3/torch.sqrt((r0-qa0)**2+dsb0**2+rhoPPSD) - \
                                                ev3/torch.sqrt((r0+qa0)**2+dsb0**2+rhoPPSD)) -  \
                                              0.25*(ev2/torch.sqrt(r0**2+(qa0-dsb0)**2+rhoPPSD) +\
                                                 ev2/torch.sqrt(r0**2+(qa0+dsb0)**2+rhoPPSD) -\
                                                 ev1/torch.sqrt(r0**2+qa0**2+dsb0**2+rhoPPSD))

 

        DDQtzx_onehalfQtxy_PPQtzx_onehalfQtxy = (ev3/torch.sqrt(r0**2+(qa0-ddb0)**2+rhoPPDD) + \
                                                ev3/torch.sqrt(r0**2+(qa0+ddb0)**2+rhoPPDD) - \
                                                ev3/torch.sqrt((r0+ddb0)**2+qa0**2+rhoPPDD) - \
                                                ev3/torch.sqrt((r0-ddb0)**2+qa0**2+rhoPPDD) + \
                                                ev4/torch.sqrt((r0-ddb0+qa0)**2+rhoPPDD) + \
                                                ev4/torch.sqrt((r0+ddb0-qa0)**2+rhoPPDD) + \
                                                ev4/torch.sqrt((r0+ddb0+qa0)**2+rhoPPDD) + \
                                                ev4/torch.sqrt((r0-ddb0-qa0)**2+rhoPPDD) - \
                                                ev3/torch.sqrt((r0-qa0)**2+ddb0**2+rhoPPDD) - \
                                                ev3/torch.sqrt((r0+qa0)**2+ddb0**2+rhoPPDD))- \
                                              0.25*(ev2/torch.sqrt(r0**2+(qa0-ddb0)**2+rhoPPDD) +\
                                                 ev2/torch.sqrt(r0**2+(qa0+ddb0)**2+rhoPPDD) -\
                                                 ev1/torch.sqrt(r0**2+qa0**2+ddb0**2+rhoPPDD))


        DDQtzx_onehalfQtxy_DSQtzx_onehalfQtxy = (ev3/torch.sqrt(r0**2+(dsa0-ddb0)**2+rhoSDDD) + \
                                                ev3/torch.sqrt(r0**2+(dsa0+ddb0)**2+rhoSDDD) - \
                                                ev3/torch.sqrt((r0+ddb0)**2+dsa0**2+rhoSDDD) - \
                                                ev3/torch.sqrt((r0-ddb0)**2+dsa0**2+rhoSDDD) + \
                                                ev4/torch.sqrt((r0-ddb0+dsa0)**2+rhoSDDD) + \
                                                ev4/torch.sqrt((r0+ddb0-dsa0)**2+rhoSDDD) + \
                                                ev4/torch.sqrt((r0+ddb0+dsa0)**2+rhoSDDD) + \
                                                ev4/torch.sqrt((r0-ddb0-dsa0)**2+rhoSDDD) - \
                                                ev3/torch.sqrt((r0-dsa0)**2+ddb0**2+rhoSDDD) - \
                                                ev3/torch.sqrt((r0+dsa0)**2+ddb0**2+rhoSDDD))- \
                                              0.25*(ev2/torch.sqrt(r0**2+(dsa0-ddb0)**2+rhoSDDD) +\
                                                 ev2/torch.sqrt(r0**2+(dsa0+ddb0)**2+rhoSDDD) -\
                                                 ev1/torch.sqrt(r0**2+dsa0**2+ddb0**2+rhoSDDD))

    
        qa0 = qa0/math.sqrt(2)
        qb0 = qb0/math.sqrt(2)
        ddb0 = ddb0/math.sqrt(2)
        dda0 = dda0/math.sqrt(2)
        dsa0 = dsa0/math.sqrt(2)
        dsb0 = dsb0/math.sqrt(2)


        ########    21|21    ########
        dsb0 = dsb0/math.sqrt(2) 
        dsa0 = dsa0/math.sqrt(2)
        PPQpiz_PPQpiz =  ev3/torch.sqrt((r0+qa0-qb0)**2+(qa0-qb0)**2+rhoPPPP) - \
                         ev3/torch.sqrt((r0+qa0-qb0)**2+(qa0+qb0)**2+rhoPPPP) - \
                         ev3/torch.sqrt((r0+qa0+qb0)**2+(qa0-qb0)**2+rhoPPPP) + \
                         ev3/torch.sqrt((r0+qa0+qb0)**2+(qa0+qb0)**2+rhoPPPP) - \
                         ev3/torch.sqrt((r0-qa0-qb0)**2+(qa0-qb0)**2+rhoPPPP) + \
                         ev3/torch.sqrt((r0-qa0-qb0)**2+(qa0+qb0)**2+rhoPPPP) + \
                         ev3/torch.sqrt((r0-qa0+qb0)**2+(qa0-qb0)**2+rhoPPPP) - \
                         ev3/torch.sqrt((r0-qa0+qb0)**2+(qa0+qb0)**2+rhoPPPP)
    
        PPQpiz_DSQpiz  = ev3/torch.sqrt((r0+qa0-dsb0)**2+(qa0-dsb0)**2+rhoSDPP) - \
                         ev3/torch.sqrt((r0+qa0-dsb0)**2+(qa0+dsb0)**2+rhoSDPP) - \
                         ev3/torch.sqrt((r0+qa0+dsb0)**2+(qa0-dsb0)**2+rhoSDPP) + \
                         ev3/torch.sqrt((r0+qa0+dsb0)**2+(qa0+dsb0)**2+rhoSDPP) - \
                         ev3/torch.sqrt((r0-qa0-dsb0)**2+(qa0-dsb0)**2+rhoSDPP) + \
                         ev3/torch.sqrt((r0-qa0-dsb0)**2+(qa0+dsb0)**2+rhoSDPP) + \
                         ev3/torch.sqrt((r0-qa0+dsb0)**2+(qa0-dsb0)**2+rhoSDPP) - \
                         ev3/torch.sqrt((r0-qa0+dsb0)**2+(qa0+dsb0)**2+rhoSDPP)
 
        PPQpiz_DDQpiz  = ev3/torch.sqrt((r0+qa0-ddb0)**2+(qa0-ddb0)**2+rhoDDPP) - \
                         ev3/torch.sqrt((r0+qa0-ddb0)**2+(qa0+ddb0)**2+rhoDDPP) - \
                         ev3/torch.sqrt((r0+qa0+ddb0)**2+(qa0-ddb0)**2+rhoDDPP) + \
                         ev3/torch.sqrt((r0+qa0+ddb0)**2+(qa0+ddb0)**2+rhoDDPP) - \
                         ev3/torch.sqrt((r0-qa0-ddb0)**2+(qa0-ddb0)**2+rhoDDPP) + \
                         ev3/torch.sqrt((r0-qa0-ddb0)**2+(qa0+ddb0)**2+rhoDDPP) + \
                         ev3/torch.sqrt((r0-qa0+ddb0)**2+(qa0-ddb0)**2+rhoDDPP) - \
                         ev3/torch.sqrt((r0-qa0+ddb0)**2+(qa0+ddb0)**2+rhoDDPP)
    
        DSQpiz_DSQpiz  = ev3/torch.sqrt((r0+dsa0-dsb0)**2+(dsa0-dsb0)**2+rhoSDSD) - \
                         ev3/torch.sqrt((r0+dsa0-dsb0)**2+(dsa0+dsb0)**2+rhoSDSD) - \
                         ev3/torch.sqrt((r0+dsa0+dsb0)**2+(dsa0-dsb0)**2+rhoSDSD) + \
                         ev3/torch.sqrt((r0+dsa0+dsb0)**2+(dsa0+dsb0)**2+rhoSDSD) - \
                         ev3/torch.sqrt((r0-dsa0-dsb0)**2+(dsa0-dsb0)**2+rhoSDSD) + \
                         ev3/torch.sqrt((r0-dsa0-dsb0)**2+(dsa0+dsb0)**2+rhoSDSD) + \
                         ev3/torch.sqrt((r0-dsa0+dsb0)**2+(dsa0-dsb0)**2+rhoSDSD) - \
                         ev3/torch.sqrt((r0-dsa0+dsb0)**2+(dsa0+dsb0)**2+rhoSDSD)
    
        DSQpiz_DDQpiz  = ev3/torch.sqrt((r0+dsa0-ddb0)**2+(dsa0-ddb0)**2+rhoDDSD) - \
                         ev3/torch.sqrt((r0+dsa0-ddb0)**2+(dsa0+ddb0)**2+rhoDDSD) - \
                         ev3/torch.sqrt((r0+dsa0+ddb0)**2+(dsa0-ddb0)**2+rhoDDSD) + \
                         ev3/torch.sqrt((r0+dsa0+ddb0)**2+(dsa0+ddb0)**2+rhoDDSD) - \
                         ev3/torch.sqrt((r0-dsa0-ddb0)**2+(dsa0-ddb0)**2+rhoDDSD) + \
                         ev3/torch.sqrt((r0-dsa0-ddb0)**2+(dsa0+ddb0)**2+rhoDDSD) + \
                         ev3/torch.sqrt((r0-dsa0+ddb0)**2+(dsa0-ddb0)**2+rhoDDSD) - \
                         ev3/torch.sqrt((r0-dsa0+ddb0)**2+(dsa0+ddb0)**2+rhoDDSD)
    
        DDQpiz_DDQpiz  = ev3/torch.sqrt((r0+dda0-ddb0)**2+(dda0-ddb0)**2+rhoDDDD) - \
                         ev3/torch.sqrt((r0+dda0-ddb0)**2+(dda0+ddb0)**2+rhoDDDD) - \
                         ev3/torch.sqrt((r0+dda0+ddb0)**2+(dda0-ddb0)**2+rhoDDDD) + \
                         ev3/torch.sqrt((r0+dda0+ddb0)**2+(dda0+ddb0)**2+rhoDDDD) - \
                         ev3/torch.sqrt((r0-dda0-ddb0)**2+(dda0-ddb0)**2+rhoDDDD) + \
                         ev3/torch.sqrt((r0-dda0-ddb0)**2+(dda0+ddb0)**2+rhoDDDD) + \
                         ev3/torch.sqrt((r0-dda0+ddb0)**2+(dda0-ddb0)**2+rhoDDDD) - \
                         ev3/torch.sqrt((r0-dda0+ddb0)**2+(dda0+ddb0)**2+rhoDDDD)
        #############################
        DDQpiz_DSQpiz =  ev3/torch.sqrt((r0+dsb0-dda0)**2+(dsb0-dda0)**2+rhoSDDD) - \
                         ev3/torch.sqrt((r0+dsb0-dda0)**2+(dsb0+dda0)**2+rhoSDDD) - \
                         ev3/torch.sqrt((r0+dsb0+dda0)**2+(dsb0-dda0)**2+rhoSDDD) + \
                         ev3/torch.sqrt((r0+dsb0+dda0)**2+(dsb0+dda0)**2+rhoSDDD) - \
                         ev3/torch.sqrt((r0-dsb0-dda0)**2+(dsb0-dda0)**2+rhoSDDD) + \
                         ev3/torch.sqrt((r0-dsb0-dda0)**2+(dsb0+dda0)**2+rhoSDDD) + \
                         ev3/torch.sqrt((r0-dsb0+dda0)**2+(dsb0-dda0)**2+rhoSDDD) - \
                         ev3/torch.sqrt((r0-dsb0+dda0)**2+(dsb0+dda0)**2+rhoSDDD)
    
        DSQpiz_PPQpiz =  ev3/torch.sqrt((r0+qb0-dsa0)**2+(qb0-dsa0)**2+rhoPPSD) - \
                         ev3/torch.sqrt((r0+qb0-dsa0)**2+(qb0+dsa0)**2+rhoPPSD) - \
                         ev3/torch.sqrt((r0+qb0+dsa0)**2+(qb0-dsa0)**2+rhoPPSD) + \
                         ev3/torch.sqrt((r0+qb0+dsa0)**2+(qb0+dsa0)**2+rhoPPSD) - \
                         ev3/torch.sqrt((r0-qb0-dsa0)**2+(qb0-dsa0)**2+rhoPPSD) + \
                         ev3/torch.sqrt((r0-qb0-dsa0)**2+(qb0+dsa0)**2+rhoPPSD) + \
                         ev3/torch.sqrt((r0-qb0+dsa0)**2+(qb0-dsa0)**2+rhoPPSD) - \
                         ev3/torch.sqrt((r0-qb0+dsa0)**2+(qb0+dsa0)**2+rhoPPSD)

        DDQpiz_PPQpiz =  ev3/torch.sqrt((r0+qb0-dda0)**2+(qb0-dda0)**2+rhoPPDD) - \
                         ev3/torch.sqrt((r0+qb0-dda0)**2+(qb0+dda0)**2+rhoPPDD) - \
                         ev3/torch.sqrt((r0+qb0+dda0)**2+(qb0-dda0)**2+rhoPPDD) + \
                         ev3/torch.sqrt((r0+qb0+dda0)**2+(qb0+dda0)**2+rhoPPDD) - \
                         ev3/torch.sqrt((r0-qb0-dda0)**2+(qb0-dda0)**2+rhoPPDD) + \
                         ev3/torch.sqrt((r0-qb0-dda0)**2+(qb0+dda0)**2+rhoPPDD) + \
                         ev3/torch.sqrt((r0-qb0+dda0)**2+(qb0-dda0)**2+rhoPPDD) - \
                         ev3/torch.sqrt((r0-qb0+dda0)**2+(qb0+dda0)**2+rhoPPDD)
        dsb0 = dsb0*math.sqrt(2)
        dsa0 = dsa0*math.sqrt(2)
    
    
    
        ########    22|22    ########
        qa0 = qa0*math.sqrt(2)
        ddb0 = ddb0*math.sqrt(2)
        qb0 = qb0*math.sqrt(2)
        dda0 = dda0*math.sqrt(2)

        PPQxy_PPQxy = ev2/torch.sqrt((r0)**2+(qa0-qb0)**2+rhoPPPP) + \
                      ev2/torch.sqrt((r0)**2+(qa0+qb0)**2+rhoPPPP) - \
                      ev1/torch.sqrt((r0)**2+(qa0)**2+(qb0)**2+rhoPPPP)
    
        PPQxy_DSQxy = ev2/torch.sqrt((r0)**2+(qa0-dsb0)**2+rhoSDPP) + \
                      ev2/torch.sqrt((r0)**2+(qa0+dsb0)**2+rhoSDPP) - \
                      ev1/torch.sqrt((r0)**2+(qa0)**2+(dsb0)**2+rhoSDPP)

        PPQxy_DDQxy = ev2/torch.sqrt((r0)**2+(qa0-ddb0)**2+rhoDDPP) + \
                      ev2/torch.sqrt((r0)**2+(qa0+ddb0)**2+rhoDDPP) - \
                      ev1/torch.sqrt((r0)**2+(qa0)**2+(ddb0)**2+rhoDDPP)

        DSQxy_DSQxy = ev2/torch.sqrt((r0)**2+(dsa0-dsb0)**2+rhoSDSD) + \
                      ev2/torch.sqrt((r0)**2+(dsa0+dsb0)**2+rhoSDSD) - \
                      ev1/torch.sqrt((r0)**2+(dsa0)**2+(dsb0)**2+rhoSDSD)
    
        DSQxy_DDQxy = ev2/torch.sqrt((r0)**2+(dsa0-ddb0)**2+rhoDDSD) + \
                      ev2/torch.sqrt((r0)**2+(dsa0+ddb0)**2+rhoDDSD) - \
                      ev1/torch.sqrt((r0)**2+(dsa0)**2+(ddb0)**2+rhoDDSD)
    
        DDQxy_DDQxy = ev2/torch.sqrt((r0)**2+(dda0-ddb0)**2+rhoDDDD) + \
                      ev2/torch.sqrt((r0)**2+(dda0+ddb0)**2+rhoDDDD) - \
                      ev1/torch.sqrt((r0)**2+(dda0)**2+(ddb0)**2+rhoDDDD)
        #############################
        DSQxy_PPQxy = ev2/torch.sqrt((r0)**2+(qb0-dsa0)**2+rhoPPSD) + \
                      ev2/torch.sqrt((r0)**2+(qb0+dsa0)**2+rhoPPSD) - \
                      ev1/torch.sqrt((r0)**2+(qb0)**2+(dsa0)**2+rhoPPSD)
    
        DDQxy_PPQxy = ev2/torch.sqrt((r0)**2+(qb0-dda0)**2+rhoPPDD) + \
                      ev2/torch.sqrt((r0)**2+(qb0+dda0)**2+rhoPPDD) - \
                      ev1/torch.sqrt((r0)**2+(qb0)**2+(dda0)**2+rhoPPDD)
    
        DDQxy_DSQxy = ev2/torch.sqrt((r0)**2+(dsb0-dda0)**2+rhoSDDD) + \
                      ev2/torch.sqrt((r0)**2+(dsb0+dda0)**2+rhoSDDD) - \
                      ev1/torch.sqrt((r0)**2+(dsb0)**2+(dda0)**2+rhoSDDD)
        qa0 = qa0/math.sqrt(2)
        ddb0 = ddb0/math.sqrt(2)
        qb0 = qb0/math.sqrt(2)
        dda0 = dda0/math.sqrt(2)



        ########    22|2-2    ########
        qa0 = qa0*math.sqrt(2)
        qb0 = qb0*math.sqrt(2)
        ddb0 = ddb0*math.sqrt(2)
        dda0 = dda0*math.sqrt(2)
        tildaPPQxy_PPQxy = ev2/torch.sqrt((r0)**2+(qa0-qb0)**2+rhoPPPP) + \
                      ev2/torch.sqrt((r0)**2+(qa0+qb0)**2+rhoPPPP) - \
                      ev1/torch.sqrt((r0)**2+(qa0)**2+(qb0)**2+rhoPPPP)

        tildaPPQxy_DSQxy = ev2/torch.sqrt((r0)**2+(qa0-dsb0)**2+rhoSDPP) + \
                      ev2/torch.sqrt((r0)**2+(qa0+dsb0)**2+rhoSDPP) - \
                      ev1/torch.sqrt((r0)**2+(qa0)**2+(dsb0)**2+rhoSDPP)
        tildaPPQxy_DDQxy = ev2/torch.sqrt((r0)**2+(qa0-ddb0)**2+rhoDDPP) + \
                      ev2/torch.sqrt((r0)**2+(qa0+ddb0)**2+rhoDDPP) - \
                      ev1/torch.sqrt((r0)**2+(qa0)**2+(ddb0)**2+rhoDDPP)

        tildaDSQxy_DSQxy = ev2/torch.sqrt((r0)**2+(dsa0-dsb0)**2+rhoSDSD) + \
                      ev2/torch.sqrt((r0)**2+(dsa0+dsb0)**2+rhoSDSD) - \
                      ev1/torch.sqrt((r0)**2+(dsa0)**2+(dsb0)**2+rhoSDSD)

        tildaDSQxy_DDQxy = ev2/torch.sqrt((r0)**2+(dsa0-ddb0)**2+rhoDDSD) + \
                      ev2/torch.sqrt((r0)**2+(dsa0+ddb0)**2+rhoDDSD) - \
                      ev1/torch.sqrt((r0)**2+(dsa0)**2+(ddb0)**2+rhoDDSD)

        tildaDDQxy_DDQxy = ev2/torch.sqrt((r0)**2+(dda0-ddb0)**2+rhoDDDD) + \
                      ev2/torch.sqrt((r0)**2+(dda0+ddb0)**2+rhoDDDD) - \
                      ev1/torch.sqrt((r0)**2+(dda0)**2+(ddb0)**2+rhoDDDD)
        #############################
        tildaDSQxy_PPQxy = ev2/torch.sqrt((r0)**2+(qb0-dsa0)**2+rhoPPSD) + \
                      ev2/torch.sqrt((r0)**2+(qb0+dsa0)**2+rhoPPSD) - \
                      ev1/torch.sqrt((r0)**2+(qb0)**2+(dsa0)**2+rhoPPSD)

        tildaDDQxy_PPQxy = ev2/torch.sqrt((r0)**2+(qb0-dda0)**2+rhoPPDD) + \
                      ev2/torch.sqrt((r0)**2+(qb0+dda0)**2+rhoPPDD) - \
                      ev1/torch.sqrt((r0)**2+(qb0)**2+(dda0)**2+rhoPPDD)

        tildaDDQxy_DSQxy = ev2/torch.sqrt((r0)**2+(dsb0-dda0)**2+rhoSDDD) + \
                      ev2/torch.sqrt((r0)**2+(dsb0+dda0)**2+rhoSDDD) - \
                      ev1/torch.sqrt((r0)**2+(dsb0)**2+(dda0)**2+rhoSDDD)

        qa0 = qa0/math.sqrt(2)
        qb0 = qb0/math.sqrt(2)
        ddb0 = ddb0/math.sqrt(2)
        dda0 = dda0/math.sqrt(2)



        #####SS|DxyDxy#####
        riYY[...,1980]=Sq_DDq*1.000000*1.000000+Sq_DDQtzx_onehalfQtxy*1.000000*-1.333333
        
        
        #####SS|PxDxz#####
        riYY[...,765]=Sq_DPUz*1.000000*1.000000
        
        
        #####SS|DxzDxz#####
        riYY[...,900]=Sq_DDq*1.000000*1.000000+Sq_DDQtzx_onehalfQtxy*1.000000*0.666667
        
        #####SS|SDz2#####
        riYY[...,450]=Sq_DSQtzx_onehalfQtxy*1.000000*1.154701
        
        
        #####SS|PzDz2#####
        riYY[...,495]=Sq_DPUz*1.000000*1.154701
        
        
        #####SS|Dz2Dz2#####
        riYY[...,630]=Sq_DDq*1.000000*1.000000+Sq_DDQtzx_onehalfQtxy*1.000000*1.333333
        
        
        #####SS|PyDyz#####
        riYY[...,1080]=Sq_DPUz*1.000000*1.000000
        
        
        #####SS|DyzDyz#####
        riYY[...,1215]=Sq_DDq*1.000000*1.000000+Sq_DDQtzx_onehalfQtxy*1.000000*0.666667
        
        
        #####SS|Dx2-y2Dx2-y2#####
        riYY[...,1575]=Sq_DDq*1.000000*1.000000+Sq_DDQtzx_onehalfQtxy*1.000000*-1.333333
        
        
        #####SPx|PyDxy#####
        riYY[...,1758]=SPUpi_DPUpi*1.000000*1.000000
        
        
        
        #####SPx|SDxz#####
        riYY[...,678]=SPUpi_DSQpiz*1.000000*1.000000
        
        
        #####SPx|PzDxz#####
        riYY[...,723]=SPUpi_DPUpi*1.000000*1.000000
        
        
        #####SPx|PxDz2#####
        riYY[...,543]=SPUpi_DPUpi*1.000000*-0.577350
        
        #####SPx|Dz2Dxz#####
        riYY[...,858]=SPUpi_DDQpiz*1.000000*0.577350

        #####SPx|DyzDxy#####
        riYY[...,1893]=SPUpi_DDQpiz*1.000000*1.000000
        
        
        #####SPx|PxDx2-y2#####
        riYY[...,1353]=SPUpi_DPUpi*1.000000*1.000000
        
        
        #####SPx|DxzDx2-y2#####
        riYY[...,1488]=SPUpi_DDQpiz*1.000000*1.000000
        
        
        #####PxPx|DxyDxy#####
        riYY[...,1985]=Sq_DDq*1.000000*1.000000+Sq_DDQtzx_onehalfQtxy*1.000000*-1.333333+PPQtzx_onehalfQtxy_DDq*-0.666667*1.000000+PPQtzx_onehalfQtxy_DDQtzx_onehalfQtxy*-0.666667*-1.333333
        
        
        #####PxPx|PxDxz#####
        riYY[...,770]=Sq_DPUz*1.000000*1.000000+PPQtzx_onehalfQtxy_DPUz*-0.666667*1.000000
        
        
        #####PxPx|DxzDxz#####
        riYY[...,905]=Sq_DDq*1.000000*1.000000+Sq_DDQtzx_onehalfQtxy*1.000000*0.666667+PPQtzx_onehalfQtxy_DDq*-0.666667*1.000000+PPQtzx_onehalfQtxy_DDQtzx_onehalfQtxy*-0.666667*0.666667+PPQxy_DDQxy*1.000000*1.000000
        #####PxPx|SDz2#####
        riYY[...,455]=Sq_DSQtzx_onehalfQtxy*1.000000*1.154701+PPQtzx_onehalfQtxy_DSQtzx_onehalfQtxy*-0.666667*1.154701
        
        #####PxPx|PzDz2#####
        riYY[...,500]=Sq_DPUz*1.000000*1.154701+PPQtzx_onehalfQtxy_DPUz*-0.666667*1.154701
        
        
        #####PxPx|Dz2Dz2#####
        riYY[...,635]=Sq_DDq*1.000000*1.000000+Sq_DDQtzx_onehalfQtxy*1.000000*1.333333+PPQtzx_onehalfQtxy_DDq*-0.666667*1.000000+PPQtzx_onehalfQtxy_DDQtzx_onehalfQtxy*-0.666667*1.333333
        
        #####PxPx|PyDyz#####
        riYY[...,1085]=Sq_DPUz*1.000000*1.000000+PPQtzx_onehalfQtxy_DPUz*-0.666667*1.000000
        
        #####PxPx|DyzDyz#####
        riYY[...,1220]=Sq_DDq*1.000000*1.000000+Sq_DDQtzx_onehalfQtxy*1.000000*0.666667+PPQtzx_onehalfQtxy_DDq*-0.666667*1.000000+PPQtzx_onehalfQtxy_DDQtzx_onehalfQtxy*-0.666667*0.666667-PPQxy_DDQxy*1.000000*1.000000
        
        #####PxPx|SDx2-y2#####
        riYY[...,1265]=PPQxy_DSQxy*1.000000*1.000000
        
        
        #####PxPx|Dz2Dx2-y2#####
        riYY[...,1445]=PPQxy_DDQxy*1.000000*-1.154701
        
        
        #####PxPx|Dx2-y2Dx2-y2#####
        riYY[...,1580]=Sq_DDq*1.000000*1.000000+Sq_DDQtzx_onehalfQtxy*1.000000*-1.333333+PPQtzx_onehalfQtxy_DDq*-0.666667*1.000000+PPQtzx_onehalfQtxy_DDQtzx_onehalfQtxy*-0.666667*-1.333333
        
        
        #####SPy|PxDxy#####
        riYY[...,1716]=SPUpi_DPUpi*1.000000*1.000000
        
        
        #####SPy|DxzDxy#####
        riYY[...,1851]=SPUpi_DDQpiz*1.000000*1.000000
        
        
        #####SPy|PyDz2#####
        riYY[...,591]=SPUpi_DPUpi*1.000000*1.000000*-0.57735026918962995
        
        
        #####SPy|SDyz#####
        riYY[...,951]=SPUpi_DSQpiz*1.000000*1.000000
        
        
        #####SPy|PzDyz#####
        riYY[...,996]=SPUpi_DPUpi*1.000000*1.000000
        
        
        #####SPy|Dz2Dyz#####
        riYY[...,1131]=SPUpi_DDQpiz*1.000000*1.000000*0.57735026918962995
        
        
        #####SPy|PyDx2-y2#####
        riYY[...,1401]=-SPUpi_DPUpi*1.000000*1.000000
        
        #####SPy|DyzDx2-y2#####
        riYY[...,1536]=-SPUpi_DDQpiz*1.000000*1.000000
        
        #####PxPy|SDxy#####
        riYY[...,1628]=tildaPPQxy_DSQxy*1.000000*1.000000
        
        #####PxPy|Dz2Dxy#####
        riYY[...,1808]=tildaPPQxy_DDQxy*1.000000*1.000000*-1.1547005383793001
        
        #####PxPy|DxzDyz#####
        riYY[...,1178]=tildaPPQxy_DDQxy*1.000000*1.000000
        
        #####PyPy|DxyDxy#####
        riYY[...,1989]=Sq_DDq*1.000000*1.000000+Sq_DDQtzx_onehalfQtxy*1.000000*-1.333333+PPQtzx_onehalfQtxy_DDq*-0.666667*1.000000+PPQtzx_onehalfQtxy_DDQtzx_onehalfQtxy*-0.666667*-1.333333
        
        
        #####PyPy|PxDxz#####
        riYY[...,774]=Sq_DPUz*1.000000*1.000000+PPQtzx_onehalfQtxy_DPUz*-0.666667*1.000000
        
        
        #####PyPy|DxzDxz#####
        riYY[...,909]=Sq_DDq*1.000000*1.000000+Sq_DDQtzx_onehalfQtxy*1.000000*0.666667+PPQtzx_onehalfQtxy_DDq*-0.666667*1.000000+PPQtzx_onehalfQtxy_DDQtzx_onehalfQtxy*-0.666667*0.666667-PPQxy_DDQxy*1.000000*1.000000
        
        #####PyPy|SDz2#####
        riYY[...,459]=Sq_DSQtzx_onehalfQtxy*1.000000*1.154701+PPQtzx_onehalfQtxy_DSQtzx_onehalfQtxy*-0.666667*1.154701
        
        #####PyPy|PzDz2#####
        riYY[...,504]=Sq_DPUz*1.000000*1.154701+PPQtzx_onehalfQtxy_DPUz*-0.666667*1.154701
        
        
        #####PyPy|Dz2Dz2#####
        riYY[...,639]=Sq_DDq*1.000000*1.000000+Sq_DDQtzx_onehalfQtxy*1.000000*1.333333+PPQtzx_onehalfQtxy_DDq*-0.666667*1.000000+PPQtzx_onehalfQtxy_DDQtzx_onehalfQtxy*-0.666667*1.333333
        
        
        #####PyPy|PyDyz#####
        riYY[...,1089]=Sq_DPUz*1.000000*1.000000+PPQtzx_onehalfQtxy_DPUz*-0.666667*1.000000
        
        
        #####PyPy|DyzDyz#####
        riYY[...,1224]=Sq_DDq*1.000000*1.000000+Sq_DDQtzx_onehalfQtxy*1.000000*0.666667+PPQtzx_onehalfQtxy_DDq*-0.666667*1.000000+PPQtzx_onehalfQtxy_DDQtzx_onehalfQtxy*-0.666667*0.666667+PPQxy_DDQxy*1.000000*1.000000
        
        #####PyPy|SDx2-y2#####
        riYY[...,1269]=-PPQxy_DSQxy*1.000000*1.000000
        
        
        #####PyPy|Dz2Dx2-y2#####
        riYY[...,1449]=-PPQxy_DDQxy*1.000000*-1.154701
        
        
        #####PyPy|Dx2-y2Dx2-y2#####
        riYY[...,1584]=Sq_DDq*1.000000*1.000000+Sq_DDQtzx_onehalfQtxy*1.000000*-1.333333+PPQtzx_onehalfQtxy_DDq*-0.666667*1.000000+PPQtzx_onehalfQtxy_DDQtzx_onehalfQtxy*-0.666667*-1.333333
        
        
        #####SPz|DxyDxy#####
        riYY[...,1981]=SPUz_DDq*1.000000*1.000000+SPUz_DDQtzx_onehalfQtxy*1.000000*-1.333333
        
        
        #####SPz|PxDxz#####
        riYY[...,766]=SPUz_DPUz*1.000000*1.000000
        
        
        #####SPz|DxzDxz#####
        riYY[...,901]=SPUz_DDq*1.000000*1.000000+SPUz_DDQtzx_onehalfQtxy*1.000000*0.666667
        
        #####SPz|SDz2#####
        riYY[...,451]=SPUz_DSQtzx_onehalfQtxy*1.000000*1.154701
        
        
        #####SPz|PzDz2#####
        riYY[...,496]=SPUz_DPUz*1.000000*1.154701
        
        
        #####SPz|Dz2Dz2#####
        riYY[...,631]=SPUz_DDq*1.000000*1.000000+SPUz_DDQtzx_onehalfQtxy*1.000000*1.333333
        
        
        #####SPz|PyDyz#####
        riYY[...,1081]=SPUz_DPUz*1.000000*1.000000
        
        
        #####SPz|DyzDyz#####
        riYY[...,1216]=SPUz_DDq*1.000000*1.000000+SPUz_DDQtzx_onehalfQtxy*1.000000*0.666667
        
        
        #####SPz|Dx2-y2Dx2-y2#####
        riYY[...,1576]=SPUz_DDq*1.000000*1.000000+SPUz_DDQtzx_onehalfQtxy*1.000000*-1.333333
        
        
        #####PzPx|PyDxy#####
        riYY[...,1759]=PPQpiz_DPUpi*1.000000*1.000000
        
        
        #####PzPx|SDxz#####
        riYY[...,679]=PPQpiz_DSQpiz*1.000000*1.000000
        
        #####PzPx|PzDxz#####
        riYY[...,724]=PPQpiz_DPUpi*1.000000*1.000000
        
        
        #####PzPx|PxDz2#####
        riYY[...,544]=PPQpiz_DPUpi*1.000000*-0.577350
        
        
        #####PzPx|Dz2Dxz#####
        riYY[...,859]=PPQpiz_DDQpiz*1.000000*0.577350
        
        
        #####PzPx|DyzDxy#####
        riYY[...,1894]=PPQpiz_DDQpiz*1.000000*1.000000
        
        
        #####PzPx|PxDx2-y2#####
        riYY[...,1354]=PPQpiz_DPUpi*1.000000*1.000000
        
        
        #####PzPx|DxzDx2-y2#####
        riYY[...,1489]=PPQpiz_DDQpiz*1.000000*1.000000
        
        
        #####PzPy|PxDxy#####
        riYY[...,1717]=PPQpiz_DPUpi*1.000000*1.000000
        
        
        #####PzPy|DxzDxy#####
        riYY[...,1852]=PPQpiz_DDQpiz*1.000000*1.000000
        
        
        #####PzPy|PyDz2#####
        riYY[...,592]=-PPQpiz_DPUpi*1.000000*1.000000*0.57735026918962995
        
        
        #####PzPy|SDyz#####
        riYY[...,952]=PPQpiz_DSQpiz*1.000000*1.000000
        
        
        #####PzPy|PzDyz#####
        riYY[...,997]=PPQpiz_DPUpi*1.000000*1.000000
        
        
        #####PzPy|Dz2Dyz#####
        riYY[...,1132]=PPQpiz_DDQpiz*1.000000*1.000000*0.57735026918962995
        
        
        #####PzPy|PyDx2-y2#####
        riYY[...,1402]=-PPQpiz_DPUpi*1.000000*1.000000
        
        
        #####PzPy|DyzDx2-y2#####
        riYY[...,1537]=-PPQpiz_DDQpiz*1.000000*1.000000
        
        
        #####PzPz|DxyDxy#####
        riYY[...,1982]=Sq_DDq*1.000000*1.000000+Sq_DDQtzx_onehalfQtxy*1.000000*-1.333333+PPQtzx_onehalfQtxy_DDq*1.333333*1.000000+PPQtzx_onehalfQtxy_DDQtzx_onehalfQtxy*1.333333*-1.333333
        
        
        #####PzPz|PxDxz#####
        riYY[...,767]=Sq_DPUz*1.000000*1.000000+PPQtzx_onehalfQtxy_DPUz*1.333333*1.000000
        
        
        #####PzPz|DxzDxz#####
        riYY[...,902]=Sq_DDq*1.000000*1.000000+Sq_DDQtzx_onehalfQtxy*1.000000*0.666667+PPQtzx_onehalfQtxy_DDq*1.333333*1.000000+PPQtzx_onehalfQtxy_DDQtzx_onehalfQtxy*1.333333*0.666667
        
        #####PzPz|SDz2#####
        riYY[...,452]=Sq_DSQtzx_onehalfQtxy*1.000000*1.154701+PPQtzx_onehalfQtxy_DSQtzx_onehalfQtxy*1.333333*1.154701
        
        #####PzPz|PzDz2#####
        riYY[...,497]=Sq_DPUz*1.000000*1.154701+PPQtzx_onehalfQtxy_DPUz*1.333333*1.154701
        
        
        #####PzPz|Dz2Dz2#####
        riYY[...,632]=Sq_DDq*1.000000*1.000000+Sq_DDQtzx_onehalfQtxy*1.000000*1.333333+PPQtzx_onehalfQtxy_DDq*1.333333*1.000000+PPQtzx_onehalfQtxy_DDQtzx_onehalfQtxy*1.333333*1.333333
        
        #####PzPz|PyDyz#####
        riYY[...,1082]=Sq_DPUz*1.000000*1.000000+PPQtzx_onehalfQtxy_DPUz*1.333333*1.000000
        
        
        #####PzPz|DyzDyz#####
        riYY[...,1217]=Sq_DDq*1.000000*1.000000+Sq_DDQtzx_onehalfQtxy*1.000000*0.666667+PPQtzx_onehalfQtxy_DDq*1.333333*1.000000+PPQtzx_onehalfQtxy_DDQtzx_onehalfQtxy*1.333333*0.666667
        
        
        #####PzPz|Dx2-y2Dx2-y2#####
        riYY[...,1577]=Sq_DDq*1.000000*1.000000+Sq_DDQtzx_onehalfQtxy*1.000000*-1.333333+PPQtzx_onehalfQtxy_DDq*1.333333*1.000000+PPQtzx_onehalfQtxy_DDQtzx_onehalfQtxy*1.333333*-1.333333
        
        
        #####SDxy|PxPy#####
        riYY[...,396]=tildaDSQxy_PPQxy*1.000000*1.000000
        
        
        #####SDxy|SDxy#####
        riYY[...,1656]=tildaDSQxy_DSQxy*1.000000*1.000000
        
        
        #####SDxy|Dz2Dxy#####
        riYY[...,1836]=tildaDSQxy_DDQxy*1.000000*1.000000*-1.1547005383793001
        
        
        #####SDxy|DxzDyz#####
        riYY[...,1206]=tildaDSQxy_DDQxy*1.000000*1.000000
        
        
        #####PxDxy|SPy#####
        riYY[...,308]=DPUpi_SPUpi*1.000000*1.000000
        
        #####PxDxy|PzPy#####
        riYY[...,353]=DPUpi_PPQpiz*1.000000*1.000000
        
        
        #####PxDxy|PxDxy#####
        riYY[...,1748]=DPUpi_DPUpi*1.000000*1.000000
        
        
        #####PxDxy|DxzDxy#####
        riYY[...,1883]=DPUpi_DDQpiz*1.000000*1.000000
        
        
        #####PxDxy|PyDz2#####
        riYY[...,623]=DPUpi_DPUpi*1.000000*1.000000*-0.57735026918962995
        
        
        #####PxDxy|SDyz#####
        riYY[...,983]=DPUpi_DSQpiz*1.000000*1.000000
        
        
        #####PxDxy|PzDyz#####
        riYY[...,1028]=DPUpi_DPUpi*1.000000*1.000000
        
        
        #####PxDxy|Dz2Dyz#####
        riYY[...,1163]=DPUpi_DDQpiz*1.000000*1.000000*0.57735026918962995
        
        
        #####PxDxy|PyDx2-y2#####
        riYY[...,1433]=-DPUpi_DPUpi*1.000000*1.000000
        
        
        #####PxDxy|DyzDx2-y2#####
        riYY[...,1568]=-DPUpi_DDQpiz*1.000000*1.000000
        
        
        #####PyDxy|SPx#####
        riYY[...,174]=DPUpi_SPUpi*1.000000*1.000000
        
        
        #####PyDxy|PzPx#####
        riYY[...,219]=DPUpi_PPQpiz*1.000000*1.000000
        
        
        #####PyDxy|PyDxy#####
        riYY[...,1794]=DPUpi_DPUpi*1.000000*1.000000
        
        
        #####PyDxy|SDxz#####
        riYY[...,714]=DPUpi_DSQpiz*1.000000*1.000000
        
        
        #####PyDxy|PzDxz#####
        riYY[...,759]=DPUpi_DPUpi*1.000000*1.000000
        
        
        #####PyDxy|PxDz2#####
        riYY[...,579]=DPUpi_DPUpi*1.000000*-0.577350
        
        
        #####PyDxy|Dz2Dxz#####
        riYY[...,894]=DPUpi_DDQpiz*1.000000*0.577350
        
        
        #####PyDxy|DyzDxy#####
        riYY[...,1929]=DPUpi_DDQpiz*1.000000*1.000000
        
        
        #####PyDxy|PxDx2-y2#####
        riYY[...,1389]=DPUpi_DPUpi*1.000000*1.000000
        
        
        #####PyDxy|DxzDx2-y2#####
        riYY[...,1524]=DPUpi_DDQpiz*1.000000*1.000000
        
        
        #####DxyDxy|SS#####
        riYY[...,44]=DDq_Sq*1.000000*1.000000+DDQtzx_onehalfQtxy_Sq*-1.333333*1.000000
        
        
        #####DxyDxy|PxPx#####
        riYY[...,269]=DDq_Sq*1.000000*1.000000+DDq_PPQtzx_onehalfQtxy*1.000000*-0.666667+DDQtzx_onehalfQtxy_Sq*-1.333333*1.000000+DDQtzx_onehalfQtxy_PPQtzx_onehalfQtxy*-1.333333*-0.666667
        
        
        #####DxyDxy|PyPy#####
        riYY[...,449]=DDq_Sq*1.000000*1.000000+DDq_PPQtzx_onehalfQtxy*1.000000*-0.666667+DDQtzx_onehalfQtxy_Sq*-1.333333*1.000000+DDQtzx_onehalfQtxy_PPQtzx_onehalfQtxy*-1.333333*-0.666667
        
        #####DxyDxy|SPz#####
        riYY[...,89]=DDq_SPUz*1.000000*1.000000+DDQtzx_onehalfQtxy_SPUz*-1.333333*1.000000
        
        
        #####DxyDxy|PzPz#####
        riYY[...,134]=DDq_Sq*1.000000*1.000000+DDq_PPQtzx_onehalfQtxy*1.000000*1.333333+DDQtzx_onehalfQtxy_Sq*-1.333333*1.000000+DDQtzx_onehalfQtxy_PPQtzx_onehalfQtxy*-1.333333*1.333333
        
        
        #####DxyDxy|DxyDxy#####
        riYY[...,2024]=DDq_DDq*1.000000*1.000000+DDq_DDQtzx_onehalfQtxy*1.000000*-1.333333+DDQtzx_onehalfQtxy_DDq*-1.333333*1.000000+DDQtzx_onehalfQtxy_DDQtzx_onehalfQtxy*-1.333333*-1.333333
        
        
        #####DxyDxy|PxDxz#####
        riYY[...,809]=DDq_DPUz*1.000000*1.000000+DDQtzx_onehalfQtxy_DPUz*-1.333333*1.000000
        
        
        #####DxyDxy|DxzDxz#####
        riYY[...,944]=DDq_DDq*1.000000*1.000000+DDq_DDQtzx_onehalfQtxy*1.000000*0.666667+DDQtzx_onehalfQtxy_DDq*-1.333333*1.000000+DDQtzx_onehalfQtxy_DDQtzx_onehalfQtxy*-1.333333*0.666667
        
        #####DxyDxy|SDz2#####
        riYY[...,494]=DDq_DSQtzx_onehalfQtxy*1.000000*1.154701+DDQtzx_onehalfQtxy_DSQtzx_onehalfQtxy*-1.333333*1.154701
        
        
        #####DxyDxy|PzDz2#####
        riYY[...,539]=DDq_DPUz*1.000000*1.154701+DDQtzx_onehalfQtxy_DPUz*-1.333333*1.154701
        
        
        #####DxyDxy|Dz2Dz2#####
        riYY[...,674]=DDq_DDq*1.000000*1.000000+DDq_DDQtzx_onehalfQtxy*1.000000*1.333333+DDQtzx_onehalfQtxy_DDq*-1.333333*1.000000+DDQtzx_onehalfQtxy_DDQtzx_onehalfQtxy*-1.333333*1.333333
        
        
        #####DxyDxy|PyDyz#####
        riYY[...,1124]=DDq_DPUz*1.000000*1.000000+DDQtzx_onehalfQtxy_DPUz*-1.333333*1.000000
        
        
        #####DxyDxy|DyzDyz#####
        riYY[...,1259]=DDq_DDq*1.000000*1.000000+DDq_DDQtzx_onehalfQtxy*1.000000*0.666667+DDQtzx_onehalfQtxy_DDq*-1.333333*1.000000+DDQtzx_onehalfQtxy_DDQtzx_onehalfQtxy*-1.333333*0.666667
        
        
        #####DxyDxy|Dx2-y2Dx2-y2#####
        riYY[...,1619]=DDq_DDq*1.000000*1.000000+DDq_DDQtzx_onehalfQtxy*1.000000*-1.333333+DDQtzx_onehalfQtxy_DDq*-1.333333*1.000000+DDQtzx_onehalfQtxy_DDQtzx_onehalfQtxy*-1.333333*-1.333333
        
        
        #####SDxz|SPx#####
        riYY[...,150]=DSQpiz_SPUpi*1.000000*1.000000
        
        
        #####SDxz|PzPx#####
        riYY[...,195]=DSQpiz_PPQpiz*1.000000*1.000000
        
        
        #####SDxz|PyDxy#####
        riYY[...,1770]=DSQpiz_DPUpi*1.000000*1.000000
        
        
        #####SDxz|SDxz#####
        riYY[...,690]=DSQpiz_DSQpiz*1.000000*1.000000
        
        
        #####SDxz|PzDxz#####
        riYY[...,735]=DSQpiz_DPUpi*1.000000*1.000000
        
        
        #####SDxz|PxDz2#####
        riYY[...,555]=DSQpiz_DPUpi*1.000000*-0.577350
        
        
        #####SDxz|Dz2Dxz#####
        riYY[...,870]=DSQpiz_DDQpiz*1.000000*0.577350
        
        
        #####SDxz|DyzDxy#####
        riYY[...,1905]=DSQpiz_DDQpiz*1.000000*1.000000
        
        
        #####SDxz|PxDx2-y2#####
        riYY[...,1365]=DSQpiz_DPUpi*1.000000*1.000000
        
        
        #####SDxz|DxzDx2-y2#####
        riYY[...,1500]=DSQpiz_DDQpiz*1.000000*1.000000
        
        
        #####PxDxz|SS#####
        riYY[...,17]=DPUz_Sq*1.000000*1.000000
        
        
        #####PxDxz|PxPx#####
        riYY[...,242]=DPUz_Sq*1.000000*1.000000+DPUz_PPQtzx_onehalfQtxy*1.000000*-0.666667
        
        
        #####PxDxz|PyPy#####
        riYY[...,422]=DPUz_Sq*1.000000*1.000000+DPUz_PPQtzx_onehalfQtxy*1.000000*-0.666667
        
        #####PxDxz|SPz#####
        riYY[...,62]=DPUz_SPUz*1.000000*1.000000
        
        
        #####PxDxz|PzPz#####
        riYY[...,107]=DPUz_Sq*1.000000*1.000000+DPUz_PPQtzx_onehalfQtxy*1.000000*1.333333
        
        
        #####PxDxz|DxyDxy#####
        riYY[...,1997]=DPUz_DDq*1.000000*1.000000+DPUz_DDQtzx_onehalfQtxy*1.000000*-1.333333
        
        
        #####PxDxz|PxDxz#####
        riYY[...,782]=DPUz_DPUz*1.000000*1.000000
        
        
        #####PxDxz|DxzDxz#####
        riYY[...,917]=DPUz_DDq*1.000000*1.000000+DPUz_DDQtzx_onehalfQtxy*1.000000*0.666667
        
        #####PxDxz|SDz2#####
        riYY[...,467]=DPUz_DSQtzx_onehalfQtxy*1.000000*1.154701
        
        
        #####PxDxz|PzDz2#####
        riYY[...,512]=DPUz_DPUz*1.000000*1.154701
        
        
        #####PxDxz|Dz2Dz2#####
        riYY[...,647]=DPUz_DDq*1.000000*1.000000+DPUz_DDQtzx_onehalfQtxy*1.000000*1.333333
        
        
        #####PxDxz|PyDyz#####
        riYY[...,1097]=DPUz_DPUz*1.000000*1.000000
        
        
        #####PxDxz|DyzDyz#####
        riYY[...,1232]=DPUz_DDq*1.000000*1.000000+DPUz_DDQtzx_onehalfQtxy*1.000000*0.666667
        
        
        #####PxDxz|Dx2-y2Dx2-y2#####
        riYY[...,1592]=DPUz_DDq*1.000000*1.000000+DPUz_DDQtzx_onehalfQtxy*1.000000*-1.333333
        
        
        #####PzDxz|SPx#####
        riYY[...,151]=DPUpi_SPUpi*1.000000*1.000000
        
        
        #####PzDxz|PzPx#####
        riYY[...,196]=DPUpi_PPQpiz*1.000000*1.000000
        
        
        #####PzDxz|PyDxy#####
        riYY[...,1771]=DPUpi_DPUpi*1.000000*1.000000
        
        
        #####PzDxz|SDxz#####
        riYY[...,691]=DPUpi_DSQpiz*1.000000*1.000000
        
        
        #####PzDxz|PzDxz#####
        riYY[...,736]=DPUpi_DPUpi*1.000000*1.000000
        
        
        #####PzDxz|PxDz2#####
        riYY[...,556]=DPUpi_DPUpi*1.000000*-0.577350
        
        
        #####PzDxz|Dz2Dxz#####
        riYY[...,871]=DPUpi_DDQpiz*1.000000*0.577350
        
        
        #####PzDxz|DyzDxy#####
        riYY[...,1906]=DPUpi_DDQpiz*1.000000*1.000000
        
        
        #####PzDxz|PxDx2-y2#####
        riYY[...,1366]=DPUpi_DPUpi*1.000000*1.000000
        
        
        #####PzDxz|DxzDx2-y2#####
        riYY[...,1501]=DPUpi_DDQpiz*1.000000*1.000000
        
        
        #####DxzDxy|SPy#####
        riYY[...,311]=DDQpiz_SPUpi*1.000000*1.000000
        
        
        #####DxzDxy|PzPy#####
        riYY[...,356]=DDQpiz_PPQpiz*1.000000*1.000000
        
        
        #####DxzDxy|PxDxy#####
        riYY[...,1751]=DDQpiz_DPUpi*1.000000*1.000000
        
        
        #####DxzDxy|DxzDxy#####
        riYY[...,1886]=DDQpiz_DDQpiz*1.000000*1.000000
        
        
        #####DxzDxy|PyDz2#####
        riYY[...,626]=DDQpiz_DPUpi*1.000000*1.000000*-0.57735026918962995
        
        
        #####DxzDxy|SDyz#####
        riYY[...,986]=DDQpiz_DSQpiz*1.000000*1.000000
        
        
        #####DxzDxy|PzDyz#####
        riYY[...,1031]=DDQpiz_DPUpi*1.000000*1.000000
        
        
        #####DxzDxy|Dz2Dyz#####
        riYY[...,1166]=DDQpiz_DDQpiz*1.000000*1.000000*0.57735026918962995
        
        
        #####DxzDxy|PyDx2-y2#####
        riYY[...,1436]=-DDQpiz_DPUpi*1.000000*1.000000
        
        
        #####DxzDxy|DyzDx2-y2#####
        riYY[...,1571]=-DDQpiz_DDQpiz*1.000000*1.000000
        
        
        #####DxzDxz|SS#####
        riYY[...,20]=DDq_Sq*1.000000*1.000000+DDQtzx_onehalfQtxy_Sq*0.666667*1.000000
        
        
        #####DxzDxz|PxPx#####
        riYY[...,245]=DDq_Sq*1.000000*1.000000+DDq_PPQtzx_onehalfQtxy*1.000000*-0.666667+DDQtzx_onehalfQtxy_Sq*0.666667*1.000000+DDQtzx_onehalfQtxy_PPQtzx_onehalfQtxy*0.666667*-0.666667+DDQxy_PPQxy*1.000000*1.000000
        
        
        #####DxzDxz|PyPy#####
        riYY[...,425]=DDq_Sq*1.000000*1.000000+DDq_PPQtzx_onehalfQtxy*1.000000*-0.666667+DDQtzx_onehalfQtxy_Sq*0.666667*1.000000+DDQtzx_onehalfQtxy_PPQtzx_onehalfQtxy*0.666667*-0.666667-DDQxy_PPQxy*1.000000*1.000000
        
        #####DxzDxz|SPz#####
        riYY[...,65]=DDq_SPUz*1.000000*1.000000+DDQtzx_onehalfQtxy_SPUz*0.666667*1.000000
        
        
        #####DxzDxz|PzPz#####
        riYY[...,110]=DDq_Sq*1.000000*1.000000+DDq_PPQtzx_onehalfQtxy*1.000000*1.333333+DDQtzx_onehalfQtxy_Sq*0.666667*1.000000+DDQtzx_onehalfQtxy_PPQtzx_onehalfQtxy*0.666667*1.333333
        
        
        #####DxzDxz|DxyDxy#####
        riYY[...,2000]=DDq_DDq*1.000000*1.000000+DDq_DDQtzx_onehalfQtxy*1.000000*-1.333333+DDQtzx_onehalfQtxy_DDq*0.666667*1.000000+DDQtzx_onehalfQtxy_DDQtzx_onehalfQtxy*0.666667*-1.333333
        
        
        #####DxzDxz|PxDxz#####
        riYY[...,785]=DDq_DPUz*1.000000*1.000000+DDQtzx_onehalfQtxy_DPUz*0.666667*1.000000
        
        
        #####DxzDxz|DxzDxz#####
        riYY[...,920]=DDq_DDq*1.000000*1.000000+DDq_DDQtzx_onehalfQtxy*1.000000*0.666667+DDQtzx_onehalfQtxy_DDq*0.666667*1.000000+DDQtzx_onehalfQtxy_DDQtzx_onehalfQtxy*0.666667*0.666667+DDQxy_DDQxy*1.000000*1.000000
        
        #####DxzDxz|SDz2#####
        riYY[...,470]=DDq_DSQtzx_onehalfQtxy*1.000000*1.154701+DDQtzx_onehalfQtxy_DSQtzx_onehalfQtxy*0.666667*1.154701
        
        
        #####DxzDxz|PzDz2#####
        riYY[...,515]=DDq_DPUz*1.000000*1.154701+DDQtzx_onehalfQtxy_DPUz*0.666667*1.154701
        
        
        #####DxzDxz|Dz2Dz2#####
        riYY[...,650]=DDq_DDq*1.000000*1.000000+DDq_DDQtzx_onehalfQtxy*1.000000*1.333333+DDQtzx_onehalfQtxy_DDq*0.666667*1.000000+DDQtzx_onehalfQtxy_DDQtzx_onehalfQtxy*0.666667*1.333333
        
        
        #####DxzDxz|PyDyz#####
        riYY[...,1100]=DDq_DPUz*1.000000*1.000000+DDQtzx_onehalfQtxy_DPUz*0.666667*1.000000
        
        
        #####DxzDxz|DyzDyz#####
        riYY[...,1235]=DDq_DDq*1.000000*1.000000+DDq_DDQtzx_onehalfQtxy*1.000000*0.666667+DDQtzx_onehalfQtxy_DDq*0.666667*1.000000+DDQtzx_onehalfQtxy_DDQtzx_onehalfQtxy*0.666667*0.666667-DDQxy_DDQxy*1.000000*1.000000
        
        #####DxzDxz|SDx2-y2#####
        riYY[...,1280]=DDQxy_DSQxy*1.000000*1.000000
        
        
        #####DxzDxz|Dz2Dx2-y2#####
        riYY[...,1460]=DDQxy_DDQxy*1.000000*-1.154701
        
        
        #####DxzDxz|Dx2-y2Dx2-y2#####
        riYY[...,1595]=DDq_DDq*1.000000*1.000000+DDq_DDQtzx_onehalfQtxy*1.000000*-1.333333+DDQtzx_onehalfQtxy_DDq*0.666667*1.000000+DDQtzx_onehalfQtxy_DDQtzx_onehalfQtxy*0.666667*-1.333333
        
        #####SDz2|SS#####
        riYY[...,10]=DSQtzx_onehalfQtxy_Sq*1.154701*1.000000
        
        
        #####SDz2|PxPx#####
        riYY[...,235]=DSQtzx_onehalfQtxy_Sq*1.154701*1.000000+DSQtzx_onehalfQtxy_PPQtzx_onehalfQtxy*1.154701*-0.666667
        
        
        #####SDz2|PyPy#####
        riYY[...,415]=DSQtzx_onehalfQtxy_Sq*1.154701*1.000000+DSQtzx_onehalfQtxy_PPQtzx_onehalfQtxy*1.154701*-0.666667
        
        #####SDz2|SPz#####
        riYY[...,55]=DSQtzx_onehalfQtxy_SPUz*1.154701*1.000000
        
        
        #####SDz2|PzPz#####
        riYY[...,100]=DSQtzx_onehalfQtxy_Sq*1.154701*1.000000+DSQtzx_onehalfQtxy_PPQtzx_onehalfQtxy*1.154701*1.333333
        
        
        #####SDz2|DxyDxy#####
        riYY[...,1990]=DSQtzx_onehalfQtxy_DDq*1.154701*1.000000+DSQtzx_onehalfQtxy_DDQtzx_onehalfQtxy*1.154701*-1.333333
        
        
        #####SDz2|PxDxz#####
        riYY[...,775]=DSQtzx_onehalfQtxy_DPUz*1.154701*1.000000
        
        
        #####SDz2|DxzDxz#####
        riYY[...,910]=DSQtzx_onehalfQtxy_DDq*1.154701*1.000000+DSQtzx_onehalfQtxy_DDQtzx_onehalfQtxy*1.154701*0.666667
        
        #####SDz2|SDz2#####
        riYY[...,460]=DSQtzx_onehalfQtxy_DSQtzx_onehalfQtxy*1.154701*1.154701
        
        
        #####SDz2|PzDz2#####
        riYY[...,505]=DSQtzx_onehalfQtxy_DPUz*1.154701*1.154701
        
        
        #####SDz2|Dz2Dz2#####
        riYY[...,640]=DSQtzx_onehalfQtxy_DDq*1.154701*1.000000+DSQtzx_onehalfQtxy_DDQtzx_onehalfQtxy*1.154701*1.333333
        
        
        #####SDz2|PyDyz#####
        riYY[...,1090]=DSQtzx_onehalfQtxy_DPUz*1.154701*1.000000
        
        
        #####SDz2|DyzDyz#####
        riYY[...,1225]=DSQtzx_onehalfQtxy_DDq*1.154701*1.000000+DSQtzx_onehalfQtxy_DDQtzx_onehalfQtxy*1.154701*0.666667
        
        
        #####SDz2|Dx2-y2Dx2-y2#####
        riYY[...,1585]=DSQtzx_onehalfQtxy_DDq*1.154701*1.000000+DSQtzx_onehalfQtxy_DDQtzx_onehalfQtxy*1.154701*-1.333333
        
        
        #####PxDz2|SPx#####
        riYY[...,147]=DPUpi_SPUpi*-0.577350*1.000000
        
        
        #####PxDz2|PzPx#####
        riYY[...,192]=DPUpi_PPQpiz*-0.577350*1.000000
        
        
        #####PxDz2|PyDxy#####
        riYY[...,1767]=DPUpi_DPUpi*-0.577350*1.000000
        
        
        #####PxDz2|SDxz#####
        riYY[...,687]=DPUpi_DSQpiz*-0.577350*1.000000
        
        
        #####PxDz2|PzDxz#####
        riYY[...,732]=DPUpi_DPUpi*-0.577350*1.000000
        
        
        #####PxDz2|PxDz2#####
        riYY[...,552]=DPUpi_DPUpi*-0.577350*-0.577350
        
        
        #####PxDz2|Dz2Dxz#####
        riYY[...,867]=DPUpi_DDQpiz*-0.577350*0.577350
        
        
        #####PxDz2|DyzDxy#####
        riYY[...,1902]=DPUpi_DDQpiz*-0.577350*1.000000
        
        
        #####PxDz2|PxDx2-y2#####
        riYY[...,1362]=DPUpi_DPUpi*-0.577350*1.000000
        
        
        #####PxDz2|DxzDx2-y2#####
        riYY[...,1497]=DPUpi_DDQpiz*-0.577350*1.000000
        
        #####PyDz2|SPy#####
        riYY[...,283]=DPUpi_SPUpi*1.000000*1.000000*-0.577350
        
        
        #####PyDz2|PzPy#####
        riYY[...,328]=-DPUpi_PPQpiz*1.000000*1.000000*0.57735026918962995
        
        
        #####PyDz2|PxDxy#####
        riYY[...,1723]=DPUpi_DPUpi*1.000000*1.000000*-0.57735026918962995
        
        
        #####PyDz2|DxzDxy#####
        riYY[...,1858]=DPUpi_DDQpiz*1.000000*1.000000*-0.57735026918962995
        
        
        #####PyDz2|PyDz2#####
        riYY[...,598]=DPUpi_DPUpi*0.57735026918962995*0.57735026918962995
        
        
        #####PyDz2|SDyz#####
        riYY[...,958]=DPUpi_DSQpiz*1.000000*1.000000*-0.57735026918962995
        
        
        #####PyDz2|PzDyz#####
        riYY[...,1003]=DPUpi_DPUpi*1.000000*1.000000*-0.57735026918962995
        
        
        #####PyDz2|Dz2Dyz#####
        riYY[...,1138]=DPUpi_DDQpiz*1.000000*1.000000*0.57735026918962995*-0.57735026918962995
        
        
        #####PyDz2|PyDx2-y2#####
        riYY[...,1408]=DPUpi_DPUpi*1.000000*1.000000*0.57735026918962995
        
        
        #####PyDz2|DyzDx2-y2#####
        riYY[...,1543]=DPUpi_DDQpiz*1.000000*1.000000*0.57735026918962995
        
        
        #####PzDz2|SS#####
        riYY[...,11]=DPUz_Sq*1.154701*1.000000
        
        
        #####PzDz2|PxPx#####
        riYY[...,236]=DPUz_Sq*1.154701*1.000000+DPUz_PPQtzx_onehalfQtxy*1.154701*-0.666667
        
        
        #####PzDz2|PyPy#####
        riYY[...,416]=DPUz_Sq*1.154701*1.000000+DPUz_PPQtzx_onehalfQtxy*1.154701*-0.666667
        
        #####PzDz2|SPz#####
        riYY[...,56]=DPUz_SPUz*1.154701*1.000000
        
        
        #####PzDz2|PzPz#####
        riYY[...,101]=DPUz_Sq*1.154701*1.000000+DPUz_PPQtzx_onehalfQtxy*1.154701*1.333333
        
        #####PzDz2|DxyDxy#####
        riYY[...,1991]=DPUz_DDq*1.154701*1.000000+DPUz_DDQtzx_onehalfQtxy*1.154701*-1.333333
        
        
        #####PzDz2|PxDxz#####
        riYY[...,776]=DPUz_DPUz*1.154701*1.000000
        
        
        #####PzDz2|DxzDxz#####
        riYY[...,911]=DPUz_DDq*1.154701*1.000000+DPUz_DDQtzx_onehalfQtxy*1.154701*0.666667
        
        #####PzDz2|SDz2#####
        riYY[...,461]=DPUz_DSQtzx_onehalfQtxy*1.154701*1.154701
        
        
        #####PzDz2|PzDz2#####
        riYY[...,506]=DPUz_DPUz*1.154701*1.154701
        
        
        #####PzDz2|Dz2Dz2#####
        riYY[...,641]=DPUz_DDq*1.154701*1.000000+DPUz_DDQtzx_onehalfQtxy*1.154701*1.333333
        
        
        #####PzDz2|PyDyz#####
        riYY[...,1091]=DPUz_DPUz*1.154701*1.000000
        
        
        #####PzDz2|DyzDyz#####
        riYY[...,1226]=DPUz_DDq*1.154701*1.000000+DPUz_DDQtzx_onehalfQtxy*1.154701*0.666667
        
        
        #####PzDz2|Dx2-y2Dx2-y2#####
        riYY[...,1586]=DPUz_DDq*1.154701*1.000000+DPUz_DDQtzx_onehalfQtxy*1.154701*-1.333333
        
        
        #####Dz2Dxy|PxPy#####
        riYY[...,400]=tildaDDQxy_PPQxy*1.000000*1.000000*-1.1547005383793001
        
        
        #####Dz2Dxy|SDxy#####
        riYY[...,1660]=tildaDDQxy_DSQxy*1.000000*1.000000*-1.1547005383793001
        
        
        #####Dz2Dxy|Dz2Dxy#####
        riYY[...,1840]=tildaDDQxy_DDQxy*1.000000*1.000000*1.333333333333333
        
        
        #####Dz2Dxy|DxzDyz#####
        riYY[...,1210]=tildaDDQxy_DDQxy*1.000000*1.000000*-1.1547005383793001
        
        
        #####Dz2Dxz|SPx#####
        riYY[...,154]=DDQpiz_SPUpi*0.577350*1.000000
        
        
        #####Dz2Dxz|PzPx#####
        riYY[...,199]=DDQpiz_PPQpiz*0.577350*1.000000
        
        
        #####Dz2Dxz|PyDxy#####
        riYY[...,1774]=DDQpiz_DPUpi*0.577350*1.000000
        
        
        #####Dz2Dxz|SDxz#####
        riYY[...,694]=DDQpiz_DSQpiz*0.577350*1.000000
        
        
        #####Dz2Dxz|PzDxz#####
        riYY[...,739]=DDQpiz_DPUpi*0.577350*1.000000
        
        
        #####Dz2Dxz|PxDz2#####
        riYY[...,559]=DDQpiz_DPUpi*0.577350*-0.577350
        
        
        #####Dz2Dxz|Dz2Dxz#####
        riYY[...,874]=DDQpiz_DDQpiz*0.577350*0.577350
        
        
        #####Dz2Dxz|DyzDxy#####
        riYY[...,1909]=DDQpiz_DDQpiz*0.577350*1.000000
        
        
        #####Dz2Dxz|PxDx2-y2#####
        riYY[...,1369]=DDQpiz_DPUpi*0.577350*1.000000
        
        
        #####Dz2Dxz|DxzDx2-y2#####
        riYY[...,1504]=DDQpiz_DDQpiz*0.577350*1.000000
        
        
        #####Dz2Dz2|SS#####
        riYY[...,14]=DDq_Sq*1.000000*1.000000+DDQtzx_onehalfQtxy_Sq*1.333333*1.000000
        
        
        #####Dz2Dz2|PxPx#####
        riYY[...,239]=DDq_Sq*1.000000*1.000000+DDq_PPQtzx_onehalfQtxy*1.000000*-0.666667+DDQtzx_onehalfQtxy_Sq*1.333333*1.000000+DDQtzx_onehalfQtxy_PPQtzx_onehalfQtxy*1.333333*-0.666667
        
        
        #####Dz2Dz2|PyPy#####
        riYY[...,419]=DDq_Sq*1.000000*1.000000+DDq_PPQtzx_onehalfQtxy*1.000000*-0.666667+DDQtzx_onehalfQtxy_Sq*1.333333*1.000000+DDQtzx_onehalfQtxy_PPQtzx_onehalfQtxy*1.333333*-0.666667
        
        #####Dz2Dz2|SPz#####
        riYY[...,59]=DDq_SPUz*1.000000*1.000000+DDQtzx_onehalfQtxy_SPUz*1.333333*1.000000
        
        
        #####Dz2Dz2|PzPz#####
        riYY[...,104]=DDq_Sq*1.000000*1.000000+DDq_PPQtzx_onehalfQtxy*1.000000*1.333333+DDQtzx_onehalfQtxy_Sq*1.333333*1.000000+DDQtzx_onehalfQtxy_PPQtzx_onehalfQtxy*1.333333*1.333333
        
        #####Dz2Dz2|DxyDxy#####
        riYY[...,1994]=DDq_DDq*1.000000*1.000000+DDq_DDQtzx_onehalfQtxy*1.000000*-1.333333+DDQtzx_onehalfQtxy_DDq*1.333333*1.000000+DDQtzx_onehalfQtxy_DDQtzx_onehalfQtxy*1.333333*-1.333333
        
        
        #####Dz2Dz2|PxDxz#####
        riYY[...,779]=DDq_DPUz*1.000000*1.000000+DDQtzx_onehalfQtxy_DPUz*1.333333*1.000000
        
        
        #####Dz2Dz2|DxzDxz#####
        riYY[...,914]=DDq_DDq*1.000000*1.000000+DDq_DDQtzx_onehalfQtxy*1.000000*0.666667+DDQtzx_onehalfQtxy_DDq*1.333333*1.000000+DDQtzx_onehalfQtxy_DDQtzx_onehalfQtxy*1.333333*0.666667
        
        #####Dz2Dz2|SDz2#####
        riYY[...,464]=DDq_DSQtzx_onehalfQtxy*1.000000*1.154701+DDQtzx_onehalfQtxy_DSQtzx_onehalfQtxy*1.333333*1.154701
        
        
        #####Dz2Dz2|PzDz2#####
        riYY[...,509]=DDq_DPUz*1.000000*1.154701+DDQtzx_onehalfQtxy_DPUz*1.333333*1.154701
        
        
        #####Dz2Dz2|Dz2Dz2#####
        riYY[...,644]=DDq_DDq*1.000000*1.000000+DDq_DDQtzx_onehalfQtxy*1.000000*1.333333+DDQtzx_onehalfQtxy_DDq*1.333333*1.000000+DDQtzx_onehalfQtxy_DDQtzx_onehalfQtxy*1.333333*1.333333
        
        
        #####Dz2Dz2|PyDyz#####
        riYY[...,1094]=DDq_DPUz*1.000000*1.000000+DDQtzx_onehalfQtxy_DPUz*1.333333*1.000000
        
        
        #####Dz2Dz2|DyzDyz#####
        riYY[...,1229]=DDq_DDq*1.000000*1.000000+DDq_DDQtzx_onehalfQtxy*1.000000*0.666667+DDQtzx_onehalfQtxy_DDq*1.333333*1.000000+DDQtzx_onehalfQtxy_DDQtzx_onehalfQtxy*1.333333*0.666667
        
        
        #####Dz2Dz2|Dx2-y2Dx2-y2#####
        riYY[...,1589]=DDq_DDq*1.000000*1.000000+DDq_DDQtzx_onehalfQtxy*1.000000*-1.333333+DDQtzx_onehalfQtxy_DDq*1.333333*1.000000+DDQtzx_onehalfQtxy_DDQtzx_onehalfQtxy*1.333333*-1.333333
        
        
        #####SDyz|SPy#####
        riYY[...,291]=DSQpiz_SPUpi*1.000000*1.000000
        
        
        #####SDyz|PzPy#####
        riYY[...,336]=DSQpiz_PPQpiz*1.000000*1.000000
        
        
        #####SDyz|PxDxy#####
        riYY[...,1731]=DSQpiz_DPUpi*1.000000*1.000000
        
        
        #####SDyz|DxzDxy#####
        riYY[...,1866]=DSQpiz_DDQpiz*1.000000*1.000000
        
        
        #####SDyz|PyDz2#####
        riYY[...,606]=DSQpiz_DPUpi*1.000000*1.000000*-0.57735026918962995
        
        
        #####SDyz|SDyz#####
        riYY[...,966]=DSQpiz_DSQpiz*1.000000*1.000000
        
        
        #####SDyz|PzDyz#####
        riYY[...,1011]=DSQpiz_DPUpi*1.000000*1.000000
        
        
        #####SDyz|Dz2Dyz#####
        riYY[...,1146]=DSQpiz_DDQpiz*1.000000*1.000000*0.57735026918962995
        
        
        #####SDyz|PyDx2-y2#####
        riYY[...,1416]=-DSQpiz_DPUpi*1.000000*1.000000
        
        
        #####SDyz|DyzDx2-y2#####
        riYY[...,1551]=-DSQpiz_DDQpiz*1.000000*1.000000
        
        
        #####PyDyz|SS#####
        riYY[...,24]=DPUz_Sq*1.000000*1.000000
        
        
        #####PyDyz|PxPx#####
        riYY[...,249]=DPUz_Sq*1.000000*1.000000+DPUz_PPQtzx_onehalfQtxy*1.000000*-0.666667
        
        
        #####PyDyz|PyPy#####
        riYY[...,429]=DPUz_Sq*1.000000*1.000000+DPUz_PPQtzx_onehalfQtxy*1.000000*-0.666667
        
        #####PyDyz|SPz#####
        riYY[...,69]=DPUz_SPUz*1.000000*1.000000
        
        
        #####PyDyz|PzPz#####
        riYY[...,114]=DPUz_Sq*1.000000*1.000000+DPUz_PPQtzx_onehalfQtxy*1.000000*1.333333
        
        
        #####PyDyz|DxyDxy#####
        riYY[...,2004]=DPUz_DDq*1.000000*1.000000+DPUz_DDQtzx_onehalfQtxy*1.000000*-1.333333
        
        
        #####PyDyz|PxDxz#####
        riYY[...,789]=DPUz_DPUz*1.000000*1.000000
        
        
        #####PyDyz|DxzDxz#####
        riYY[...,924]=DPUz_DDq*1.000000*1.000000+DPUz_DDQtzx_onehalfQtxy*1.000000*0.666667
        
        #####PyDyz|SDz2#####
        riYY[...,474]=DPUz_DSQtzx_onehalfQtxy*1.000000*1.154701
        
        
        #####PyDyz|PzDz2#####
        riYY[...,519]=DPUz_DPUz*1.000000*1.154701
        
        
        #####PyDyz|Dz2Dz2#####
        riYY[...,654]=DPUz_DDq*1.000000*1.000000+DPUz_DDQtzx_onehalfQtxy*1.000000*1.333333
        
        
        #####PyDyz|PyDyz#####
        riYY[...,1104]=DPUz_DPUz*1.000000*1.000000
        
        
        #####PyDyz|DyzDyz#####
        riYY[...,1239]=DPUz_DDq*1.000000*1.000000+DPUz_DDQtzx_onehalfQtxy*1.000000*0.666667
        
        
        #####PyDyz|Dx2-y2Dx2-y2#####
        riYY[...,1599]=DPUz_DDq*1.000000*1.000000+DPUz_DDQtzx_onehalfQtxy*1.000000*-1.333333
        
        
        #####PzDyz|SPy#####
        riYY[...,292]=DPUpi_SPUpi*1.000000*1.000000
        
        
        #####PzDyz|PzPy#####
        riYY[...,337]=DPUpi_PPQpiz*1.000000*1.000000
        
        
        #####PzDyz|PxDxy#####
        riYY[...,1732]=DPUpi_DPUpi*1.000000*1.000000
        
        
        #####PzDyz|DxzDxy#####
        riYY[...,1867]=DPUpi_DDQpiz*1.000000*1.000000
        
        
        #####PzDyz|PyDz2#####
        riYY[...,607]=DPUpi_DPUpi*1.000000*1.000000*-0.57735026918962995
        
        
        #####PzDyz|SDyz#####
        riYY[...,967]=DPUpi_DSQpiz*1.000000*1.000000
        
        
        #####PzDyz|PzDyz#####
        riYY[...,1012]=DPUpi_DPUpi*1.000000*1.000000
        
        
        #####PzDyz|Dz2Dyz#####
        riYY[...,1147]=DPUpi_DDQpiz*1.000000*1.000000*0.57735026918962995
        
        
        #####PzDyz|PyDx2-y2#####
        riYY[...,1417]=-DPUpi_DPUpi*1.000000*1.000000
        
        
        #####PzDyz|DyzDx2-y2#####
        riYY[...,1552]=-DPUpi_DDQpiz*1.000000*1.000000
        
        
        #####DyzDxy|SPx#####
        riYY[...,177]=DDQpiz_SPUpi*1.000000*1.000000
        
        
        #####DyzDxy|PzPx#####
        riYY[...,222]=DDQpiz_PPQpiz*1.000000*1.000000
        
        
        #####DyzDxy|PyDxy#####
        riYY[...,1797]=DDQpiz_DPUpi*1.000000*1.000000
        
        
        #####DyzDxy|SDxz#####
        riYY[...,717]=DDQpiz_DSQpiz*1.000000*1.000000
        
        
        #####DyzDxy|PzDxz#####
        riYY[...,762]=DDQpiz_DPUpi*1.000000*1.000000
        
        
        #####DyzDxy|PxDz2#####
        riYY[...,582]=DDQpiz_DPUpi*1.000000*-0.577350
        
        
        #####DyzDxy|Dz2Dxz#####
        riYY[...,897]=DDQpiz_DDQpiz*1.000000*0.577350
        
        
        #####DyzDxy|DyzDxy#####
        riYY[...,1932]=DDQpiz_DDQpiz*1.000000*1.000000
        
        
        #####DyzDxy|PxDx2-y2#####
        riYY[...,1392]=DDQpiz_DPUpi*1.000000*1.000000
        
        
        #####DyzDxy|DxzDx2-y2#####
        riYY[...,1527]=DDQpiz_DDQpiz*1.000000*1.000000
        
        
        #####DxzDyz|PxPy#####
        riYY[...,386]=tildaDDQxy_PPQxy*1.000000*1.000000
        
        
        #####DxzDyz|SDxy#####
        riYY[...,1646]=DDQxy_DSQxy*1.000000*1.000000
        
        
        #####DxzDyz|Dz2Dxy#####
        riYY[...,1826]=tildaDDQxy_DDQxy*1.000000*1.000000*-1.1547005383793001
        
        
        #####DxzDyz|DxzDyz#####
        riYY[...,1196]=tildaDDQxy_DDQxy*1.000000*1.000000
        
        
        #####Dz2Dyz|SPy#####
        riYY[...,295]=DDQpiz_SPUpi*1.000000*1.000000*0.57735026918962995
        
        
        #####Dz2Dyz|PzPy#####
        riYY[...,340]=DDQpiz_PPQpiz*1.000000*1.000000*0.57735026918962995
        
        
        #####Dz2Dyz|PxDxy#####
        riYY[...,1735]=DDQpiz_DPUpi*1.000000*1.000000*0.57735026918962995
        
        
        #####Dz2Dyz|DxzDxy#####
        riYY[...,1870]=DDQpiz_DDQpiz*1.000000*1.000000*0.5773502691896299
        
        
        #####Dz2Dyz|PyDz2#####
        riYY[...,610]=DDQpiz_DPUpi*1.000000*1.000000*0.57735026918962995*-0.5773502691896299
        
        
        #####Dz2Dyz|SDyz#####
        riYY[...,970]=DDQpiz_DSQpiz*1.000000*1.000000*0.57735026918962995
        
        
        #####Dz2Dyz|PzDyz#####
        riYY[...,1015]=DDQpiz_DPUpi*1.000000*1.000000*0.57735026918962995
        
        
        #####Dz2Dyz|Dz2Dyz#####
        riYY[...,1150]=DDQpiz_DDQpiz*1.000000*1.000000*0.57735026918962995*0.5773502691896295
        
        
        #####Dz2Dyz|PyDx2-y2#####
        riYY[...,1420]=DDQpiz_DPUpi*1.000000*1.000000*-0.5773502691896295
        
        
        #####Dz2Dyz|DyzDx2-y2#####
        riYY[...,1555]=DDQpiz_DDQpiz*1.000000*1.000000*-0.5773502691896295
        
        
        #####DyzDyz|SS#####
        riYY[...,27]=DDq_Sq*1.000000*1.000000+DDQtzx_onehalfQtxy_Sq*0.666667*1.000000
        
        
        #####DyzDyz|PxPx#####
        riYY[...,252]=DDq_Sq*1.000000*1.000000+DDq_PPQtzx_onehalfQtxy*1.000000*-0.666667+DDQtzx_onehalfQtxy_Sq*0.666667*1.000000+DDQtzx_onehalfQtxy_PPQtzx_onehalfQtxy*0.666667*-0.666667-DDQxy_PPQxy*1.000000*1.000000
        
        
        #####DyzDyz|PyPy#####
        riYY[...,432]=DDq_Sq*1.000000*1.000000+DDq_PPQtzx_onehalfQtxy*1.000000*-0.666667+DDQtzx_onehalfQtxy_Sq*0.666667*1.000000+DDQtzx_onehalfQtxy_PPQtzx_onehalfQtxy*0.666667*-0.666667+DDQxy_PPQxy*1.000000*1.000000
        
        #####DyzDyz|SPz#####
        riYY[...,72]=DDq_SPUz*1.000000*1.000000+DDQtzx_onehalfQtxy_SPUz*0.666667*1.000000
        
        
        #####DyzDyz|PzPz#####
        riYY[...,117]=DDq_Sq*1.000000*1.000000+DDq_PPQtzx_onehalfQtxy*1.000000*1.333333+DDQtzx_onehalfQtxy_Sq*0.666667*1.000000+DDQtzx_onehalfQtxy_PPQtzx_onehalfQtxy*0.666667*1.333333
        
        
        #####DyzDyz|DxyDxy#####
        riYY[...,2007]=DDq_DDq*1.000000*1.000000+DDq_DDQtzx_onehalfQtxy*1.000000*-1.333333+DDQtzx_onehalfQtxy_DDq*0.666667*1.000000+DDQtzx_onehalfQtxy_DDQtzx_onehalfQtxy*0.666667*-1.333333
        
        
        #####DyzDyz|PxDxz#####
        riYY[...,792]=DDq_DPUz*1.000000*1.000000+DDQtzx_onehalfQtxy_DPUz*0.666667*1.000000
        
        
        #####DyzDyz|DxzDxz#####
        riYY[...,927]=DDq_DDq*1.000000*1.000000+DDq_DDQtzx_onehalfQtxy*1.000000*0.666667+DDQtzx_onehalfQtxy_DDq*0.666667*1.000000+DDQtzx_onehalfQtxy_DDQtzx_onehalfQtxy*0.666667*0.666667-DDQxy_DDQxy*1.000000*1.000000
        
        #####DyzDyz|SDz2#####
        riYY[...,477]=DDq_DSQtzx_onehalfQtxy*1.000000*1.154701+DDQtzx_onehalfQtxy_DSQtzx_onehalfQtxy*0.666667*1.154701
        
        
        #####DyzDyz|PzDz2#####
        riYY[...,522]=DDq_DPUz*1.000000*1.154701+DDQtzx_onehalfQtxy_DPUz*0.666667*1.154701
        
        
        #####DyzDyz|Dz2Dz2#####
        riYY[...,657]=DDq_DDq*1.000000*1.000000+DDq_DDQtzx_onehalfQtxy*1.000000*1.333333+DDQtzx_onehalfQtxy_DDq*0.666667*1.000000+DDQtzx_onehalfQtxy_DDQtzx_onehalfQtxy*0.666667*1.333333
        
        
        #####DyzDyz|PyDyz#####
        riYY[...,1107]=DDq_DPUz*1.000000*1.000000+DDQtzx_onehalfQtxy_DPUz*0.666667*1.000000
        
        
        #####DyzDyz|DyzDyz#####
        riYY[...,1242]=DDq_DDq*1.000000*1.000000+DDq_DDQtzx_onehalfQtxy*1.000000*0.666667+DDQtzx_onehalfQtxy_DDq*0.666667*1.000000+DDQtzx_onehalfQtxy_DDQtzx_onehalfQtxy*0.666667*0.666667+DDQxy_DDQxy*1.000000*1.000000
        
        #####DyzDyz|SDx2-y2#####
        riYY[...,1287]=-DDQxy_DSQxy*1.000000*1.000000
        
        
        #####DyzDyz|Dz2Dx2-y2#####
        riYY[...,1467]=-DDQxy_DDQxy*1.000000*-1.154701
        
        
        #####DyzDyz|Dx2-y2Dx2-y2#####
        riYY[...,1602]=DDq_DDq*1.000000*1.000000+DDq_DDQtzx_onehalfQtxy*1.000000*-1.333333+DDQtzx_onehalfQtxy_DDq*0.666667*1.000000+DDQtzx_onehalfQtxy_DDQtzx_onehalfQtxy*0.666667*-1.333333
        
        
        #####SDx2-y2|PxPx#####
        riYY[...,253]=DSQxy_PPQxy*1.000000*1.000000
        
        
        #####SDx2-y2|PyPy#####
        riYY[...,433]=-DSQxy_PPQxy*1.000000*1.000000
        
        
        #####SDx2-y2|DxzDxz#####
        riYY[...,928]=DSQxy_DDQxy*1.000000*1.000000
        
        
        #####SDx2-y2|DyzDyz#####
        riYY[...,1243]=-DSQxy_DDQxy*1.000000*1.000000
        
        #####SDx2-y2|SDx2-y2#####
        riYY[...,1288]=DSQxy_DSQxy*1.000000*1.000000
        
        
        #####SDx2-y2|Dz2Dx2-y2#####
        riYY[...,1468]=DSQxy_DDQxy*1.000000*-1.154701
        
        
        #####PxDx2-y2|SPx#####
        riYY[...,165]=DPUpi_SPUpi*1.000000*1.000000
        
        
        #####PxDx2-y2|PzPx#####
        riYY[...,210]=DPUpi_PPQpiz*1.000000*1.000000
        
        
        #####PxDx2-y2|PyDxy#####
        riYY[...,1785]=DPUpi_DPUpi*1.000000*1.000000
        
        
        #####PxDx2-y2|SDxz#####
        riYY[...,705]=DPUpi_DSQpiz*1.000000*1.000000
        
        
        #####PxDx2-y2|PzDxz#####
        riYY[...,750]=DPUpi_DPUpi*1.000000*1.000000
        
        
        #####PxDx2-y2|PxDz2#####
        riYY[...,570]=DPUpi_DPUpi*1.000000*-0.577350
        
        
        #####PxDx2-y2|Dz2Dxz#####
        riYY[...,885]=DPUpi_DDQpiz*1.000000*0.577350
        
        
        #####PxDx2-y2|DyzDxy#####
        riYY[...,1920]=DPUpi_DDQpiz*1.000000*1.000000
        
        
        #####PxDx2-y2|PxDx2-y2#####
        riYY[...,1380]=DPUpi_DPUpi*1.000000*1.000000
        
        
        #####PxDx2-y2|DxzDx2-y2#####
        riYY[...,1515]=DPUpi_DDQpiz*1.000000*1.000000
        
        
        #####PyDx2-y2|SPy#####
        riYY[...,301]=-DPUpi_SPUpi*1.000000*1.000000
        
        
        #####PyDx2-y2|PzPy#####
        riYY[...,346]=-DPUpi_PPQpiz*1.000000*1.000000
        
        
        #####PyDx2-y2|PxDxy#####
        riYY[...,1741]=-DPUpi_DPUpi*1.000000*1.000000
        
        
        #####PyDx2-y2|DxzDxy#####
        riYY[...,1876]=-DPUpi_DDQpiz*1.000000*1.000000
        
        
        #####PyDx2-y2|PyDz2#####
        riYY[...,616]=DPUpi_DPUpi*1.000000*1.000000*0.57735026918962995
        
        
        #####PyDx2-y2|SDyz#####
        riYY[...,976]=-DPUpi_DSQpiz*1.000000*1.000000
        
        
        #####PyDx2-y2|PzDyz#####
        riYY[...,1021]=-DPUpi_DPUpi*1.000000*1.000000
        
        
        #####PyDx2-y2|Dz2Dyz#####
        riYY[...,1156]=DPUpi_DDQpiz*1.000000*1.000000*-0.5773502691896295
        
        
        #####PyDx2-y2|PyDx2-y2#####
        riYY[...,1426]=DPUpi_DPUpi*1.000000*1.000000
        
        
        #####PyDx2-y2|DyzDx2-y2#####
        riYY[...,1561]=DPUpi_DDQpiz*1.000000*1.000000
        
        #####DxzDx2-y2|SPx#####
        riYY[...,168]=DDQpiz_SPUpi*1.000000*1.000000
        
        
        #####DxzDx2-y2|PzPx#####
        riYY[...,213]=DDQpiz_PPQpiz*1.000000*1.000000
        
        
        #####DxzDx2-y2|PyDxy#####
        riYY[...,1788]=DDQpiz_DPUpi*1.000000*1.000000
        
        
        #####DxzDx2-y2|SDxz#####
        riYY[...,708]=DDQpiz_DSQpiz*1.000000*1.000000
        
        
        #####DxzDx2-y2|PzDxz#####
        riYY[...,753]=DDQpiz_DPUpi*1.000000*1.000000
        
        
        #####DxzDx2-y2|PxDz2#####
        riYY[...,573]=DDQpiz_DPUpi*1.000000*-0.577350
        
        
        #####DxzDx2-y2|Dz2Dxz#####
        riYY[...,888]=DDQpiz_DDQpiz*1.000000*0.577350
        
        
        #####DxzDx2-y2|DyzDxy#####
        riYY[...,1923]=DDQpiz_DDQpiz*1.000000*1.000000
        
        
        #####DxzDx2-y2|PxDx2-y2#####
        riYY[...,1383]=DDQpiz_DPUpi*1.000000*1.000000
        
        
        #####DxzDx2-y2|DxzDx2-y2#####
        riYY[...,1518]=DDQpiz_DDQpiz*1.000000*1.000000
        
        
        #####Dz2Dx2-y2|PxPx#####
        riYY[...,257]=DDQxy_PPQxy*-1.154701*1.000000
        
        
        #####Dz2Dx2-y2|PyPy#####
        riYY[...,437]=-DDQxy_PPQxy*-1.154701*1.000000
        
        
        #####Dz2Dx2-y2|DxzDxz#####
        riYY[...,932]=DDQxy_DDQxy*-1.154701*1.000000
        
        
        #####Dz2Dx2-y2|DyzDyz#####
        riYY[...,1247]=-DDQxy_DDQxy*-1.154701*1.000000
        
        #####Dz2Dx2-y2|SDx2-y2#####
        riYY[...,1292]=DDQxy_DSQxy*-1.154701*1.000000
        
        
        #####Dz2Dx2-y2|Dz2Dx2-y2#####
        riYY[...,1472]=DDQxy_DDQxy*-1.154701*-1.154701
        
        
        #####DyzDx2-y2|SPy#####
        riYY[...,304]=-DDQpiz_SPUpi*1.000000*1.000000
        
        
        #####DyzDx2-y2|PzPy#####
        riYY[...,349]=-DDQpiz_PPQpiz*1.000000*1.000000
        
        
        #####DyzDx2-y2|PxDxy#####
        riYY[...,1744]=-DDQpiz_DPUpi*1.000000*1.000000
        
        
        #####DyzDx2-y2|DxzDxy#####
        riYY[...,1879]=-DDQpiz_DDQpiz*1.000000*1.000000
        
        
        #####DyzDx2-y2|PyDz2#####
        riYY[...,619]=DDQpiz_DPUpi*1.000000*1.000000*0.57735026918962995
        
        
        #####DyzDx2-y2|SDyz#####
        riYY[...,979]=-DDQpiz_DSQpiz*1.000000*1.000000
        
        
        #####DyzDx2-y2|PzDyz#####
        riYY[...,1024]=-DDQpiz_DPUpi*1.000000*1.000000
        
        
        #####DyzDx2-y2|Dz2Dyz#####
        riYY[...,1159]=DDQpiz_DDQpiz*1.000000*1.000000*-0.5773502691896295
        
        
        #####DyzDx2-y2|PyDx2-y2#####
        riYY[...,1429]=DDQpiz_DPUpi*1.000000*1.000000
        
        
        #####DyzDx2-y2|DyzDx2-y2#####
        riYY[...,1564]=DDQpiz_DDQpiz*1.000000*1.000000
        
        
        #####Dx2-y2Dx2-y2|SS#####
        riYY[...,35]=DDq_Sq*1.000000*1.000000+DDQtzx_onehalfQtxy_Sq*-1.333333*1.000000
        
        
        #####Dx2-y2Dx2-y2|PxPx#####
        riYY[...,260]=DDq_Sq*1.000000*1.000000+DDq_PPQtzx_onehalfQtxy*1.000000*-0.666667+DDQtzx_onehalfQtxy_Sq*-1.333333*1.000000+DDQtzx_onehalfQtxy_PPQtzx_onehalfQtxy*-1.333333*-0.666667
        
        
        
        #####Dx2-y2Dx2-y2|PyPy#####
        riYY[...,440]=DDq_Sq*1.000000*1.000000+DDq_PPQtzx_onehalfQtxy*1.000000*-0.666667+DDQtzx_onehalfQtxy_Sq*-1.333333*1.000000+DDQtzx_onehalfQtxy_PPQtzx_onehalfQtxy*-1.333333*-0.666667
        
        #####Dx2-y2Dx2-y2|SPz#####
        riYY[...,80]=DDq_SPUz*1.000000*1.000000+DDQtzx_onehalfQtxy_SPUz*-1.333333*1.000000
        
        
        #####Dx2-y2Dx2-y2|PzPz#####
        riYY[...,125]=DDq_Sq*1.000000*1.000000+DDq_PPQtzx_onehalfQtxy*1.000000*1.333333+DDQtzx_onehalfQtxy_Sq*-1.333333*1.000000+DDQtzx_onehalfQtxy_PPQtzx_onehalfQtxy*-1.333333*1.333333
        
        
        #####Dx2-y2Dx2-y2|DxyDxy#####
        riYY[...,2015]=DDq_DDq*1.000000*1.000000+DDq_DDQtzx_onehalfQtxy*1.000000*-1.333333+DDQtzx_onehalfQtxy_DDq*-1.333333*1.000000+DDQtzx_onehalfQtxy_DDQtzx_onehalfQtxy*-1.333333*-1.333333
        
        
        #####Dx2-y2Dx2-y2|PxDxz#####
        riYY[...,800]=DDq_DPUz*1.000000*1.000000+DDQtzx_onehalfQtxy_DPUz*-1.333333*1.000000
        
        
        #####Dx2-y2Dx2-y2|DxzDxz#####
        riYY[...,935]=DDq_DDq*1.000000*1.000000+DDq_DDQtzx_onehalfQtxy*1.000000*0.666667+DDQtzx_onehalfQtxy_DDq*-1.333333*1.000000+DDQtzx_onehalfQtxy_DDQtzx_onehalfQtxy*-1.333333*0.666667
        
        #####Dx2-y2Dx2-y2|SDz2#####
        riYY[...,485]=DDq_DSQtzx_onehalfQtxy*1.000000*1.154701+DDQtzx_onehalfQtxy_DSQtzx_onehalfQtxy*-1.333333*1.154701
        
        
        #####Dx2-y2Dx2-y2|PzDz2#####
        riYY[...,530]=DDq_DPUz*1.000000*1.154701+DDQtzx_onehalfQtxy_DPUz*-1.333333*1.154701
        
        
        #####Dx2-y2Dx2-y2|Dz2Dz2#####
        riYY[...,665]=DDq_DDq*1.000000*1.000000+DDq_DDQtzx_onehalfQtxy*1.000000*1.333333+DDQtzx_onehalfQtxy_DDq*-1.333333*1.000000+DDQtzx_onehalfQtxy_DDQtzx_onehalfQtxy*-1.333333*1.333333
        
        
        #####Dx2-y2Dx2-y2|PyDyz#####
        riYY[...,1115]=DDq_DPUz*1.000000*1.000000+DDQtzx_onehalfQtxy_DPUz*-1.333333*1.000000
        
        
        #####Dx2-y2Dx2-y2|DyzDyz#####
        riYY[...,1250]=DDq_DDq*1.000000*1.000000+DDq_DDQtzx_onehalfQtxy*1.000000*0.666667+DDQtzx_onehalfQtxy_DDq*-1.333333*1.000000+DDQtzx_onehalfQtxy_DDQtzx_onehalfQtxy*-1.333333*0.666667
        
        #print('Wdd last', riYY[...,1250], DDq_DDq, DDq_DDQtzx_onehalfQtxy, DDQtzx_onehalfQtxy_DDq, DDQtzx_onehalfQtxy_DDQtzx_onehalfQtxy)
        
        #####Dx2-y2Dx2-y2|Dx2-y2Dx2-y2#####
        riYY[...,1610]=DDq_DDq*1.000000*1.000000+DDq_DDQtzx_onehalfQtxy*1.000000*-1.333333+DDQtzx_onehalfQtxy_DDq*-1.333333*1.000000+DDQtzx_onehalfQtxy_DDQtzx_onehalfQtxy*-1.333333*-1.333333
        
        
        coreYY = torch.zeros(YY.sum(),90,dtype=dtype, device=device)
        coreYYLocal = torch.zeros(YY.sum(),90,dtype=dtype, device=device)
        YYY = (((ni[XX] > 12) & (ni[XX] <18)) | ((ni[XX] > 20) & (ni[XX] <30)) | ((ni[XX] > 32) & (ni[XX] <36)) | ((ni[XX] > 38) & (ni[XX] <48)) | ((ni[XX] > 50) & (ni[XX] <54)) | ((ni[XX] > 70) & (ni[XX] <80)) | (ni[XX] ==57)) &\
         (((nj[XX] > 12) & (nj[XX] <18)) | ((nj[XX] > 20) & (nj[XX] <30)) | ((nj[XX] > 32) & (nj[XX] <36)) | ((nj[XX] > 38) & (nj[XX] <48)) | ((nj[XX] > 50) & (nj[XX] <54)) | ((nj[XX] > 70) & (nj[XX] <80)) | (nj[XX] ==57))

        coreYYLocal[...,0] = tore[ni[YY]]*riPM6b[YYY,1-1]
        coreYYLocal[...,2] = tore[ni[YY]]*riPM6b[YYY,4-1]   ##PP
        coreYYLocal[...,6] = tore[ni[YY]]*riPM6b[YYY,2-1]  ##SP
        coreYYLocal[...,9] = tore[ni[YY]]*riPM6b[YYY,3-1]  ##PP
        coreYYLocal[...,14] = tore[ni[YY]]*riYY[...,44] ##DD
        coreYYLocal[...,16] = tore[ni[YY]]*riYY[...,17]
        coreYYLocal[...,20] = tore[ni[YY]]*riYY[...,20]  ##DD
        coreYYLocal[...,21] = tore[ni[YY]]*riYY[...,10]
        coreYYLocal[...,24] = tore[ni[YY]]*riYY[...,11]
        coreYYLocal[...,27] = tore[ni[YY]]*riYY[...,14]  ## DD
        coreYYLocal[...,45] = tore[nj[YY]]*riPM6a[YYY,1-1]  
        coreYYLocal[...,47] = tore[nj[YY]]*riPM6a[YYY,12-1]
        coreYYLocal[...,51] = tore[nj[YY]]*riPM6a[YYY,5-1]
        coreYYLocal[...,54] = tore[nj[YY]]*riPM6a[YYY,11-1]
        coreYYLocal[...,59] = tore[nj[YY]]*riYY[...,1980]
        coreYYLocal[...,61] = tore[nj[YY]]*riYY[...,765]
        coreYYLocal[...,65] = tore[nj[YY]]*riYY[...,900]
        coreYYLocal[...,66] = tore[nj[YY]]*riYY[...,450]
        coreYYLocal[...,69] = tore[nj[YY]]*riYY[...,495]
        coreYYLocal[...,72] = tore[nj[YY]]*riYY[...,630]

        if(drho_corea.sum() > 0 or drho_coreb.sum() > 0 ):
            rho_coreb = drho_coreb[YY]
            rho_corea = drho_corea[YY]
            rhoSDD0 = (rho_coreb+rho3a)**2
            rhoD0SD = (rho_corea+rho3b)**2
    
            rhoSDD = (rho_coreb+rho6a)**2
            rhoSPD = (rho_coreb+rho4a)**2
            rhoSSD = (rho5b+rho_corea)**2
    
    
            rhoDDS = (rho_corea+rho6b)**2
            rhoPDS = (rho_corea+rho4b)**2
            rhoSDS = (rho5a+rho_coreb)**2
    
    
            Sq_DDq = ev/torch.sqrt(r0**2+rhoD0SD)
            #############################
            DDq_Sq = ev/torch.sqrt(r0**2+rhoSDD0)
            
            qa0 = qa0*math.sqrt(2)
            qb0 = qb0*math.sqrt(2)
    
    
    
            Sq_DDQtzx_onehalfQtxy = ev2/torch.sqrt((r0-ddb0)**2+rhoDDS)+ev2/torch.sqrt((r0+ddb0)**2+rhoDDS)-ev1/torch.sqrt((r0)**2+(ddb0)**2+rhoDDS)
            #############################
            DDQtzx_onehalfQtxy_Sq = ev2/torch.sqrt((r0-dda0)**2+rhoSDD)+ev2/torch.sqrt((r0+dda0)**2+rhoSDD)-ev1/torch.sqrt((r0)**2+(dda0)**2+rhoSDD)
    
    
            Sq_DSQtzx_onehalfQtxy = ev2/torch.sqrt((r0-dsb0)**2+rhoSSD)+ev2/torch.sqrt((r0+dsb0)**2+rhoSSD)-ev1/torch.sqrt((r0)**2+(dsb0)**2+rhoSSD)
            #############################
            DSQtzx_onehalfQtxy_Sq = ev2/torch.sqrt((r0-dsa0)**2+rhoSDS)+ev2/torch.sqrt((r0+dsa0)**2+rhoSDS)-ev1/torch.sqrt((r0)**2+(dsa0)**2+rhoSDS)
    
            qa0 = qa0/math.sqrt(2)
            qb0 = qb0/math.sqrt(2)
    
    
            Sq_DPUz = -(ev1/torch.sqrt((r0+dpb0)**2+rhoPDS)-ev1/torch.sqrt((r0-dpb0)**2+rhoPDS))
            #############################
            DPUz_Sq = (ev1/torch.sqrt((r0+dpa0)**2+rhoSPD)-ev1/torch.sqrt((r0-dpa0)**2+rhoSPD))
    
    
            coreYYLocal[Ycore,14] = tore[ni[YY][Ycore]]*(DDq_Sq[Ycore]*1.000000*1.000000 + DDQtzx_onehalfQtxy_Sq[Ycore]*-1.333333*1.000000)
            coreYYLocal[Ycore,16] = tore[ni[YY][Ycore]]*(DPUz_Sq[Ycore]*1.000000*1.000000)
            coreYYLocal[Ycore,20] = tore[ni[YY][Ycore]]*(DDq_Sq[Ycore]*1.000000*1.000000 + DDQtzx_onehalfQtxy_Sq[Ycore]*0.666667*1.000000)
            coreYYLocal[Ycore,21] = tore[ni[YY][Ycore]]*(DSQtzx_onehalfQtxy_Sq[Ycore]*1.154701*1.000000)
            coreYYLocal[Ycore,24] = tore[ni[YY][Ycore]]*(DPUz_Sq[Ycore]*1.154701*1.000000)
            coreYYLocal[Ycore,27] = tore[ni[YY][Ycore]]*(DDq_Sq[Ycore]*1.000000*1.000000 + DDQtzx_onehalfQtxy_Sq[Ycore]*1.333333*1.000000)
    
    
    
            coreYYLocal[Ycore,59] = tore[nj[YY][Ycore]]*(Sq_DDq[Ycore]*1.000000*1.000000 + Sq_DDQtzx_onehalfQtxy[Ycore]*1.000000*-1.333333)
            coreYYLocal[Ycore,61] = tore[nj[YY][Ycore]]*(Sq_DPUz[Ycore]*1.000000*1.000000)
            coreYYLocal[Ycore,65] = tore[nj[YY][Ycore]]*(Sq_DDq[Ycore]*1.000000*1.000000 + Sq_DDQtzx_onehalfQtxy[Ycore]*1.000000*0.666667)
            coreYYLocal[Ycore,66] = tore[nj[YY][Ycore]]*(Sq_DSQtzx_onehalfQtxy[Ycore]*1.000000*1.154701)
            coreYYLocal[Ycore,69] = tore[nj[YY][Ycore]]*(Sq_DPUz[Ycore]*1.000000*1.154701)
            coreYYLocal[Ycore,72] = tore[nj[YY][Ycore]]*(Sq_DDq[Ycore]*1.000000*1.000000 + Sq_DDQtzx_onehalfQtxy[Ycore]*1.000000*1.333333)
    
        coreYY[...,:45] = RotateCore(coreYYLocal[...,:45],rotationMatrix[YY,...,...],3)
        coreYY[...,45:] = RotateCore(coreYYLocal[...,45:],rotationMatrix[YY,...,...],3)
    else:
        coreYY = torch.zeros(YY.sum(),90,dtype=dtype, device=device)
        riYY =  torch.zeros(YY.sum(),2025,dtype=dtype, device=device)
    return riYH, riYX, riYY, coreYH, coreYX, coreYY
