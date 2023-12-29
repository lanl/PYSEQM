import torch
from torch import sqrt
from .constants import ev
import sys
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
def two_elec_two_center_int_local_frame(ni,nj,r0, tore, da0,db0, qa0,qb0, rho0a,rho0b, rho1a,rho1b, rho2a,rho2b,themethod ):
    """
    two electron two center integrals in local frame for each pair
    """
    dtype = r0.dtype
    device = r0.device
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
    ##if(themethod == "PM6"):
        ##HH = (ni==1) & (nj==1)
        ##XH =((ni <= 12) | ((ni >= 18) & (ni <=20)) | ((ni >= 30) & (ni <= 32)) | ((ni >= 36) & (ni <= 38)) | ((ni >= 48) & (ni <= 50)) | ((ni >= 54) & (ni <= 56)) | ((ni >= 80) & (ni <= 83))) & (ni !=1) & (nj==1)
        ##XX =((ni <= 12) | ((ni >= 18) & (ni <=20)) | ((ni >= 30) & (ni <= 32)) | ((ni >= 36) & (ni <= 38)) | ((ni >= 48) & (ni <= 50)) | ((ni >= 54) & (ni <= 56)) | ((ni >= 80) & (ni <= 83))) & (ni !=1) & \
            ##((nj <= 12) | ((nj >= 18) & (nj <=20)) | ((nj >= 30) & (nj <= 32)) | ((nj >= 36) & (nj <= 38)) | ((nj >= 48) & (nj <= 50)) | ((nj >= 54) & (nj <= 56)) | ((nj >= 80) & (nj <= 83))) & (nj !=1)

    ##else:
    HH = (ni==1) & (nj==1)
    XH = (ni>1) & (nj==1)
    XX = (ni>1) & (nj>1)

    # Hydrogen - Hydrogen
    #aeeHH = (rho0a[HH]+rho0b[HH])**2
    riHH = ev/sqrt(r0[HH]**2+(rho0a[HH]+rho0b[HH])**2)
    coreHH = torch.zeros(HH.sum(),2,dtype=dtype, device=device)
    #ni=nj=1
    coreHH[...,0] = tore[1]*riHH
    coreHH[...,1] = tore[1]*riHH

    # Heavy atom - Hydrogen
    aeeXH = (rho0a[XH]+rho0b[XH])**2
    rXH = r0[XH]
    daXH = da0[XH]
    qaXH = qa0[XH]*2.0
    adeXH = (rho1a[XH]+rho0b[XH])**2
    aqeXH = (rho2a[XH]+rho0b[XH])**2
    ev1dsqr6XH = ev1/sqrt(rXH**2 + aqeXH)
    riXH = torch.zeros(XH.sum(),4,dtype=dtype, device=device)
    eeXH = ev/sqrt(rXH**2+aeeXH)
    riXH[...,1-1] = eeXH
    riXH[...,2-1] = ev1/sqrt((rXH+daXH)**2+adeXH) \
                   - ev1/sqrt((rXH-daXH)**2+adeXH)
    riXH[...,3-1] = eeXH + ev2/sqrt((rXH+qaXH)**2+aqeXH) \
                       + ev2/sqrt((rXH-qaXH)**2+aqeXH) \
                       - ev1dsqr6XH
    riXH[...,4-1] = eeXH + ev1/sqrt(rXH**2+qaXH**2+aqeXH) - ev1dsqr6XH
    coreXH = torch.zeros(XH.sum(),5,dtype=dtype, device=device)
    # core(1,1), core(2,1), core(3,1), core(4,1),core(1,2)
    #nj=1
    #CORE(1,1) = TORE(NJ)*RI(1)
    #CORE(2,1) = TORE(NJ)*RI(2)
    #CORE(3,1) = TORE(NJ)*RI(3)
    #CORE(4,1) = TORE(NJ)*RI(4)
    #CORE(1,2) = TORE(NI)*RI(1)
    #CORE(1,1)
    coreXH[...,0] = tore[1]*riXH[...,1-1]
    #CORE(2,1)
    coreXH[...,1] = tore[1]*riXH[...,2-1]
    #CORE(3,1)
    coreXH[...,2] = tore[1]*riXH[...,3-1]
    #CORE(4,1)
    coreXH[...,3] = tore[1]*riXH[...,4-1]
    #CORE(1,2)
    coreXH[...,4] = tore[ni[XH]]*riXH[...,1-1]

    # Heavy atom - Heavy atom
    r =r0[XX]
    da = da0[XX]
    db = db0[XX]
    qa = qa0[XX]*2.0
    qb = qb0[XX]*2.0
    qa1 = qa0[XX]
    qb1 = qb0[XX]
    #sqr(54)-sqr(72) use qa1 and qb1
    ri =  torch.zeros(XX.sum(),22,dtype=dtype, device=device)
    core = torch.zeros(XX.sum(),8,dtype=dtype, device=device)

    #only the repeated terms are listed here
    aee = (rho0a[XX]+rho0b[XX])**2
    ade = (rho1a[XX]+rho0b[XX])**2
    aqe = (rho2a[XX]+rho0b[XX])**2
    aed = (rho0a[XX]+rho1b[XX])**2
    aeq = (rho0a[XX]+rho2b[XX])**2
    axx = (rho1a[XX]+rho1b[XX])**2
    adq = (rho1a[XX]+rho2b[XX])**2
    aqd = (rho2a[XX]+rho1b[XX])**2
    aqq = (rho2a[XX]+rho2b[XX])**2
    ee  = ev/sqrt(r**2+aee)
    dze = -ev1/sqrt((r+da)**2 + ade) + ev1/sqrt((r-da)**2 + ade)
    ev1dsqr6 = ev1/sqrt(r**2 + aqe)
    qzze = ev2/sqrt((r-qa)**2 + aqe) + ev2/sqrt((r+qa)**2 + aqe) - ev1dsqr6
    qxxe = ev1/sqrt(r**2 + qa**2 + aqe) - ev1dsqr6
    edz = -ev1/sqrt((r-db)**2 + aed) + ev1/sqrt((r+db)**2 + aed)
    ev1dsqr12 = ev1/sqrt(r**2 + aeq)
    eqzz = ev2/sqrt((r-qb)**2 + aeq) + ev2/sqrt((r+qb)**2 + aeq) -ev1dsqr12
    eqxx = ev1/sqrt(r**2 + qb**2 + aeq) - ev1dsqr12
    ev2dsqr20 = ev2/sqrt((r+da)**2 + adq)
    ev2dsqr22 = ev2/sqrt((r-da)**2 + adq)
    ev2dsqr24 = ev2/sqrt((r-db)**2 + aqd)
    ev2dsqr26 = ev2/sqrt((r+db)**2 + aqd)
    ev2dsqr36 = ev2/sqrt(r**2 + aqq)
    ev2dsqr39 = ev2/sqrt(r**2 + qa**2 + aqq)
    ev2dsqr40 = ev2/sqrt(r**2 + qb**2 + aqq)
    ev3dsqr42 = ev3/sqrt((r-qb)**2 + aqq)
    ev3dsqr44 = ev3/sqrt((r+qb)**2 + aqq)
    ev3dsqr46 = ev3/sqrt((r+qa)**2 + aqq)
    ev3dsqr48 = ev3/sqrt((r-qa)**2 + aqq)
    #all the index for ri is shfited by 1 to save space
    #C     (SS/SS)=1,   (SO/SS)=2,   (OO/SS)=3,   (PP/SS)=4,   (SS/OS)=5,
    #C     (SO/SO)=6,   (SP/SP)=7,   (OO/SO)=8,   (PP/SO)=9,   (PO/SP)=10,
    #C     (SS/OO)=11,  (SS/PP)=12,  (SO/OO)=13,  (SO/PP)=14,  (SP/OP)=15,
    #C     (OO/OO)=16,  (PP/OO)=17,  (OO/PP)=18,  (PP/PP)=19,  (PO/PO)=20,
    #C     (PP/P*P*)=21,   (P*P/P*P)=22
    ri[...,1-1] = ee
    ri[...,2-1] = -dze
    ri[...,3-1] = ee + qzze
    ri[...,4-1] = ee + qxxe
    ri[...,5-1] = -edz
    #RI(6) = DZDZ = EV2/SQR(16) + EV2/SQR(17) - EV2/SQR(18) - EV2/SQR(19)
    ri[...,6-1] = ev2/sqrt((r+da-db)**2 + axx) + ev2/sqrt((r-da+db)**2 + axx) \
                  - ev2/sqrt((r-da-db)**2 + axx) - ev2/sqrt((r+da+db)**2 + axx)
    #RI(7) = DXDX = EV1/SQR(14) - EV1/SQR(15)
    ri[...,7-1] = ev1/sqrt(r**2 + (da-db)**2 + axx) -ev1/sqrt(r**2 + (da+db)**2 + axx)
    #RI(8) = -EDZ -QZZDZ
    # QZZDZ = -EV3/SQR(32) + EV3/SQR(33) - EV3/SQR(34) + EV3/SQR(35)
    # + EV2/SQR(24) - EV2/SQR(26)
    ri[...,8-1] = -edz + ev3/sqrt((r+qa-db)**2 + aqd) - ev3/sqrt((r+qa+db)**2 + aqd) \
                + ev3/sqrt((r-qa-db)**2 + aqd) - ev3/sqrt((r-qa+db)**2 + aqd) \
                - ev2dsqr24 + ev2dsqr26
    #RI(9) = -EDZ -QXXDZ
    #QXXDZ =  EV2/SQR(24) - EV2/SQR(25) - EV2/SQR(26) + EV2/SQR(27)
    ri[...,9-1] = -edz - ev2dsqr24 + ev2/sqrt((r-db)**2 + qa**2 + aqd) \
                + ev2dsqr26 - ev2/sqrt((r+db)**2 + qa**2 + aqd)

    #sqr(54)-sqr(72) use qa1 and qb1
    #RI(10) = -QXZDX
    #QXZDX = -EV2/SQR(58) + EV2/SQR(59) + EV2/SQR(60) - EV2/SQR(61)
    ri[...,10-1] = ev2/sqrt((qa1-db)**2 + (r+qa1)**2 + aqd) \
                 - ev2/sqrt((qa1-db)**2 + (r-qa1)**2 + aqd) \
                 - ev2/sqrt((qa1+db)**2 + (r+qa1)**2 + aqd) \
                 + ev2/sqrt((qa1+db)**2 + (r-qa1)**2 + aqd)
    #RI(11) =  EE + EQZZ
    ri[...,11-1] = ee + eqzz
    #RI(12) =  EE + EQXX
    ri[...,12-1] = ee + eqxx
    #RI(13) = -DZE -DZQZZ
    #DZQZZ = -EV3/SQR(28) + EV3/SQR(29) - EV3/SQR(30) + EV3/SQR(31)
    #  - EV2/SQR(22) + EV2/SQR(20)
    ri[...,13-1] = -dze + ev3/sqrt((r+da-qb)**2 + adq) \
                 - ev3/sqrt((r-da-qb)**2 + adq) \
                 + ev3/sqrt((r+da+qb)**2 + adq) \
                 - ev3/sqrt((r-da+qb)**2 + adq) \
                 + ev2dsqr22 - ev2dsqr20
    #
    #RI(14) = -DZE -DZQXX
    #DZQXX =  EV2/SQR(20) - EV2/SQR(21) - EV2/SQR(22) + EV2/SQR(23)
    ri[...,14-1] = -dze - ev2dsqr20 + ev2/sqrt((r+da)**2 + qb**2 + adq) \
                 + ev2dsqr22 - ev2/sqrt((r-da)**2 + qb**2 + adq)
    #RI(15) = -DXQXZ
    #DXQXZ = -EV2/SQR(54) + EV2/SQR(55) + EV2/SQR(56) - EV2/SQR(57)
    #sqr(54)-sqr(72) use qa1 and qb1
    ri[...,15-1] = ev2/sqrt((da-qb1)**2 + (r-qb1)**2 + adq) \
                 - ev2/sqrt((da-qb1)**2 + (r+qb1)**2 + adq) \
                 - ev2/sqrt((da+qb1)**2 + (r-qb1)**2 + adq) \
                 + ev2/sqrt((da+qb1)**2 + (r+qb1)**2 + adq)
    #RI(16) = EE +EQZZ +QZZE +QZZQZZ
    #QZZQZZ = EV4/SQR(50) + EV4/SQR(51) + EV4/SQR(52) + EV4/SQR(53)
    # - EV3/SQR(48) - EV3/SQR(46) - EV3/SQR(42) - EV3/SQR(44)
    # + EV2/SQR(36)
    ri[...,16-1] = ee + eqzz + qzze \
                 + ev4/sqrt((r+qa-qb)**2 + aqq) \
                 + ev4/sqrt((r+qa+qb)**2 + aqq) \
                 + ev4/sqrt((r-qa-qb)**2 + aqq) \
                 + ev4/sqrt((r-qa+qb)**2 + aqq) \
                 - ev3dsqr48 - ev3dsqr46 -ev3dsqr42 - ev3dsqr44 + ev2dsqr36
    #RI(17) = EE +EQZZ +QXXE +QXXQZZ
    #QXXQZZ = EV3/SQR(43) + EV3/SQR(45) - EV3/SQR(42) - EV3/SQR(44)
    #  - EV2/SQR(39) + EV2/SQR(36)
    ri[...,17-1] = ee + eqzz + qxxe \
                 + ev3/sqrt((r-qb)**2 + qa**2 + aqq) \
                 + ev3/sqrt((r+qb)**2 + qa**2 + aqq) \
                 - ev3dsqr42 - ev3dsqr44 - ev2dsqr39 + ev2dsqr36
    #RI(18) = EE +EQXX +QZZE +QZZQXX
    #QZZQXX = EV3/SQR(47) + EV3/SQR(49) - EV3/SQR(46) - EV3/SQR(48)
    #  - EV2/SQR(40) + EV2/SQR(36)
    ri[...,18-1] = ee + eqxx + qzze \
                 + ev3/sqrt((r+qa)**2 + qb**2 + aqq) \
                 + ev3/sqrt((r-qa)**2 + qb**2 + aqq) \
                 - ev3dsqr46 - ev3dsqr48 - ev2dsqr40 + ev2dsqr36
    #RI(19) = EE +EQXX +QXXE +QXXQXX
    #QXXQXX = EV3/SQR(37) + EV3/SQR(38) - EV2/SQR(39) - EV2/SQR(40)
    # + EV2/SQR(36)
    qxxqxx = ev3/sqrt(r**2 + (qa-qb)**2 + aqq) \
           + ev3/sqrt(r**2 + (qa+qb)**2 + aqq) \
           - ev2dsqr39 - ev2dsqr40 + ev2dsqr36
    ri[...,19-1] = ee + eqxx + qxxe + qxxqxx
    #RI(20) = QXZQXZ
    #QXZQXZ = EV3/SQR(65) - EV3/SQR(67) - EV3/SQR(69) + EV3/SQR(71)
    #- EV3/SQR(66) + EV3/SQR(68) + EV3/SQR(70) - EV3/SQR(72)
    #sqr(54)-sqr(72) use qa1 and qb1
    ri[...,20-1] = ev3/sqrt((r+qa1-qb1)**2 + (qa1-qb1)**2 + aqq) \
                 - ev3/sqrt((r+qa1+qb1)**2 + (qa1-qb1)**2 + aqq) \
                 - ev3/sqrt((r-qa1-qb1)**2 + (qa1-qb1)**2 + aqq) \
                 + ev3/sqrt((r-qa1+qb1)**2 + (qa1-qb1)**2 + aqq) \
                 - ev3/sqrt((r+qa1-qb1)**2 + (qa1+qb1)**2 + aqq) \
                 + ev3/sqrt((r+qa1+qb1)**2 + (qa1+qb1)**2 + aqq) \
                 + ev3/sqrt((r-qa1-qb1)**2 + (qa1+qb1)**2 + aqq) \
                 - ev3/sqrt((r-qa1+qb1)**2 + (qa1+qb1)**2 + aqq)
    # RI(21) = EE +EQXX +QXXE +QXXQYY
    #QXXQYY = EV2/SQR(41) - EV2/SQR(39) - EV2/SQR(40) + EV2/SQR(36)
    qxxqyy = ev2/sqrt(r**2 + qa**2 + qb**2 + aqq) \
           - ev2dsqr39 - ev2dsqr40 + ev2dsqr36
    ri[...,21-1] = ee + eqxx + qxxe + qxxqyy
    #RI(22) = PP * (QXXQXX -QXXQYY)
    ri[...,22-1] = 0.5 * (qxxqxx - qxxqyy)

    # CALCULATE CORE-ELECTRON ATTRACTIONS.
    #CORE(1,1) = TORE(NJ)*RI(1)
    #CORE(2,1) = TORE(NJ)*RI(2)
    #CORE(3,1) = TORE(NJ)*RI(3)
    #CORE(4,1) = TORE(NJ)*RI(4)
    #CORE(1,2) = TORE(NI)*RI(1)
    #CORE(2,2) = TORE(NI)*RI(5)
    #CORE(3,2) = TORE(NI)*RI(11)
    #CORE(4,2) = TORE(NI)*RI(12)
    core[...,0] =  tore[nj[XX]]*ri[...,1-1]
    core[...,1] =  tore[nj[XX]]*ri[...,2-1]
    core[...,2] =  tore[nj[XX]]*ri[...,3-1]
    core[...,3] =  tore[nj[XX]]*ri[...,4-1]
    core[...,4] =  tore[ni[XX]]*ri[...,1-1]
    core[...,5] =  tore[ni[XX]]*ri[...,5-1]
    core[...,6] =  tore[ni[XX]]*ri[...,11-1]
    core[...,7] =  tore[ni[XX]]*ri[...,12-1]
    return riHH, riXH, ri, coreHH, coreXH, core
