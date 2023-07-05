import torch as th

def diatom_overlap_matrixD(ni,nj, xij, rij, zeta_a, zeta_b, qn_int, qnD_int):
    """
    compute the overlap matrix for each pair
    """
    #t0 =time.time()
    dtype = xij.dtype
    device = xij.device
    # xij unit vector points from i to j: xij = (xj-xi)/|xi-xj|, shape (npairs, 3)
    #rij : distance between atom i and j in atomic units, shape (npairs,)
    #ni, nj, atomic number of atom i and j, shape (npairs,)
    # x = x2-x1
    # y = y2-y1
    # z = z2-z1
    # for first atom index == second atom index, (same atom)
    # x = x1, y = y1, z = z1
    npairs=xij.shape[0]
    # zeta_a, zeta_b: zeta_s and zeta_p for atom pair a-b, shape(npairs, 2)
    # output di: overlap matrix between AOs from atom i an j, shape (npairs, 4,4)
    # 0,1,2,3: sigma, px, py, pz


    xy = th.norm(xij[...,:2],dim=1)


    tmp = th.where(xij[...,2]<0.0,th.tensor(-1.0,dtype=dtype, device=device), \
          th.where(xij[...,2]>0.0,th.tensor(1.0,dtype=dtype, device=device), \
          th.tensor(0.0,dtype=dtype, device=device)))

    cond_xy = xy>=1.0e-10
    ca = tmp.clone()
    ca[cond_xy] = xij[cond_xy,0]/xy[cond_xy]

    cb = th.where(xy>=1.0e-10, xij[...,2], tmp)  #xij is a unti vector already
    sa = th.zeros_like(xy)

    sa[cond_xy] = xij[cond_xy,1]/xy[cond_xy]
    #sb = th.where(xy>=1.0e-10, xy/rij, th.tensor(0.0,dtype=dtype))
    sb = th.where(xy>=1.0e-10, xy, th.tensor(0.0,dtype=dtype, device=device))
    ################################
    #ok to use th.where here as postion doesn't require grad
    #if update to do MD, ca, cb, sa, sb should be chaneged to the indexing version
    ################################

    #overlap matrix in the local frame
    # first row  - first row  : ii = 1, jcall = 2
    # first row  - second row : ii = 2, jcall = 3
    # second row - secpmd row : ii = 4, jcall = 4

    #only first and second row are included here
    #one-element slice to keep dim
    #jcall shape (npairs, )

    jcall = th.zeros_like(ni)
    jcallds = th.zeros_like(ni)
    jcallsd = th.zeros_like(ni)
    jcalldd = th.zeros_like(ni)
    qni = qn_int[ni]
    qnj = qn_int[nj]
    dqni = qnD_int[ni]
    dqnj = qnD_int[nj]
    jcall[(qni==1) & (qnj==1)] = 2
    jcall[(qni==2) & (qnj==1)] = 3
    jcall[(qni==2) & (qnj==2)] = 4


    jcall[(qni==3) & (qnj==1)] = 431
    jcall[(qni==3) & (qnj==2)] = 5
    jcall[(qni==3) & (qnj==3)] = 6


    jcall[(qni==4) & (qnj==1)] = 541
    jcall[(qni==4) & (qnj==2)] = 642
    jcall[(qni==4) & (qnj==3)] = 7
    jcall[(qni==4) & (qnj==4)] = 8

    jcall[(qni==5) & (qnj==1)] = 651
    jcall[(qni==5) & (qnj==2)] = 752
    jcall[(qni==5) & (qnj==3)] = 853
    jcall[(qni==5) & (qnj==4)] = 9
    jcall[(qni==5) & (qnj==5)] = 10

    jcall[(qni==6) & (qnj==1)] = 761
    jcall[(qni==6) & (qnj==2)] = 862
    jcall[(qni==6) & (qnj==3)] = 963
    jcall[(qni==6) & (qnj==4)] = 1064
    jcall[(qni==6) & (qnj==5)] = 11
    jcall[(qni==6) & (qnj==6)] = 12



    jcallds[(dqni==3) & (qnj==1)] = 431
    jcallds[(dqni==3) & (qnj==2)] = 5
    jcallds[(dqni==3) & (qnj==3)] = 6
    jcallds[(dqni==3) & (qnj==4)] = 734


    jcallds[(dqni==4) & (qnj==1)] = 541
    jcallds[(dqni==4) & (qnj==2)] = 642
    jcallds[(dqni==4) & (qnj==3)] = 7
    jcallds[(dqni==4) & (qnj==4)] = 8
    jcallds[(dqni==4) & (qnj==5)] = 945

    jcallds[(dqni==5) & (qnj==1)] = 651
    jcallds[(dqni==5) & (qnj==2)] = 752
    jcallds[(dqni==5) & (qnj==3)] = 853
    jcallds[(dqni==5) & (qnj==4)] = 9
    jcallds[(dqni==5) & (qnj==5)] = 10
    jcallds[(dqni==5) & (qnj==6)] = 11

    
    jcallsd[(qni==3) & (dqnj==3)] = 6


    jcallsd[(qni==4) & (dqnj==3)] = 7
    jcallsd[(qni==4) & (dqnj==4)] = 8

    jcallsd[(qni==5) & (dqnj==3)] = 853
    jcallsd[(qni==5) & (dqnj==4)] = 9
    jcallsd[(qni==5) & (dqnj==5)] = 10

    jcallsd[(qni==6) & (dqnj==3)] = 963
    jcallsd[(qni==6) & (dqnj==4)] = 1064
    jcallsd[(qni==6) & (dqnj==5)] = 11
    jcallsd[(qni==6) & (dqnj==6)] = 12


    jcalldd[(dqni==3) & (dqnj==3)] = 6


    jcalldd[(dqni==4) & (dqnj==3)] = 7
    jcalldd[(dqni==4) & (dqnj==4)] = 8

    jcalldd[(dqni==5) & (dqnj==3)] = 853
    jcalldd[(dqni==5) & (dqnj==4)] = 9
    jcalldd[(dqni==5) & (dqnj==5)] = 10




    if th.any(jcall==0):
        raise ValueError("\nError from diat.py, overlap matrix\nSome elements are not supported yet")


    #na>=nb
    #setc.isp=2
    #setc.ips=1
    #setc.sa = s2 = zeta_B
    #setc.sb = s1 = zeta_A

    #change to atomic units, a0 here is taken from mopac, different from the standard value
    # r is already in unit of bohr radius, AU



    #parameter_set = ['U_ss', 'U_pp', 'zeta_s', 'zeta_p','beta_s', 'beta_p',
    #                 'g_ss', 'g_sp', 'g_pp', 'g_p2', 'h_sp', 'alpha']


    A111,B111 = SET(rij, jcall, zeta_a[...,0],zeta_b[...,0])
    S111 = th.zeros((npairs,),dtype=dtype, device=device)

    S211=th.zeros_like(S111)
    A211, B211 = SET(rij, jcall, zeta_a[...,1],zeta_b[...,0])

    S121 = th.zeros_like(S111)
    A121, B121 = SET(rij, jcall, zeta_a[...,0],zeta_b[...,1])

    S221 = th.zeros_like(S111)
    A22, B22 = SET(rij, jcall, zeta_a[...,1],zeta_b[...,1])

    S222 = th.zeros_like(S221)

    #print(id(S111))
    # s-s
    jcall2 = (jcall==2) #ii=1
    if(jcall2.sum() != 0):
        S111[jcall2] = th.pow(zeta_a[jcall2,0]* \
                          zeta_b[jcall2,0]* \
                          rij[jcall2]**2,1.5)* \
                  (A111[jcall2,2]*B111[jcall2,0]-B111[jcall2,2]*A111[jcall2,0])/4.0
    jcall3 = (jcall==3) #ii=2
    if(jcall3.sum() != 0):
        S111[jcall3] = th.pow(zeta_b[jcall3,0],1.5)* \
                   th.pow(zeta_a[jcall3,0],2.5)* \
                   rij[jcall3]**4 * \
                  (A111[jcall3,3]*B111[jcall3,0]-B111[jcall3,3]*A111[jcall3,0]+ \
                   A111[jcall3,2]*B111[jcall3,1]-B111[jcall3,2]*A111[jcall3,1])/ \
                  (th.sqrt(th.tensor(3.0))*8.0)
        S211[jcall3] = th.pow(zeta_b[jcall3,0],1.5)* \
                   th.pow(zeta_a[jcall3,1],2.5)* \
                   rij[jcall3]**4 * \
                  (A211[jcall3,2]*B211[jcall3,0]-B211[jcall3,2]*A211[jcall3,0]+ \
                   A211[jcall3,3]*B211[jcall3,1]-B211[jcall3,3]*A211[jcall3,1])/8.0

    jcall4 = (jcall==4) #ii=4
    if(jcall4.sum() != 0):
        S111[jcall4] = th.pow(zeta_b[jcall4,0]* \
                          zeta_a[jcall4,0],2.5)* \
                   rij[jcall4]**5 * \
                  (A111[jcall4,4]*B111[jcall4,0]+B111[jcall4,4]*A111[jcall4,0] \
                   -2.0*A111[jcall4,2]*B111[jcall4,2])/48.0
        S211[jcall4] = th.pow(zeta_b[jcall4,0]* \
                          zeta_a[jcall4,1],2.5)* \
                   rij[jcall4]**5 * \
                  (A211[jcall4,3]*(B211[jcall4,0]-B211[jcall4,2]) \
                  -A211[jcall4,1]*(B211[jcall4,2]-B211[jcall4,4]) \
                  +B211[jcall4,3]*(A211[jcall4,0]-A211[jcall4,2]) \
                  -B211[jcall4,1]*(A211[jcall4,2]-A211[jcall4,4])) \
                  /(16.0*th.sqrt(th.tensor(3.0)))
        S121[jcall4] = th.pow(zeta_b[jcall4,1]* \
                          zeta_a[jcall4,0],2.5)* \
                   rij[jcall4]**5 * \
                  (A121[jcall4,3]*(B121[jcall4,0]-B121[jcall4,2]) \
                  -A121[jcall4,1]*(B121[jcall4,2]-B121[jcall4,4]) \
                  -B121[jcall4,3]*(A121[jcall4,0]-A121[jcall4,2]) \
                  +B121[jcall4,1]*(A121[jcall4,2]-A121[jcall4,4])) \
                   /(16.0*th.sqrt(th.tensor(3.0)))
        w = th.pow(zeta_b[jcall4,1]* \
               zeta_a[jcall4,1],2.5)* \
        rij[jcall4]**5/16.0
        S221[jcall4] = -w* \
                  (B22[jcall4,2]*(A22[jcall4,4]+A22[jcall4,0]) \
                  -A22[jcall4,2]*(B22[jcall4,4]+B22[jcall4,0]))
        S222[jcall4] = 0.5*w* \
                  (A22[jcall4,4]*(B22[jcall4,0]-B22[jcall4,2]) \
                  -B22[jcall4,4]*(A22[jcall4,0]-A22[jcall4,2]) \
                  -A22[jcall4,2]*B22[jcall4,0]+B22[jcall4,2]*A22[jcall4,0])




    jcall431 = (jcall==431) #ii=4
    if(jcall431.sum() != 0):
        S111[jcall431] = th.pow(zeta_b[jcall431,0],1.5)* \
                   th.pow(zeta_a[jcall431,0],3.5)* \
                   rij[jcall431]**5 * \
                  (A111[jcall431,4]*B111[jcall431,0]+2*B111[jcall431,1]*A111[jcall431,3]- \
                   2.0*A111[jcall431,1]*B111[jcall431,3]-B111[jcall431,4]*A111[jcall431,0])/ \
                  (th.sqrt(th.tensor(10.0))*24.0)
        S211[jcall431] = th.pow(zeta_b[jcall431,0],1.5)* \
                   th.pow(zeta_a[jcall431,1],3.5)* \
                   rij[jcall431]**5 * \
                  (A211[jcall431,3]*(B211[jcall431,0]+B211[jcall431,2])-A211[jcall431,1]* \
                   (B211[jcall431,4]+B211[jcall431,2])+B211[jcall431,1]*(A211[jcall431,2]+A211[jcall431,4])-B211[jcall431,3]*(A211[jcall431,2]+A211[jcall431,0]))/(8.0*th.sqrt(th.tensor(30.0)))




    jcall5 = (jcall==5) #ii=4
    if(jcall5.sum() != 0):
        S111[jcall5] = th.pow(zeta_b[jcall5,0],2.5)* \
                   th.pow(zeta_a[jcall5,0],3.5)* \
                   rij[jcall5]**6 * \
                  (A111[jcall5,5]*B111[jcall5,0]+B111[jcall5,1]*A111[jcall5,4]-2*B111[jcall5,2]*A111[jcall5,3]- \
                   2.0*A111[jcall5,2]*B111[jcall5,3]+B111[jcall5,4]*A111[jcall5,1]+B111[jcall5,5]*A111[jcall5,0])/ \
                  (th.sqrt(th.tensor(30.0))*48.0)
        S211[jcall5] = th.pow(zeta_b[jcall5,0],2.5)* \
                   th.pow(zeta_a[jcall5,1],3.5)* \
                   rij[jcall5]**6 * \
                  (A211[jcall5,4]*B211[jcall5,0]+B211[jcall5,1]*A211[jcall5,5]- \
                   2.0*B211[jcall5,3]*A211[jcall5,3]-2.0*A211[jcall5,2]*B211[jcall5,2]+A211[jcall5,1]*B211[jcall5,5]+A211[jcall5,0]*B211[jcall5,4])/(48.0*th.sqrt(th.tensor(10.0)))
        S121[jcall5] = th.pow(zeta_b[jcall5,1],2.5)* \
                   th.pow(zeta_a[jcall5,0],3.5)* \
                   rij[jcall5]**6 * \
                  ((A121[jcall5,4]*B121[jcall5,0]-A121[jcall5,5]*B121[jcall5,1]) + \
                  2.0*(A121[jcall5,3]*B121[jcall5,1]-A121[jcall5,4]*B121[jcall5,2]) - \
                  2.0*(A121[jcall5,1]*B121[jcall5,3]-A121[jcall5,2]*B121[jcall5,4]) - \
                  (A121[jcall5,0]*B121[jcall5,4]-A121[jcall5,1]*B121[jcall5,5]))/(48.0*th.sqrt(th.tensor(10.0)))
        S221[jcall5] = th.pow(zeta_b[jcall5,1],2.5)* \
                   th.pow(zeta_a[jcall5,1],3.5)* \
                   rij[jcall5]**6 * \
                  ((A22[jcall5,3]*B22[jcall5,0]-A22[jcall5,5]*B22[jcall5,2]) + \
                  (A22[jcall5,2]*B22[jcall5,1]-A22[jcall5,4]*B22[jcall5,3]) - \
                  (A22[jcall5,1]*B22[jcall5,2]-A22[jcall5,3]*B22[jcall5,4]) - \
                  (A22[jcall5,0]*B22[jcall5,3]-A22[jcall5,2]*B22[jcall5,5])) \
                  /(16.0*th.sqrt(th.tensor(30.0)))
        S222[jcall5] = th.pow(zeta_b[jcall5,1],2.5) * \
                   th.pow(zeta_a[jcall5,1],3.5)* \
                   rij[jcall5]**6 * \
                  ((A22[jcall5,5]-A22[jcall5,3])*(B22[jcall5,0]-B22[jcall5,2]) + \
                  (A22[jcall5,4]-A22[jcall5,2])*(B22[jcall5,1]-B22[jcall5,3]) - \
                  (A22[jcall5,3]-A22[jcall5,1])*(B22[jcall5,2]-B22[jcall5,4]) - \
                  (A22[jcall5,2]-A22[jcall5,0])*(B22[jcall5,3]-B22[jcall5,5])) \
                  /(32.0*th.sqrt(th.tensor(30.0)))






    jcall6 = (jcall==6)
    if(jcall6.sum() != 0):
        S111[jcall6] = th.pow(zeta_b[jcall6,0]* \
                          zeta_a[jcall6,0],3.5)* \
                   rij[jcall6]**7 * \
                  (A111[jcall6,6]*B111[jcall6,0]-3.0*B111[jcall6,2]*A111[jcall6,4] \
                   +3.0*A111[jcall6,2]*B111[jcall6,4]-A111[jcall6,0]*B111[jcall6,6])/1440.0
        S211[jcall6] = th.pow(zeta_b[jcall6,0]* \
                          zeta_a[jcall6,1],3.5)* \
                   rij[jcall6]**7 * \
                  ((A211[jcall6,5]*B211[jcall6,0]+A211[jcall6,6]*B211[jcall6,1]) + \
                  (-A211[jcall6,4]*B211[jcall6,1]-A211[jcall6,5]*B211[jcall6,2]) - \
                  2.0*(A211[jcall6,3]*B211[jcall6,2]+A211[jcall6,4]*B211[jcall6,3]) - \
                  2.0*(-A211[jcall6,2]*B211[jcall6,3]-A211[jcall6,3]*B211[jcall6,4]) + \
                  (A211[jcall6,1]*B211[jcall6,4]+A211[jcall6,2]*B211[jcall6,5]) + \
                  (-A211[jcall6,0]*B211[jcall6,5]-A211[jcall6,1]*B211[jcall6,6]))  \
                  /(480.0*th.sqrt(th.tensor(3.0)))
        S121[jcall6] = th.pow(zeta_b[jcall6,1]* \
                          zeta_a[jcall6,0],3.5)* \
                   rij[jcall6]**7 * \
                  ((A121[jcall6,5]*B121[jcall6,0]-A121[jcall6,6]*B121[jcall6,1]) + \
                  (A121[jcall6,4]*B121[jcall6,1]-A121[jcall6,5]*B121[jcall6,2]) - \
                  2.0*(A121[jcall6,3]*B121[jcall6,2]-A121[jcall6,4]*B121[jcall6,3]) - \
                  2.0*(A121[jcall6,2]*B121[jcall6,3]-A121[jcall6,3]*B121[jcall6,4]) + \
                  (A121[jcall6,1]*B121[jcall6,4]-A121[jcall6,2]*B121[jcall6,5]) + \
                  (A121[jcall6,0]*B121[jcall6,5]-A121[jcall6,1]*B121[jcall6,6]))  \
                  /(480.0*th.sqrt(th.tensor(3.0)))
        S221[jcall6] = th.pow(zeta_b[jcall6,1],3.5)* \
                   th.pow(zeta_a[jcall6,1],3.5)* \
                   rij[jcall6]**7 * \
                  ((A22[jcall6,4]*B22[jcall6,0]-A22[jcall6,6]*B22[jcall6,2]) - \
                  2.0*(A22[jcall6,2]*B22[jcall6,2]-A22[jcall6,4]*B22[jcall6,4]) + \
                  (A22[jcall6,0]*B22[jcall6,4]-A22[jcall6,2]*B22[jcall6,6])) \
                  /(480.0)
        S222[jcall6] = th.pow(zeta_b[jcall6,1],3.5)* \
                   th.pow(zeta_a[jcall6,1],3.5)* \
                   rij[jcall6]**7 * \
                  ((A22[jcall6,6]-A22[jcall6,4])*(B22[jcall6,0]-B22[jcall6,2]) - \
                  2.0*(A22[jcall6,4]-A22[jcall6,2])*(B22[jcall6,2]-B22[jcall6,4]) + \
                  (A22[jcall6,2]-A22[jcall6,0])*(B22[jcall6,4]-B22[jcall6,6])) \
                  /(960.0)



###up to here



    jcall541 = (jcall==541) #ii=4
    if(jcall541.sum() != 0):
        S111[jcall541] = th.pow(zeta_b[jcall541,0],1.5)* \
                   th.pow(zeta_a[jcall541,0],4.5)* \
                   rij[jcall541]**6 * \
                  (A111[jcall541,5]*B111[jcall541,0]+3.0*B111[jcall541,1]*A111[jcall541,4]+2.0*B111[jcall541,2]*A111[jcall541,3]- \
                   2.0*B111[jcall541,3]*A111[jcall541,2]-3.0*A111[jcall541,1]*B111[jcall541,4]-B111[jcall541,5]*A111[jcall541,0])/ \
                  (th.sqrt(th.tensor(35.0))*96.0)
        S211[jcall541] = th.pow(zeta_b[jcall541,0],1.5)* \
                   th.pow(zeta_a[jcall541,1],4.5)* \
                   rij[jcall541]**6 * \
                  ((A211[jcall541,4]*B211[jcall541,0]+A211[jcall541,5]*B211[jcall541,1]) - \
                  2.0*(-A211[jcall541,3]*B211[jcall541,1]-A211[jcall541,4]*B211[jcall541,2]) + \
                  2.0*(-A211[jcall541,1]*B211[jcall541,3]-A211[jcall541,2]*B211[jcall541,4]) - \
                  (A211[jcall541,0]*B211[jcall541,4]+A211[jcall541,1]*B211[jcall541,5])) \
                   /(32.0*th.sqrt(th.tensor(105.0)))


    jcall642 = (jcall==642) #ii=4
    if(jcall642.sum() != 0):
        S111[jcall642] = th.pow(zeta_b[jcall642,0],2.5)* \
                   th.pow(zeta_a[jcall642,0],4.5)* \
                   rij[jcall642]**7 * \
                  (A111[jcall642,6]*B111[jcall642,0]+2.0*B111[jcall642,1]*A111[jcall642,5]-1.0*B111[jcall642,2]*A111[jcall642,4]- \
                   4.0*B111[jcall642,3]*A111[jcall642,3]-1.0*A111[jcall642,2]*B111[jcall642,4]+2.0*B111[jcall642,5]*A111[jcall642,1]+\
                   A111[jcall642,0]*B111[jcall642,6])/ \
                  (th.sqrt(th.tensor(105.0))*192.0)
        S211[jcall642] = th.pow(zeta_b[jcall642,0],2.5)* \
                   th.pow(zeta_a[jcall642,1],4.5)* \
                   rij[jcall642]**7 * \
                  ((A211[jcall642,5]*B211[jcall642,0]+A211[jcall642,6]*B211[jcall642,1]) - \
                  (-A211[jcall642,4]*B211[jcall642,1]-A211[jcall642,5]*B211[jcall642,2]) - \
                  2.0*(A211[jcall642,3]*B211[jcall642,2]+A211[jcall642,4]*B211[jcall642,3]) + \
                  2.0*(-A211[jcall642,2]*B211[jcall642,3]-A211[jcall642,3]*B211[jcall642,4]) + \
                  (A211[jcall642,1]*B211[jcall642,4]+A211[jcall642,2]*B211[jcall642,5]) - \
                  (-A211[jcall642,0]*B211[jcall642,5]-A211[jcall642,1]*B211[jcall642,6])) \
                  /(192.0*th.sqrt(th.tensor(35.0)))
        S121[jcall642] = th.pow(zeta_b[jcall642,1],2.5)* \
                   th.pow(zeta_a[jcall642,0],4.5)* \
                   rij[jcall642]**7 * \
                  ((A121[jcall642,5]*B121[jcall642,0]-A121[jcall642,6]*B121[jcall642,1]) + \
                  3.0*(A121[jcall642,4]*B121[jcall642,1]-A121[jcall642,5]*B121[jcall642,2]) + \
                  2.0*(A121[jcall642,3]*B121[jcall642,2]-A121[jcall642,4]*B121[jcall642,3]) - \
                  2.0*(A121[jcall642,2]*B121[jcall642,3]-A121[jcall642,3]*B121[jcall642,4]) - \
                  3.0*(A121[jcall642,1]*B121[jcall642,4]-A121[jcall642,2]*B121[jcall642,5]) - \
                  (A121[jcall642,0]*B121[jcall642,5]-A121[jcall642,1]*B121[jcall642,6])) \
                  /(192.0*th.sqrt(th.tensor(35.0)))
        S221[jcall642] = th.pow(zeta_b[jcall642,1],2.5)* \
                   th.pow(zeta_a[jcall642,1],4.5)* \
                   rij[jcall642]**7 * \
                  ((A22[jcall642,4]*B22[jcall642,0]-A22[jcall642,6]*B22[jcall642,2]) + \
                  2.0*(A22[jcall642,3]*B22[jcall642,1]-A22[jcall642,5]*B22[jcall642,3]) - \
                  2.0*(A22[jcall642,1]*B22[jcall642,3]-A22[jcall642,3]*B22[jcall642,5]) - \
                  (A22[jcall642,0]*B22[jcall642,4]-A22[jcall642,2]*B22[jcall642,6])) \
                  /(64.0*th.sqrt(th.tensor(105.0)))
        S222[jcall642] = th.pow(zeta_b[jcall642,1],2.5)* \
                   th.pow(zeta_a[jcall642,1],4.5)* \
                   rij[jcall642]**7 * \
                  ((A22[jcall642,6]-A22[jcall642,4])*(B22[jcall642,0]-B22[jcall642,2]) + \
                  2.0*(A22[jcall642,5]-A22[jcall642,3])*(B22[jcall642,1]-B22[jcall642,3]) - \
                  2.0*(A22[jcall642,3]-A22[jcall642,1])*(B22[jcall642,3]-B22[jcall642,5]) - \
                  (A22[jcall642,2]-A22[jcall642,0])*(B22[jcall642,4]-B22[jcall642,6])) \
                  /(128.0*th.sqrt(th.tensor(105.0)))






    jcall7 = (jcall==7) #ii=4
    if(jcall7.sum() != 0):
        S111[jcall7] = th.pow(zeta_b[jcall7,0],3.5)* \
                   th.pow(zeta_a[jcall7,0],4.5)* \
                   rij[jcall7]**8 * \
                  (A111[jcall7,7]*B111[jcall7,0]+1.0*B111[jcall7,1]*A111[jcall7,6]-3.0*B111[jcall7,2]*A111[jcall7,5]- \
                   3.0*B111[jcall7,3]*A111[jcall7,4]+3.0*A111[jcall7,3]*B111[jcall7,4]+3.0*B111[jcall7,5]*A111[jcall7,2]- \
                   A111[jcall7,1]*B111[jcall7,6]-A111[jcall7,0]*B111[jcall7,7])/ \
                  (th.sqrt(th.tensor(14.0))*2880.0)
        S211[jcall7] = th.pow(zeta_b[jcall7,0],3.5)* \
                   th.pow(zeta_a[jcall7,1],4.5)* \
                   rij[jcall7]**8 * \
                  (A211[jcall7,7]*B211[jcall7,1]+B211[jcall7,0]*A211[jcall7,6]- \
                   3.0*B211[jcall7,3]*A211[jcall7,5]-3.0*A211[jcall7,4]*B211[jcall7,2]+3.0*A211[jcall7,3]*B211[jcall7,5]+3.0*A211[jcall7,2]*B211[jcall7,4]- \
                   B211[jcall7,7]*A211[jcall7,1]- B211[jcall7,6]*A211[jcall7,0])/(960.0*th.sqrt(th.tensor(42.0)))
        S121[jcall7] = th.pow(zeta_b[jcall7,1],3.5)* \
                   th.pow(zeta_a[jcall7,0],4.5)* \
                   rij[jcall7]**8 * \
                  ((A121[jcall7,6]*B121[jcall7,0]-A121[jcall7,7]*B121[jcall7,1]) + \
                  2.0*(A121[jcall7,5]*B121[jcall7,1]-A121[jcall7,6]*B121[jcall7,2]) - \
                  (A121[jcall7,4]*B121[jcall7,2]-A121[jcall7,5]*B121[jcall7,3]) - \
                  4.0*(A121[jcall7,3]*B121[jcall7,3]-A121[jcall7,4]*B121[jcall7,4]) - \
                  (A121[jcall7,2]*B121[jcall7,4]-A121[jcall7,3]*B121[jcall7,5]) + \
                  2.0*(A121[jcall7,1]*B121[jcall7,5]-A121[jcall7,2]*B121[jcall7,6]) + \
                  (A121[jcall7,0]*B121[jcall7,6]-A121[jcall7,1]*B121[jcall7,7])) \
                   /(960.0*th.sqrt(th.tensor(42.0)))
        S221[jcall7] = th.pow(zeta_b[jcall7,1],3.5)* \
                   th.pow(zeta_a[jcall7,1],4.5)* \
                   rij[jcall7]**8 * \
                  ((A22[jcall7,5]*B22[jcall7,0]-A22[jcall7,7]*B22[jcall7,2]) + \
                  (A22[jcall7,4]*B22[jcall7,1]-A22[jcall7,6]*B22[jcall7,3]) - \
                  2.0*(A22[jcall7,3]*B22[jcall7,2]-A22[jcall7,5]*B22[jcall7,4]) - \
                  2.0*(A22[jcall7,2]*B22[jcall7,3]-A22[jcall7,4]*B22[jcall7,5]) + \
                  (A22[jcall7,1]*B22[jcall7,4]-A22[jcall7,3]*B22[jcall7,6]) + \
                  (A22[jcall7,0]*B22[jcall7,5]-A22[jcall7,2]*B22[jcall7,7])) \
                  /(960.0*th.sqrt(th.tensor(14.0)))
        S222[jcall7] = th.pow(zeta_b[jcall7,1],3.5)* \
                   th.pow(zeta_a[jcall7,1],4.5)* \
                   rij[jcall7]**8 * \
                  ((A22[jcall7,7]-A22[jcall7,5])*(B22[jcall7,0]-B22[jcall7,2]) + \
                  (A22[jcall7,6]-A22[jcall7,4])*(B22[jcall7,1]-B22[jcall7,3]) - \
                  2.0*(A22[jcall7,5]-A22[jcall7,3])*(B22[jcall7,2]-B22[jcall7,4]) - \
                  2.0*(A22[jcall7,4]-A22[jcall7,2])*(B22[jcall7,3]-B22[jcall7,5]) + \
                  (A22[jcall7,3]-A22[jcall7,1])*(B22[jcall7,4]-B22[jcall7,6]) + \
                  (A22[jcall7,2]-A22[jcall7,0])*(B22[jcall7,5]-B22[jcall7,7])) \
                  /(1920.0*th.sqrt(th.tensor(14.0)))







    jcall8 = (jcall==8)
    if(jcall8.sum() != 0):
        S111[jcall8] = th.pow(zeta_b[jcall8,0]* \
                          zeta_a[jcall8,0],4.5)* \
                   rij[jcall8]**9 * \
                  (A111[jcall8,8]*B111[jcall8,0]-4.0*B111[jcall8,2]*A111[jcall8,6]+6.0*A111[jcall8,4]*B111[jcall8,4] \
                   -4.0*A111[jcall8,2]*B111[jcall8,6]+A111[jcall8,0]*B111[jcall8,8])/80640.0
        S211[jcall8] = th.pow(zeta_b[jcall8,0],4.5)* \
                   th.pow(zeta_a[jcall8,1],4.5)* \
                   rij[jcall8]**9 * \
                  ((A211[jcall8,7]*B211[jcall8,0]+A211[jcall8,8]*B211[jcall8,1]) + \
                  (-A211[jcall8,6]*B211[jcall8,1]-A211[jcall8,7]*B211[jcall8,2]) - \
                  3.0*(A211[jcall8,5]*B211[jcall8,2]+A211[jcall8,6]*B211[jcall8,3]) - \
                  3.0*(-A211[jcall8,4]*B211[jcall8,3]-A211[jcall8,5]*B211[jcall8,4]) + \
                  3.0*(A211[jcall8,3]*B211[jcall8,4]+A211[jcall8,4]*B211[jcall8,5]) + \
                  3.0*(-A211[jcall8,2]*B211[jcall8,5]-A211[jcall8,3]*B211[jcall8,6]) - \
                  (A211[jcall8,1]*B211[jcall8,6]+A211[jcall8,2]*B211[jcall8,7]) - \
                  (-A211[jcall8,0]*B211[jcall8,7]-A211[jcall8,1]*B211[jcall8,8])) \
                  /(26880.0*th.sqrt(th.tensor(3.0)))
        S121[jcall8] = th.pow(zeta_b[jcall8,1],4.5)* \
                   th.pow(zeta_a[jcall8,0],4.5)* \
                   rij[jcall8]**9 * \
                  ((A121[jcall8,7]*B121[jcall8,0]-A121[jcall8,8]*B121[jcall8,1]) + \
                  (A121[jcall8,6]*B121[jcall8,1]-A121[jcall8,7]*B121[jcall8,2]) - \
                  3.0*(A121[jcall8,5]*B121[jcall8,2]-A121[jcall8,6]*B121[jcall8,3]) - \
                  3.0*(A121[jcall8,4]*B121[jcall8,3]-A121[jcall8,5]*B121[jcall8,4]) + \
                  3.0*(A121[jcall8,3]*B121[jcall8,4]-A121[jcall8,4]*B121[jcall8,5]) + \
                  3.0*(A121[jcall8,2]*B121[jcall8,5]-A121[jcall8,3]*B121[jcall8,6]) - \
                  (A121[jcall8,1]*B121[jcall8,6]-A121[jcall8,2]*B121[jcall8,7]) - \
                  (A121[jcall8,0]*B121[jcall8,7]-A121[jcall8,1]*B121[jcall8,8])) \
                  /(26880.0*th.sqrt(th.tensor(3.0)))
        S221[jcall8] = th.pow(zeta_b[jcall8,1],4.5)* \
                   th.pow(zeta_a[jcall8,1],4.5)* \
                   rij[jcall8]**9 * \
                  ((A22[jcall8,6]*B22[jcall8,0]-A22[jcall8,8]*B22[jcall8,2]) - \
                  3.0*(A22[jcall8,4]*B22[jcall8,2]-A22[jcall8,6]*B22[jcall8,4]) + \
                  3.0*(A22[jcall8,2]*B22[jcall8,4]-A22[jcall8,4]*B22[jcall8,6]) - \
                  (A22[jcall8,0]*B22[jcall8,6]-A22[jcall8,2]*B22[jcall8,8])) \
                  /(26880.0)
        S222[jcall8] = th.pow(zeta_b[jcall8,1],4.5)* \
                   th.pow(zeta_a[jcall8,1],4.5)* \
                   rij[jcall8]**9 * \
                  ((A22[jcall8,8]-A22[jcall8,6])*(B22[jcall8,0]-B22[jcall8,2]) - \
                  3.0*(A22[jcall8,6]-A22[jcall8,4])*(B22[jcall8,2]-B22[jcall8,4]) + \
                  3.0*(A22[jcall8,4]-A22[jcall8,2])*(B22[jcall8,4]-B22[jcall8,6]) - \
                  (A22[jcall8,2]-A22[jcall8,0])*(B22[jcall8,6]-B22[jcall8,8])) \
                  /(53760.0)





    jcall651 = (jcall==651) #ii=4
    if(jcall651.sum() != 0):
        S111[jcall651] = th.pow(zeta_b[jcall651,0],1.5)* \
                   th.pow(zeta_a[jcall651,0],5.5)* \
                   rij[jcall651]**7 * \
                  (A111[jcall651,6]*B111[jcall651,0]+4.0*B111[jcall651,1]*A111[jcall651,5]+5.0*B111[jcall651,2]*A111[jcall651,4]- \
                   5.0*B111[jcall651,4]*A111[jcall651,2]-4.0*A111[jcall651,1]*B111[jcall651,5]-B111[jcall651,6]*A111[jcall651,0])/ \
                  (th.sqrt(th.tensor(14.0))*1440.0)
        S211[jcall651] = th.pow(zeta_b[jcall651,0],1.5)* \
                   th.pow(zeta_a[jcall651,1],5.5)* \
                   rij[jcall651]**7 * \
                  ((A211[jcall651,5]*B211[jcall651,0]+A211[jcall651,6]*B211[jcall651,1]) - \
                  3.0*(-A211[jcall651,4]*B211[jcall651,1]-A211[jcall651,5]*B211[jcall651,2]) + \
                  2.0*(A211[jcall651,3]*B211[jcall651,2]+A211[jcall651,4]*B211[jcall651,3]) + \
                  2.0*(-A211[jcall651,2]*B211[jcall651,3]-A211[jcall651,3]*B211[jcall651,4]) - \
                  3.0*(A211[jcall651,1]*B211[jcall651,4]+A211[jcall651,2]*B211[jcall651,5]) + \
                  (-A211[jcall651,0]*B211[jcall651,5]-A211[jcall651,1]*B211[jcall651,6])) \
                  /(480.0*th.sqrt(th.tensor(42.0)))




    jcall752 = (jcall==752) #ii=4
    if(jcall752.sum() != 0):
        S111[jcall752] = th.pow(zeta_b[jcall752,0],2.5)* \
                   th.pow(zeta_a[jcall752,0],5.5)* \
                   rij[jcall752]**8 * \
                  (A111[jcall752,7]*B111[jcall752,0]+3.0*B111[jcall752,1]*A111[jcall752,6]+B111[jcall752,2]*A111[jcall752,5]- \
                   5.0*B111[jcall752,3]*A111[jcall752,4]-5.0*A111[jcall752,3]*B111[jcall752,4]+1.0*B111[jcall752,5]*A111[jcall752,2]+ \
                   3.0*B111[jcall752,6]*A111[jcall752,1]+B111[jcall752,7]*A111[jcall752,0])/ \
                  (th.sqrt(th.tensor(42.0))*2880.0)
        S211[jcall752] = th.pow(zeta_b[jcall752,0],2.5)* \
                   th.pow(zeta_a[jcall752,1],5.5)* \
                   rij[jcall752]**8 * \
                  ((A211[jcall752,6]*B211[jcall752,0]+A211[jcall752,7]*B211[jcall752,1]) - \
                  2.0*(-A211[jcall752,5]*B211[jcall752,1]-A211[jcall752,6]*B211[jcall752,2]) - \
                  1.0*(A211[jcall752,4]*B211[jcall752,2]+A211[jcall752,5]*B211[jcall752,3]) + \
                  4.0*(-A211[jcall752,3]*B211[jcall752,3]-A211[jcall752,4]*B211[jcall752,4]) - \
                  1.0*(A211[jcall752,2]*B211[jcall752,4]+A211[jcall752,3]*B211[jcall752,5]) - \
                  2.0*(-A211[jcall752,1]*B211[jcall752,5]-A211[jcall752,2]*B211[jcall752,6]) + \
                  (A211[jcall752,0]*B211[jcall752,6]+A211[jcall752,1]*B211[jcall752,7])) \
                 /(2880.0*th.sqrt(th.tensor(14.0)))
        S121[jcall752] = th.pow(zeta_b[jcall752,1],2.5)* \
                   th.pow(zeta_a[jcall752,0],5.5)* \
                   rij[jcall752]**8 * \
                  ((A121[jcall752,6]*B121[jcall752,0]-A121[jcall752,7]*B121[jcall752,1]) + \
                  4.0*(A121[jcall752,5]*B121[jcall752,1]-A121[jcall752,6]*B121[jcall752,2]) + \
                  5.0*(A121[jcall752,4]*B121[jcall752,2]-A121[jcall752,5]*B121[jcall752,3]) - \
                  5.0*(A121[jcall752,2]*B121[jcall752,4]-A121[jcall752,3]*B121[jcall752,5]) - \
                  4.0*(A121[jcall752,1]*B121[jcall752,5]-A121[jcall752,2]*B121[jcall752,6]) - \
                  (A121[jcall752,0]*B121[jcall752,6]-A121[jcall752,1]*B121[jcall752,7])) \
                 /(2880.0*th.sqrt(th.tensor(14.0)))
        S221[jcall752] = th.pow(zeta_b[jcall752,1],2.5)* \
                   th.pow(zeta_a[jcall752,1],5.5)* \
                   rij[jcall752]**8 * \
                  ((A22[jcall752,5]*B22[jcall752,0]-A22[jcall752,7]*B22[jcall752,2]) + \
                  3.0*(A22[jcall752,4]*B22[jcall752,1]-A22[jcall752,6]*B22[jcall752,3]) + \
                  2.0*(A22[jcall752,3]*B22[jcall752,2]-A22[jcall752,5]*B22[jcall752,4]) - \
                  2.0*(A22[jcall752,2]*B22[jcall752,3]-A22[jcall752,4]*B22[jcall752,5]) - \
                  3.0*(A22[jcall752,1]*B22[jcall752,4]-A22[jcall752,3]*B22[jcall752,6]) - \
                  (A22[jcall752,0]*B22[jcall752,5]-A22[jcall752,2]*B22[jcall752,7])) \
                  /(960.0*th.sqrt(th.tensor(42.0)))
        S222[jcall752] = th.pow(zeta_b[jcall752,1],2.5)* \
                   th.pow(zeta_a[jcall752,1],5.5)* \
                   rij[jcall752]**8 * \
                  ((A22[jcall752,7]-A22[jcall752,5])*(B22[jcall752,0]-B22[jcall752,2]) + \
                  3.0*(A22[jcall752,6]-A22[jcall752,4])*(B22[jcall752,1]-B22[jcall752,3]) + \
                  2.0*(A22[jcall752,5]-A22[jcall752,3])*(B22[jcall752,2]-B22[jcall752,4]) - \
                  2.0*(A22[jcall752,4]-A22[jcall752,2])*(B22[jcall752,3]-B22[jcall752,5]) - \
                  3.0*(A22[jcall752,3]-A22[jcall752,1])*(B22[jcall752,4]-B22[jcall752,6]) - \
                  (A22[jcall752,2]-A22[jcall752,0])*(B22[jcall752,5]-B22[jcall752,7])) \
                  /(1920.0*th.sqrt(th.tensor(42.0)))





    jcall853 = (jcall==853) #ii=4
    if(jcall853.sum() != 0):
        S111[jcall853] = th.pow(zeta_b[jcall853,0],3.5)* \
                   th.pow(zeta_a[jcall853,0],5.5)* \
                   rij[jcall853]**9 * \
                  (A111[jcall853,8]*B111[jcall853,0]+2.0*B111[jcall853,1]*A111[jcall853,7]-2.0*B111[jcall853,2]*A111[jcall853,6]- \
                   6.0*B111[jcall853,3]*A111[jcall853,5]+6.0*A111[jcall853,3]*B111[jcall853,5]+2.0*B111[jcall853,6]*A111[jcall853,2]- \
                   2.0*B111[jcall853,7]*A111[jcall853,1]-B111[jcall853,8]*A111[jcall853,0])/ \
                  (th.sqrt(th.tensor(35.0))*17280.0)
        S211[jcall853] = th.pow(zeta_b[jcall853,0],3.5)* \
                   th.pow(zeta_a[jcall853,1],5.5)* \
                   rij[jcall853]**9 * \
                  ((A211[jcall853,7]*B211[jcall853,0]+A211[jcall853,8]*B211[jcall853,1]) - \
                  (-A211[jcall853,6]*B211[jcall853,1]-A211[jcall853,7]*B211[jcall853,2]) - \
                  3.0*(A211[jcall853,5]*B211[jcall853,2]+A211[jcall853,6]*B211[jcall853,3]) + \
                  3.0*(-A211[jcall853,4]*B211[jcall853,3]-A211[jcall853,5]*B211[jcall853,4]) + \
                  3.0*(A211[jcall853,3]*B211[jcall853,4]+A211[jcall853,4]*B211[jcall853,5]) - \
                  3.0*(-A211[jcall853,2]*B211[jcall853,5]-A211[jcall853,3]*B211[jcall853,6]) - \
                  (A211[jcall853,1]*B211[jcall853,6]+A211[jcall853,2]*B211[jcall853,7]) + \
                  (-A211[jcall853,0]*B211[jcall853,7]-A211[jcall853,1]*B211[jcall853,8])) \
                  /(5760.0*th.sqrt(th.tensor(105.0)))
        S121[jcall853] = th.pow(zeta_b[jcall853,1],3.5)* \
                   th.pow(zeta_a[jcall853,0],5.5)* \
                   rij[jcall853]**9 * \
                  ((A121[jcall853,7]*B121[jcall853,0]-A121[jcall853,8]*B121[jcall853,1]) + \
                  3.0*(A121[jcall853,6]*B121[jcall853,1]-A121[jcall853,7]*B121[jcall853,2]) + \
                  1.0*(A121[jcall853,5]*B121[jcall853,2]-A121[jcall853,6]*B121[jcall853,3]) - \
                  5.0*(A121[jcall853,4]*B121[jcall853,3]-A121[jcall853,5]*B121[jcall853,4]) - \
                  5.0*(A121[jcall853,3]*B121[jcall853,4]-A121[jcall853,4]*B121[jcall853,5]) + \
                  1.0*(A121[jcall853,2]*B121[jcall853,5]-A121[jcall853,3]*B121[jcall853,6]) + \
                  3.0*(A121[jcall853,1]*B121[jcall853,6]-A121[jcall853,2]*B121[jcall853,7]) + \
                  (A121[jcall853,0]*B121[jcall853,7]-A121[jcall853,1]*B121[jcall853,8])) \
                  /(5760.0*th.sqrt(th.tensor(105.0)))
        S221[jcall853] = th.pow(zeta_b[jcall853,1],3.5)* \
                   th.pow(zeta_a[jcall853,1],5.5)* \
                   rij[jcall853]**9 * \
                  ((A22[jcall853,6]*B22[jcall853,0]-A22[jcall853,8]*B22[jcall853,2]) + \
                  2.0*(A22[jcall853,5]*B22[jcall853,1]-A22[jcall853,7]*B22[jcall853,3]) - \
                  1.0*(A22[jcall853,4]*B22[jcall853,2]-A22[jcall853,6]*B22[jcall853,4]) + \
                  4.0*(A22[jcall853,3]*B22[jcall853,3]-A22[jcall853,5]*B22[jcall853,5]) - \
                  1.0*(A22[jcall853,2]*B22[jcall853,4]-A22[jcall853,4]*B22[jcall853,6]) + \
                  2.0*(A22[jcall853,1]*B22[jcall853,5]-A22[jcall853,3]*B22[jcall853,7]) + \
                  (A22[jcall853,0]*B22[jcall853,6]-A22[jcall853,2]*B22[jcall853,8])) \
                  /(5760.0*th.sqrt(th.tensor(35.0)))
        S222[jcall853] = th.pow(zeta_b[jcall853,1],3.5)* \
                   th.pow(zeta_a[jcall853,1],5.5)* \
                   rij[jcall853]**9 * \
                  ((A22[jcall853,8]-A22[jcall853,6])*(B22[jcall853,0]-B22[jcall853,2]) + \
                  2.0*(A22[jcall853,7]-A22[jcall853,5])*(B22[jcall853,1]-B22[jcall853,3]) - \
                  1.0*(A22[jcall853,6]-A22[jcall853,4])*(B22[jcall853,2]-B22[jcall853,4]) + \
                  4.0*(A22[jcall853,5]-A22[jcall853,3])*(B22[jcall853,3]-B22[jcall853,5]) - \
                  1.0*(A22[jcall853,4]-A22[jcall853,2])*(B22[jcall853,4]-B22[jcall853,6]) + \
                  2.0*(A22[jcall853,3]-A22[jcall853,1])*(B22[jcall853,5]-B22[jcall853,7]) + \
                  (A22[jcall853,2]-A22[jcall853,0])*(B22[jcall853,6]-B22[jcall853,8])) \
                  /(11520.0*th.sqrt(th.tensor(35.0)))




    jcall9 = (jcall==9) #ii=4
    if(jcall9.sum() != 0):
        S111[jcall9] = th.pow(zeta_b[jcall9,0],4.5)* \
                   th.pow(zeta_a[jcall9,0],5.5)* \
                   rij[jcall9]**10 * \
                  (A111[jcall9,9]*B111[jcall9,0]+1.0*B111[jcall9,1]*A111[jcall9,8]-4.0*B111[jcall9,2]*A111[jcall9,7]- \
                   4.0*B111[jcall9,3]*A111[jcall9,6]+6.0*A111[jcall9,5]*B111[jcall9,4]+6.0*B111[jcall9,5]*A111[jcall9,4]- \
                   4.0*B111[jcall9,6]*A111[jcall9,3]-4.0*B111[jcall9,7]*A111[jcall9,2]+A111[jcall9,1]*B111[jcall9,8]+A111[jcall9,0]*B111[jcall9,9])/ \
                  (th.sqrt(th.tensor(10.0))*241920.0)
        S211[jcall9] = th.pow(zeta_b[jcall9,0],4.5)* \
                   th.pow(zeta_a[jcall9,1],5.5)* \
                   rij[jcall9]**10 * \
                  ((A211[jcall9,8]*B211[jcall9,0]-A211[jcall9,9]*B211[jcall9,1]) - \
                  4.0*(A211[jcall9,6]*B211[jcall9,2]+A211[jcall9,7]*B211[jcall9,3]) + \
                  6.0*(A211[jcall9,4]*B211[jcall9,4]+A211[jcall9,5]*B211[jcall9,5]) - \
                  4.0*(A211[jcall9,2]*B211[jcall9,6]+A211[jcall9,3]*B211[jcall9,7]) + \
                  1.0*(A211[jcall9,0]*B211[jcall9,8]+A211[jcall9,1]*B211[jcall9,9])) \
                  /(80640.0*th.sqrt(th.tensor(30.0)))
        S121[jcall9] = th.pow(zeta_b[jcall9,1],4.5)* \
                   th.pow(zeta_a[jcall9,0],5.5)* \
                   rij[jcall9]**10 * \
                  ((A121[jcall9,8]*B121[jcall9,0]-A121[jcall9,9]*B121[jcall9,1]) + \
                  2.0*(A121[jcall9,7]*B121[jcall9,1]-A121[jcall9,8]*B121[jcall9,2]) - \
                  2.0*(A121[jcall9,6]*B121[jcall9,2]-A121[jcall9,7]*B121[jcall9,3]) - \
                  6.0*(A121[jcall9,5]*B121[jcall9,3]-A121[jcall9,6]*B121[jcall9,4]) + \
                  6.0*(A121[jcall9,3]*B121[jcall9,5]-A121[jcall9,4]*B121[jcall9,6]) + \
                  2.0*(A121[jcall9,2]*B121[jcall9,6]-A121[jcall9,3]*B121[jcall9,7]) - \
                  2.0*(A121[jcall9,1]*B121[jcall9,7]-A121[jcall9,2]*B121[jcall9,8]) - \
                  1.0*(A121[jcall9,0]*B121[jcall9,8]-A121[jcall9,1]*B121[jcall9,9])) \
                  /(80640.0*th.sqrt(th.tensor(30.0)))
        S221[jcall9] = th.pow(zeta_b[jcall9,1],4.5)* \
                   th.pow(zeta_a[jcall9,1],5.5)* \
                   rij[jcall9]**10 * \
                  ((A22[jcall9,7]*B22[jcall9,0]-A22[jcall9,9]*B22[jcall9,2]) + \
                  1.0*(A22[jcall9,6]*B22[jcall9,1]-A22[jcall9,8]*B22[jcall9,3]) - \
                  3.0*(A22[jcall9,5]*B22[jcall9,2]-A22[jcall9,7]*B22[jcall9,4]) - \
                  3.0*(A22[jcall9,4]*B22[jcall9,3]-A22[jcall9,6]*B22[jcall9,5]) + \
                  3.0*(A22[jcall9,3]*B22[jcall9,4]-A22[jcall9,5]*B22[jcall9,6]) + \
                  3.0*(A22[jcall9,2]*B22[jcall9,5]-A22[jcall9,4]*B22[jcall9,7]) - \
                  1.0*(A22[jcall9,1]*B22[jcall9,6]-A22[jcall9,3]*B22[jcall9,8]) - \
                  (A22[jcall9,0]*B22[jcall9,7]-A22[jcall9,2]*B22[jcall9,9])) \
                  /(80640.0*th.sqrt(th.tensor(10.0)))
        S222[jcall9] = th.pow(zeta_b[jcall9,1],4.5)* \
                   th.pow(zeta_a[jcall9,1],5.5)* \
                   rij[jcall9]**10 * \
                  ((A22[jcall9,9]-A22[jcall9,7])*(B22[jcall9,0]-B22[jcall9,2]) + \
                  1.0*(A22[jcall9,8]-A22[jcall9,6])*(B22[jcall9,1]-B22[jcall9,3]) - \
                  3.0*(A22[jcall9,7]-A22[jcall9,5])*(B22[jcall9,2]-B22[jcall9,4]) - \
                  3.0*(A22[jcall9,6]-A22[jcall9,4])*(B22[jcall9,3]-B22[jcall9,5]) + \
                  3.0*(A22[jcall9,5]-A22[jcall9,3])*(B22[jcall9,4]-B22[jcall9,6]) + \
                  3.0*(A22[jcall9,4]-A22[jcall9,2])*(B22[jcall9,5]-B22[jcall9,7]) - \
                  1.0*(A22[jcall9,3]-A22[jcall9,1])*(B22[jcall9,6]-B22[jcall9,8]) - \
                  (A22[jcall9,2]-A22[jcall9,0])*(B22[jcall9,7]-B22[jcall9,9])) \
                  /(161280.0*th.sqrt(th.tensor(10.0)))





    jcall10 = (jcall==10)
    if(jcall10.sum() != 0):
        S111[jcall10] = th.pow(zeta_b[jcall10,0]* \
                          zeta_a[jcall10,0],5.5)* \
                   rij[jcall10]**11 * \
                  (A111[jcall10,10]*B111[jcall10,0]-5.0*B111[jcall10,2]*A111[jcall10,8]+10.0*A111[jcall10,6]*B111[jcall10,4] \
                   -10.0*A111[jcall10,4]*B111[jcall10,6]+5.0*A111[jcall10,2]*B111[jcall10,8]-(A111[jcall10,0]*B111[jcall10,10]))/7257600.0
        S211[jcall10] = th.pow(zeta_b[jcall10,0],5.5)* \
                   th.pow(zeta_a[jcall10,1],5.5)* \
                   rij[jcall10]**11 * \
                  ((A211[jcall10,9]*B211[jcall10,0]+A211[jcall10,10]*B211[jcall10,1]) + \
                  1.0*(-A211[jcall10,8]*B211[jcall10,1]-A211[jcall10,9]*B211[jcall10,2]) - \
                  4.0*(A211[jcall10,7]*B211[jcall10,2]+A211[jcall10,8]*B211[jcall10,3]) - \
                  4.0*(-A211[jcall10,6]*B211[jcall10,3]-A211[jcall10,7]*B211[jcall10,4]) + \
                  6.0*(A211[jcall10,5]*B211[jcall10,4]+A211[jcall10,6]*B211[jcall10,5]) + \
                  6.0*(-A211[jcall10,4]*B211[jcall10,5]-A211[jcall10,5]*B211[jcall10,6]) - \
                  4.0*(A211[jcall10,3]*B211[jcall10,6]+A211[jcall10,4]*B211[jcall10,7]) - \
                  4.0*(-A211[jcall10,2]*B211[jcall10,7]-A211[jcall10,3]*B211[jcall10,8]) + \
                  1.0*(A211[jcall10,1]*B211[jcall10,8]+A211[jcall10,2]*B211[jcall10,9]) + \
                  1.0*(-A211[jcall10,0]*B211[jcall10,9]-A211[jcall10,1]*B211[jcall10,10])) \
                  /(2419200.0*th.sqrt(th.tensor(3.0)))
        S121[jcall10] = th.pow(zeta_b[jcall10,1],5.5)* \
                   th.pow(zeta_a[jcall10,0],5.5)* \
                   rij[jcall10]**11 * \
                  ((A121[jcall10,9]*B121[jcall10,0]-A121[jcall10,10]*B121[jcall10,1]) + \
                  1.0*(A121[jcall10,8]*B121[jcall10,1]-A121[jcall10,9]*B121[jcall10,2]) - \
                  4.0*(A121[jcall10,7]*B121[jcall10,2]-A121[jcall10,8]*B121[jcall10,3]) - \
                  4.0*(A121[jcall10,6]*B121[jcall10,3]-A121[jcall10,7]*B121[jcall10,4]) + \
                  6.0*(A121[jcall10,5]*B121[jcall10,4]-A121[jcall10,6]*B121[jcall10,5]) + \
                  6.0*(A121[jcall10,4]*B121[jcall10,5]-A121[jcall10,5]*B121[jcall10,6]) - \
                  4.0*(A121[jcall10,3]*B121[jcall10,6]-A121[jcall10,4]*B121[jcall10,7]) - \
                  4.0*(A121[jcall10,2]*B121[jcall10,7]-A121[jcall10,3]*B121[jcall10,8]) + \
                  1.0*(A121[jcall10,1]*B121[jcall10,8]-A121[jcall10,2]*B121[jcall10,9]) + \
                  1.0*(A121[jcall10,0]*B121[jcall10,9]-A121[jcall10,1]*B121[jcall10,10])) \
                  /(2419200.0*th.sqrt(th.tensor(3.0)))
        S221[jcall10] = th.pow(zeta_b[jcall10,1],5.5)* \
                   th.pow(zeta_a[jcall10,1],5.5)* \
                   rij[jcall10]**11 * \
                  ((A22[jcall10,8]*B22[jcall10,0]-A22[jcall10,10]*B22[jcall10,2]) - \
                  4.0*(A22[jcall10,6]*B22[jcall10,2]-A22[jcall10,8]*B22[jcall10,4]) + \
                  5.0*(A22[jcall10,4]*B22[jcall10,4]-A22[jcall10,6]*B22[jcall10,6]) - \
                  4.0*(A22[jcall10,2]*B22[jcall10,6]-A22[jcall10,4]*B22[jcall10,8]) + \
                  (A22[jcall10,0]*B22[jcall10,8]-A22[jcall10,2]*B22[jcall10,10])) \
                  /(2419200)
        S222[jcall10] = th.pow(zeta_b[jcall10,1],5.5)* \
                   th.pow(zeta_a[jcall10,1],5.5)* \
                   rij[jcall10]**11 * \
                  ((A22[jcall10,10]-A22[jcall10,8])*(B22[jcall10,0]-B22[jcall10,2]) - \
                  4.0*(A22[jcall10,8]-A22[jcall10,6])*(B22[jcall10,2]-B22[jcall10,4]) + \
                  5.0*(A22[jcall10,6]-A22[jcall10,4])*(B22[jcall10,4]-B22[jcall10,6]) - \
                  4.0*(A22[jcall10,4]-A22[jcall10,2])*(B22[jcall10,6]-B22[jcall10,8]) + \
                  (A22[jcall10,2]-A22[jcall10,0])*(B22[jcall10,8]-B22[jcall10,10])) \
                  /(4838400)





    jcall12 = (jcall==12)
    if(jcall12.sum() != 0):
        S111[jcall12] = th.pow(zeta_b[jcall12,0]* \
                          zeta_a[jcall12,0],6.5)* \
                   rij[jcall12]**13 * \
                  (A111[jcall12,12]*B111[jcall12,0]-6.0*B111[jcall12,2]*A111[jcall12,10]+15.0*A111[jcall12,8]*B111[jcall12,4] \
                   -20.0*A111[jcall12,6]*B111[jcall12,6]+15.0*A111[jcall12,4]*B111[jcall12,8]-6.0*(A111[jcall12,2]*B111[jcall12,10]+A111[jcall12,0]*B111[jcall12,12]))/958003200.0
        S211[jcall12] = th.pow(zeta_b[jcall12,0],6.5)* \
                   th.pow(zeta_a[jcall12,1],6.5)* \
                   rij[jcall12]**13 * \
                  ((A211[jcall12,11]*B211[jcall12,0]+A211[jcall12,12]*B211[jcall12,1]) + \
                  1.0*(-A211[jcall12,10]*B211[jcall12,1]-A211[jcall12,11]*B211[jcall12,2]) - \
                  5.0*(A211[jcall12,9]*B211[jcall12,2]+A211[jcall12,10]*B211[jcall12,3]) - \
                  5.0*(-A211[jcall12,8]*B211[jcall12,3]-A211[jcall12,9]*B211[jcall12,4]) + \
                  10.0*(A211[jcall12,7]*B211[jcall12,4]+A211[jcall12,8]*B211[jcall12,5]) + \
                  10.0*(-A211[jcall12,6]*B211[jcall12,5]-A211[jcall12,7]*B211[jcall12,6]) - \
                  10.0*(A211[jcall12,5]*B211[jcall12,6]+A211[jcall12,6]*B211[jcall12,7]) - \
                  10.0*(-A211[jcall12,4]*B211[jcall12,7]-A211[jcall12,5]*B211[jcall12,8]) + \
                  5.0*(A211[jcall12,3]*B211[jcall12,8]+A211[jcall12,4]*B211[jcall12,9]) + \
                  5.0*(-A211[jcall12,2]*B211[jcall12,9]-A211[jcall12,3]*B211[jcall12,10]) - \
                  1.0*(A211[jcall12,1]*B211[jcall12,10]+A211[jcall12,2]*B211[jcall12,11]) - \
                  1.0*(-A211[jcall12,0]*B211[jcall12,11]-A211[jcall12,1]*B211[jcall12,12])) \
                  /(319334400.0*th.sqrt(th.tensor(3.0)))
        S121[jcall12] = th.pow(zeta_b[jcall12,1],6.5)* \
                   th.pow(zeta_a[jcall12,0],6.5)* \
                   rij[jcall12]**13 * \
                  ((A121[jcall12,11]*B121[jcall12,0]-A121[jcall12,12]*B121[jcall12,1]) + \
                  1.0*(A121[jcall12,10]*B121[jcall12,1]-A121[jcall12,11]*B121[jcall12,2]) - \
                  5.0*(A121[jcall12,9]*B121[jcall12,2]-A121[jcall12,10]*B121[jcall12,3]) - \
                  5.0*(A121[jcall12,8]*B121[jcall12,3]-A121[jcall12,9]*B121[jcall12,4]) + \
                  10.0*(A121[jcall12,7]*B121[jcall12,4]-A121[jcall12,8]*B121[jcall12,5]) + \
                  10.0*(A121[jcall12,6]*B121[jcall12,5]-A121[jcall12,7]*B121[jcall12,6]) - \
                  10.0*(A121[jcall12,5]*B121[jcall12,6]-A121[jcall12,6]*B121[jcall12,7]) - \
                  10.0*(A121[jcall12,4]*B121[jcall12,7]-A121[jcall12,5]*B121[jcall12,8]) + \
                  5.0*(A121[jcall12,3]*B121[jcall12,8]-A121[jcall12,4]*B121[jcall12,9]) + \
                  5.0*(A121[jcall12,2]*B121[jcall12,9]-A121[jcall12,3]*B121[jcall12,10]) - \
                  1.0*(A121[jcall12,1]*B121[jcall12,10]-A121[jcall12,2]*B121[jcall12,11]) - \
                  1.0*(A121[jcall12,0]*B121[jcall12,11]-A121[jcall12,1]*B121[jcall12,12])) \
                  /(319334400.0*th.sqrt(th.tensor(3.0)))
        S221[jcall12] = th.pow(zeta_b[jcall12,1],6.5)* \
                   th.pow(zeta_a[jcall12,1],6.5)* \
                   rij[jcall12]**13 * \
                  ((A22[jcall12,10]*B22[jcall12,0]-A22[jcall12,12]*B22[jcall12,2]) - \
                  5.0*(A22[jcall12,8]*B22[jcall12,2]-A22[jcall12,10]*B22[jcall12,4]) + \
                  10.0*(A22[jcall12,6]*B22[jcall12,4]-A22[jcall12,8]*B22[jcall12,6]) - \
                  10.0*(A22[jcall12,4]*B22[jcall12,6]-A22[jcall12,6]*B22[jcall12,8]) + \
                  5.0*(A22[jcall12,2]*B22[jcall12,8]-A22[jcall12,4]*B22[jcall12,10]) - \
                  (A22[jcall12,0]*B22[jcall12,10]-A22[jcall12,2]*B22[jcall12,12])) \
                  /(319334400)
        S222[jcall12] = th.pow(zeta_b[jcall12,1],6.5)* \
                   th.pow(zeta_a[jcall12,1],6.5)* \
                   rij[jcall12]**13 * \
                  ((A22[jcall12,12]-A22[jcall12,10])*(B22[jcall12,0]-B22[jcall12,2]) - \
                  5.0*(A22[jcall12,10]-A22[jcall12,8])*(B22[jcall12,2]-B22[jcall12,4]) + \
                  10.0*(A22[jcall12,8]-A22[jcall12,6])*(B22[jcall12,4]-B22[jcall12,6]) - \
                  10.0*(A22[jcall12,6]-A22[jcall12,4])*(B22[jcall12,6]-B22[jcall12,8]) + \
                  5.0*(A22[jcall12,4]-A22[jcall12,2])*(B22[jcall12,8]-B22[jcall12,10]) - \
                  (A22[jcall12,2]-A22[jcall12,0])*(B22[jcall12,10]-B22[jcall12,12])) \
                  /(638668800)







    jcall761 = (jcall==761) #ii=4
    if(jcall761.sum() != 0):
        S111[jcall761] = th.pow(zeta_b[jcall761,0],1.5)* \
                   th.pow(zeta_a[jcall761,0],6.5)* \
                   rij[jcall761]**8 * \
                  (A111[jcall761,7]*B111[jcall761,0]+5.0*B111[jcall761,1]*A111[jcall761,6]+9.0*B111[jcall761,2]*A111[jcall761,5]+ \
                   5.0*B111[jcall761,3]*A111[jcall761,4]-5.0*A111[jcall761,3]*B111[jcall761,4]-9.0*B111[jcall761,5]*A111[jcall761,2]- \
                   5.0*B111[jcall761,6]*A111[jcall761,1]-1.0*B111[jcall761,7]*A111[jcall761,0])/ \
                  (th.sqrt(th.tensor(462.0))*2880.0)
        S211[jcall761] = th.pow(zeta_b[jcall761,0],1.5)* \
                   th.pow(zeta_a[jcall761,1],6.5)* \
                   rij[jcall761]**8 * \
                  ((A211[jcall761,6]*B211[jcall761,0]+A211[jcall761,7]*B211[jcall761,1]) - \
                  4.0*(-A211[jcall761,5]*B211[jcall761,1]-A211[jcall761,6]*B211[jcall761,2]) + \
                  5.0*(A211[jcall761,4]*B211[jcall761,2]+A211[jcall761,5]*B211[jcall761,3]) - \
                  5.0*(A211[jcall761,2]*B211[jcall761,4]+A211[jcall761,3]*B211[jcall761,5]) + \
                  4.0*(-A211[jcall761,1]*B211[jcall761,5]-A211[jcall761,2]*B211[jcall761,6]) - \
                  (A211[jcall761,0]*B211[jcall761,6]+A211[jcall761,1]*B211[jcall761,7])) \
                 /(2880.0*th.sqrt(th.tensor(154.0)))






    jcall862 = (jcall==862) #ii=4
    if(jcall862.sum() != 0):
        S111[jcall862] = th.pow(zeta_b[jcall862,0],2.5)* \
                   th.pow(zeta_a[jcall862,0],6.5)* \
                   rij[jcall862]**9 * \
                  (A111[jcall862,8]*B111[jcall862,0]+4.0*B111[jcall862,1]*A111[jcall862,7]+4.0*B111[jcall862,2]*A111[jcall862,6]- \
                   4.0*B111[jcall862,3]*A111[jcall862,5]-10.0*A111[jcall862,4]*B111[jcall862,4]-4.0*B111[jcall862,5]*A111[jcall862,3]+ \
                   4.0*B111[jcall862,6]*A111[jcall862,2]+4.0*B111[jcall862,7]*A111[jcall862,1]+1.0*A111[jcall862,0]*B111[jcall862,8])/ \
                  (th.sqrt(th.tensor(154.0))*17280.0)
        S211[jcall862] = th.pow(zeta_b[jcall862,0],2.5)* \
                   th.pow(zeta_a[jcall862,1],6.5)* \
                   rij[jcall862]**9 * \
                  ((A211[jcall862,7]*B211[jcall862,0]+A211[jcall862,8]*B211[jcall862,1]) - \
                  3.0*(-A211[jcall862,6]*B211[jcall862,1]-A211[jcall862,7]*B211[jcall862,2]) + \
                  1.0*(A211[jcall862,5]*B211[jcall862,2]+A211[jcall862,6]*B211[jcall862,3]) + \
                  5.0*(-A211[jcall862,4]*B211[jcall862,3]-A211[jcall862,5]*B211[jcall862,4]) - \
                  5.0*(A211[jcall862,3]*B211[jcall862,4]+A211[jcall862,4]*B211[jcall862,5]) - \
                  1.0*(-A211[jcall862,2]*B211[jcall862,5]-A211[jcall862,3]*B211[jcall862,6]) + \
                  3.0*(A211[jcall862,1]*B211[jcall862,6]+A211[jcall862,2]*B211[jcall862,7]) - \
                  (-A211[jcall862,0]*B211[jcall862,7]-A211[jcall862,1]*B211[jcall862,8])) \
                  /(5760.0*th.sqrt(th.tensor(462.0)))
        S121[jcall862] = th.pow(zeta_b[jcall862,1],2.5)* \
                   th.pow(zeta_a[jcall862,0],6.5)* \
                   rij[jcall862]**9 * \
                  ((A121[jcall862,7]*B121[jcall862,0]-A121[jcall862,8]*B121[jcall862,1]) + \
                  5.0*(A121[jcall862,6]*B121[jcall862,1]-A121[jcall862,7]*B121[jcall862,2]) + \
                  9.0*(A121[jcall862,5]*B121[jcall862,2]-A121[jcall862,6]*B121[jcall862,3]) + \
                  5.0*(A121[jcall862,4]*B121[jcall862,3]-A121[jcall862,5]*B121[jcall862,4]) - \
                  5.0*(A121[jcall862,3]*B121[jcall862,4]-A121[jcall862,4]*B121[jcall862,5]) - \
                  9.0*(A121[jcall862,2]*B121[jcall862,5]-A121[jcall862,3]*B121[jcall862,6]) - \
                  5.0*(A121[jcall862,1]*B121[jcall862,6]-A121[jcall862,2]*B121[jcall862,7]) - \
                  (A121[jcall862,0]*B121[jcall862,7]-A121[jcall862,1]*B121[jcall862,8])) \
                  /(5760.0*th.sqrt(th.tensor(462.0)))
        S221[jcall862] = th.pow(zeta_b[jcall862,1],2.5)* \
                   th.pow(zeta_a[jcall862,1],6.5)* \
                   rij[jcall862]**9 * \
                  ((A22[jcall862,6]*B22[jcall862,0]-A22[jcall862,8]*B22[jcall862,2]) + \
                  4.0*(A22[jcall862,5]*B22[jcall862,1]-A22[jcall862,7]*B22[jcall862,3]) + \
                  5.0*(A22[jcall862,4]*B22[jcall862,2]-A22[jcall862,6]*B22[jcall862,4]) - \
                  5.0*(A22[jcall862,2]*B22[jcall862,4]-A22[jcall862,4]*B22[jcall862,6]) - \
                  4.0*(A22[jcall862,1]*B22[jcall862,5]-A22[jcall862,3]*B22[jcall862,7]) - \
                  (A22[jcall862,0]*B22[jcall862,6]-A22[jcall862,2]*B22[jcall862,8])) \
                  /(5760.0*th.sqrt(th.tensor(154.0)))
        S222[jcall862] = th.pow(zeta_b[jcall862,1],2.5)* \
                   th.pow(zeta_a[jcall862,1],6.5)* \
                   rij[jcall862]**9 * \
                  ((A22[jcall862,8]-A22[jcall862,6])*(B22[jcall862,0]-B22[jcall862,2]) + \
                  4.0*(A22[jcall862,7]-A22[jcall862,5])*(B22[jcall862,1]-B22[jcall862,3]) + \
                  5.0*(A22[jcall862,6]-A22[jcall862,4])*(B22[jcall862,2]-B22[jcall862,4]) - \
                  5.0*(A22[jcall862,4]-A22[jcall862,2])*(B22[jcall862,4]-B22[jcall862,6]) - \
                  4.0*(A22[jcall862,3]-A22[jcall862,1])*(B22[jcall862,5]-B22[jcall862,7]) - \
                  (A22[jcall862,2]-A22[jcall862,0])*(B22[jcall862,6]-B22[jcall862,8])) \
                  /(11520.0*th.sqrt(th.tensor(154.0)))





    jcall963 = (jcall==963) #ii=4
    if(jcall963.sum() != 0):
        S111[jcall963] = th.pow(zeta_b[jcall963,0],3.5)* \
                   th.pow(zeta_a[jcall963,0],6.5)* \
                   rij[jcall963]**9 * \
                  (A111[jcall963,9]*B111[jcall963,0]+3.0*B111[jcall963,1]*A111[jcall963,8]-8.0*B111[jcall963,3]*A111[jcall963,6]- \
                   6.0*B111[jcall963,4]*A111[jcall963,5]+6.0*A111[jcall963,4]*B111[jcall963,5]+8.0*B111[jcall963,6]*A111[jcall963,3]- \
                   3.0*B111[jcall963,8]*A111[jcall963,1]-1.0*B111[jcall963,9]*A111[jcall963,0])/ \
                  (th.sqrt(th.tensor(1155.0))*34560.0)
        S211[jcall963] = th.pow(zeta_b[jcall963,0],3.5)* \
                   th.pow(zeta_a[jcall963,1],6.5)* \
                   rij[jcall963]**10 * \
                  ((A211[jcall963,8]*B211[jcall963,0]+A211[jcall963,9]*B211[jcall963,1]) - \
                  2.0*(-A211[jcall963,7]*B211[jcall963,1]-A211[jcall963,8]*B211[jcall963,2]) - \
                  2.0*(A211[jcall963,6]*B211[jcall963,2]+A211[jcall963,7]*B211[jcall963,3]) + \
                  6.0*(-A211[jcall963,5]*B211[jcall963,3]-A211[jcall963,6]*B211[jcall963,4]) - \
                  6.0*(-A211[jcall963,3]*B211[jcall963,5]-A211[jcall963,4]*B211[jcall963,6]) + \
                  2.0*(A211[jcall963,2]*B211[jcall963,6]+A211[jcall963,3]*B211[jcall963,7]) + \
                  2.0*(-A211[jcall963,1]*B211[jcall963,7]-A211[jcall963,2]*B211[jcall963,8]) - \
                  1.0*(A211[jcall963,0]*B211[jcall963,8]+A211[jcall963,1]*B211[jcall963,9])) \
                  /(34560.0*th.sqrt(th.tensor(385.0)))
        S121[jcall963] = th.pow(zeta_b[jcall963,1],3.5)* \
                   th.pow(zeta_a[jcall963,0],6.5)* \
                   rij[jcall963]**10 * \
                  ((A121[jcall963,8]*B121[jcall963,0]-A121[jcall963,9]*B121[jcall963,1]) + \
                  4.0*(A121[jcall963,7]*B121[jcall963,1]-A121[jcall963,8]*B121[jcall963,2]) + \
                  4.0*(A121[jcall963,6]*B121[jcall963,2]-A121[jcall963,7]*B121[jcall963,3]) - \
                  4.0*(A121[jcall963,5]*B121[jcall963,3]-A121[jcall963,6]*B121[jcall963,4]) - \
                  10.0*(A121[jcall963,4]*B121[jcall963,4]-A121[jcall963,5]*B121[jcall963,5]) - \
                  4.0*(A121[jcall963,3]*B121[jcall963,5]-A121[jcall963,4]*B121[jcall963,6]) + \
                  4.0*(A121[jcall963,2]*B121[jcall963,6]-A121[jcall963,3]*B121[jcall963,7]) + \
                  4.0*(A121[jcall963,1]*B121[jcall963,7]-A121[jcall963,2]*B121[jcall963,8]) + \
                  1.0*(A121[jcall963,0]*B121[jcall963,8]-A121[jcall963,1]*B121[jcall963,9])) \
                  /(34560.0*th.sqrt(th.tensor(385.0)))
        S221[jcall963] = th.pow(zeta_b[jcall963,1],3.5)* \
                   th.pow(zeta_a[jcall963,1],6.5)* \
                   rij[jcall963]**10 * \
                  ((A22[jcall963,7]*B22[jcall963,0]-A22[jcall963,9]*B22[jcall963,2]) + \
                  3.0*(A22[jcall963,6]*B22[jcall963,1]-A22[jcall963,8]*B22[jcall963,3]) + \
                  1.0*(A22[jcall963,5]*B22[jcall963,2]-A22[jcall963,7]*B22[jcall963,4]) - \
                  5.0*(A22[jcall963,4]*B22[jcall963,3]-A22[jcall963,6]*B22[jcall963,5]) - \
                  5.0*(A22[jcall963,3]*B22[jcall963,4]-A22[jcall963,5]*B22[jcall963,6]) + \
                  1.0*(A22[jcall963,2]*B22[jcall963,5]-A22[jcall963,4]*B22[jcall963,7]) + \
                  3.0*(A22[jcall963,1]*B22[jcall963,6]-A22[jcall963,3]*B22[jcall963,8]) + \
                  (A22[jcall963,0]*B22[jcall963,7]-A22[jcall963,2]*B22[jcall963,9])) \
                  /(11520.0*th.sqrt(th.tensor(1155.0)))
        S222[jcall963] = th.pow(zeta_b[jcall963,1],3.5)* \
                   th.pow(zeta_a[jcall963,1],6.5)* \
                   rij[jcall963]**10 * \
                  ((A22[jcall963,9]-A22[jcall963,7])*(B22[jcall963,0]-B22[jcall963,2]) + \
                  3.0*(A22[jcall963,8]-A22[jcall963,6])*(B22[jcall963,1]-B22[jcall963,3]) + \
                  1.0*(A22[jcall963,7]-A22[jcall963,5])*(B22[jcall963,2]-B22[jcall963,4]) - \
                  5.0*(A22[jcall963,6]-A22[jcall963,4])*(B22[jcall963,3]-B22[jcall963,5]) - \
                  5.0*(A22[jcall963,5]-A22[jcall963,3])*(B22[jcall963,4]-B22[jcall963,6]) + \
                  1.0*(A22[jcall963,4]-A22[jcall963,2])*(B22[jcall963,5]-B22[jcall963,7]) + \
                  3.0*(A22[jcall963,3]-A22[jcall963,1])*(B22[jcall963,6]-B22[jcall963,8]) + \
                  (A22[jcall963,2]-A22[jcall963,0])*(B22[jcall963,7]-B22[jcall963,9])) \
                  /(23040.0*th.sqrt(th.tensor(1155.0)))







    jcall1064 = (jcall==1064) #ii=4
    if(jcall1064.sum() != 0):
        S111[jcall1064] = th.pow(zeta_b[jcall1064,0],4.5)* \
                   th.pow(zeta_a[jcall1064,0],6.5)* \
                   rij[jcall1064]**11 * \
                  (A111[jcall1064,10]*B111[jcall1064,0]+2.0*B111[jcall1064,1]*A111[jcall1064,9]-3.0*B111[jcall1064,2]*A111[jcall1064,8]- \
                   8.0*B111[jcall1064,3]*A111[jcall1064,7]+2.0*A111[jcall1064,6]*B111[jcall1064,4]+12.0*B111[jcall1064,5]*A111[jcall1064,5]+ \
                   2.0*B111[jcall1064,6]*A111[jcall1064,4]-8.0*B111[jcall1064,7]*A111[jcall1064,3]-3.0*A111[jcall1064,2]*B111[jcall1064,8]+2.0*A111[jcall1064,1]*B111[jcall1064,9]+ \
                   A111[jcall1064,0]*B111[jcall1064,10])/ \
                  (th.sqrt(th.tensor(330.0))*483840)
        S211[jcall1064] = th.pow(zeta_b[jcall1064,0],4.5)* \
                   th.pow(zeta_a[jcall1064,1],6.5)* \
                   rij[jcall1064]**11 * \
                  ((A211[jcall1064,9]*B211[jcall1064,0]+A211[jcall1064,10]*B211[jcall1064,1]) - \
                  1.0*(-A211[jcall1064,8]*B211[jcall1064,1]-A211[jcall1064,9]*B211[jcall1064,2]) - \
                  4.0*(A211[jcall1064,7]*B211[jcall1064,2]+A211[jcall1064,8]*B211[jcall1064,3]) + \
                  4.0*(-A211[jcall1064,6]*B211[jcall1064,3]-A211[jcall1064,7]*B211[jcall1064,4]) + \
                  6.0*(A211[jcall1064,5]*B211[jcall1064,4]+A211[jcall1064,6]*B211[jcall1064,5]) - \
                  6.0*(-A211[jcall1064,4]*B211[jcall1064,5]-A211[jcall1064,5]*B211[jcall1064,6]) - \
                  4.0*(A211[jcall1064,3]*B211[jcall1064,6]+A211[jcall1064,4]*B211[jcall1064,7]) + \
                  4.0*(-A211[jcall1064,2]*B211[jcall1064,7]-A211[jcall1064,3]*B211[jcall1064,8]) + \
                  1.0*(A211[jcall1064,1]*B211[jcall1064,8]+A211[jcall1064,2]*B211[jcall1064,9]) - \
                  1.0*(-A211[jcall1064,0]*B211[jcall1064,9]-A211[jcall1064,1]*B211[jcall1064,10])) \
                  /(483840.0*th.sqrt(th.tensor(110.0)))
        S121[jcall1064] = th.pow(zeta_b[jcall1064,1],4.5)* \
                   th.pow(zeta_a[jcall1064,0],6.5)* \
                   rij[jcall1064]**11 * \
                  ((A121[jcall1064,9]*B121[jcall1064,0]-A121[jcall1064,10]*B121[jcall1064,1]) + \
                  3.0*(A121[jcall1064,8]*B121[jcall1064,1]-A121[jcall1064,9]*B121[jcall1064,2]) - \
                  8.0*(A121[jcall1064,6]*B121[jcall1064,3]-A121[jcall1064,7]*B121[jcall1064,4]) - \
                  6.0*(A121[jcall1064,5]*B121[jcall1064,4]-A121[jcall1064,6]*B121[jcall1064,5]) + \
                  6.0*(A121[jcall1064,4]*B121[jcall1064,5]-A121[jcall1064,5]*B121[jcall1064,6]) + \
                  8.0*(A121[jcall1064,3]*B121[jcall1064,6]-A121[jcall1064,4]*B121[jcall1064,7]) - \
                  3.0*(A121[jcall1064,1]*B121[jcall1064,8]-A121[jcall1064,2]*B121[jcall1064,9]) - \
                  1.0*(A121[jcall1064,0]*B121[jcall1064,9]-A121[jcall1064,1]*B121[jcall1064,10])) \
                  /(483840.0*th.sqrt(th.tensor(110.0)))
        S221[jcall1064] = th.pow(zeta_b[jcall1064,1],4.5)* \
                   th.pow(zeta_a[jcall1064,1],6.5)* \
                   rij[jcall1064]**11 * \
                  ((A22[jcall1064,8]*B22[jcall1064,0]-A22[jcall1064,10]*B22[jcall1064,2]) + \
                  2.0*(A22[jcall1064,7]*B22[jcall1064,1]-A22[jcall1064,9]*B22[jcall1064,3]) - \
                  2.0*(A22[jcall1064,6]*B22[jcall1064,2]-A22[jcall1064,8]*B22[jcall1064,4]) - \
                  6.0*(A22[jcall1064,5]*B22[jcall1064,3]-A22[jcall1064,7]*B22[jcall1064,5]) + \
                  6.0*(A22[jcall1064,3]*B22[jcall1064,5]-A22[jcall1064,5]*B22[jcall1064,7]) + \
                  2.0*(A22[jcall1064,2]*B22[jcall1064,6]-A22[jcall1064,4]*B22[jcall1064,8]) - \
                  2.0*(A22[jcall1064,1]*B22[jcall1064,7]-A22[jcall1064,3]*B22[jcall1064,9]) - \
                  (A22[jcall1064,0]*B22[jcall1064,8]-A22[jcall1064,2]*B22[jcall1064,10])) \
                  /(161280.0*th.sqrt(th.tensor(330.0)))
        S222[jcall1064] = th.pow(zeta_b[jcall1064,1],4.5)* \
                   th.pow(zeta_a[jcall1064,1],6.5)* \
                   rij[jcall1064]**11 * \
                  ((A22[jcall1064,10]-A22[jcall1064,8])*(B22[jcall1064,0]-B22[jcall1064,2]) + \
                  2.0*(A22[jcall1064,9]-A22[jcall1064,7])*(B22[jcall1064,1]-B22[jcall1064,3]) - \
                  2.0*(A22[jcall1064,8]-A22[jcall1064,6])*(B22[jcall1064,2]-B22[jcall1064,4]) - \
                  6.0*(A22[jcall1064,7]-A22[jcall1064,5])*(B22[jcall1064,3]-B22[jcall1064,5]) + \
                  6.0*(A22[jcall1064,5]-A22[jcall1064,3])*(B22[jcall1064,5]-B22[jcall1064,7]) + \
                  2.0*(A22[jcall1064,4]-A22[jcall1064,2])*(B22[jcall1064,6]-B22[jcall1064,8]) - \
                  2.0*(A22[jcall1064,3]-A22[jcall1064,1])*(B22[jcall1064,7]-B22[jcall1064,9]) - \
                  (A22[jcall1064,2]-A22[jcall1064,0])*(B22[jcall1064,8]-B22[jcall1064,10])) \
                  /(322560.0*th.sqrt(th.tensor(330.0)))








    jcall11 = (jcall==11) #ii=4
    if(jcall11.sum() != 0):
        S111[jcall11] = th.pow(zeta_b[jcall11,0],5.5)* \
                   th.pow(zeta_a[jcall11,0],6.5)* \
                   rij[jcall11]**12 * \
                  (A111[jcall11,11]*B111[jcall11,0]+1.0*B111[jcall11,1]*A111[jcall11,10]-5.0*B111[jcall11,2]*A111[jcall11,9]- \
                   5.0*B111[jcall11,3]*A111[jcall11,8]+10.0*A111[jcall11,7]*B111[jcall11,4]+10.0*B111[jcall11,5]*A111[jcall11,6]- \
                   10.0*B111[jcall11,6]*A111[jcall11,5]-10.0*B111[jcall11,7]*A111[jcall11,4]-5.0*A111[jcall11,3]*B111[jcall11,8]+5.0*A111[jcall11,2]*B111[jcall11,9]- \
                   A111[jcall11,1]*B111[jcall11,10]-A111[jcall11,0]*B111[jcall11,11])/ \
                  (th.sqrt(th.tensor(33.0))*14515200.0)
        S211[jcall11] = th.pow(zeta_b[jcall11,0],5.5)* \
                   th.pow(zeta_a[jcall11,1],6.5)* \
                   rij[jcall11]**12 * \
                  ((A211[jcall11,10]*B211[jcall11,0]+A211[jcall11,11]*B211[jcall11,1]) - \
                  5.0*(A211[jcall11,8]*B211[jcall11,2]+A211[jcall11,9]*B211[jcall11,3]) + \
                  10.0*(A211[jcall11,6]*B211[jcall11,4]+A211[jcall11,7]*B211[jcall11,5]) - \
                  10.0*(A211[jcall11,4]*B211[jcall11,6]+A211[jcall11,5]*B211[jcall11,7]) + \
                  5.0*(A211[jcall11,2]*B211[jcall11,8]+A211[jcall11,3]*B211[jcall11,9]) - \
                  1.0*(A211[jcall11,0]*B211[jcall11,10]+A211[jcall11,1]*B211[jcall11,11])) \
                  /(14515200.0*th.sqrt(th.tensor(11.0)))
        S121[jcall11] = th.pow(zeta_b[jcall11,1],5.5)* \
                   th.pow(zeta_a[jcall11,0],6.5)* \
                   rij[jcall11]**12 * \
                  ((A121[jcall11,10]*B121[jcall11,0]-A121[jcall11,11]*B121[jcall11,1]) + \
                  2.0*(A121[jcall11,9]*B121[jcall11,1]-A121[jcall11,10]*B121[jcall11,2]) -
                  3.0*(A121[jcall11,8]*B121[jcall11,2]-A121[jcall11,9]*B121[jcall11,3]) - \
                  8.0*(A121[jcall11,7]*B121[jcall11,3]-A121[jcall11,8]*B121[jcall11,4]) + \
                  2.0*(A121[jcall11,6]*B121[jcall11,4]-A121[jcall11,7]*B121[jcall11,5]) + \
                  12.0*(A121[jcall11,5]*B121[jcall11,5]-A121[jcall11,6]*B121[jcall11,6]) + \
                  2.0*(A121[jcall11,4]*B121[jcall11,6]-A121[jcall11,5]*B121[jcall11,7]) - \
                  8.0*(A121[jcall11,3]*B121[jcall11,7]-A121[jcall11,4]*B121[jcall11,8]) - \
                  3.0*(A121[jcall11,2]*B121[jcall11,8]-A121[jcall11,3]*B121[jcall11,9]) + \
                  2.0*(A121[jcall11,1]*B121[jcall11,9]-A121[jcall11,2]*B121[jcall11,10]) + \
                  1.0*(A121[jcall11,0]*B121[jcall11,10]-A121[jcall11,1]*B121[jcall11,11])) \
                  /(14515200.0*th.sqrt(th.tensor(11.0)))
        S221[jcall11] = th.pow(zeta_b[jcall11,1],5.5)* \
                   th.pow(zeta_a[jcall11,1],6.5)* \
                   rij[jcall11]**12 * \
                  ((A22[jcall11,9]*B22[jcall11,0]-A22[jcall11,11]*B22[jcall11,2]) + \
                  1.0*(A22[jcall11,8]*B22[jcall11,1]-A22[jcall11,10]*B22[jcall11,3]) - \
                  4.0*(A22[jcall11,7]*B22[jcall11,2]-A22[jcall11,9]*B22[jcall11,4]) - \
                  4.0*(A22[jcall11,6]*B22[jcall11,3]-A22[jcall11,8]*B22[jcall11,5]) + \
                  6.0*(A22[jcall11,5]*B22[jcall11,4]-A22[jcall11,7]*B22[jcall11,6]) + \
                  6.0*(A22[jcall11,4]*B22[jcall11,5]-A22[jcall11,6]*B22[jcall11,7]) - \
                  4.0*(A22[jcall11,3]*B22[jcall11,6]-A22[jcall11,5]*B22[jcall11,8]) - \
                  4.0*(A22[jcall11,2]*B22[jcall11,7]-A22[jcall11,4]*B22[jcall11,9]) + \
                  1.0*(A22[jcall11,1]*B22[jcall11,8]-A22[jcall11,3]*B22[jcall11,10]) + \
                  (A22[jcall11,0]*B22[jcall11,9]-A22[jcall11,2]*B22[jcall11,11])) \
                  /(4838400.0*th.sqrt(th.tensor(33.0)))
        S222[jcall11] = th.pow(zeta_b[jcall11,1],5.5)* \
                   th.pow(zeta_a[jcall11,1],6.5)* \
                   rij[jcall11]**12 * \
                  ((A22[jcall11,11]-A22[jcall11,9])*(B22[jcall11,0]-B22[jcall11,2]) + \
                  1.0*(A22[jcall11,10]-A22[jcall11,8])*(B22[jcall11,1]-B22[jcall11,3]) - \
                  4.0*(A22[jcall11,9]-A22[jcall11,7])*(B22[jcall11,2]-B22[jcall11,4]) - \
                  4.0*(A22[jcall11,8]-A22[jcall11,6])*(B22[jcall11,3]-B22[jcall11,5]) + \
                  6.0*(A22[jcall11,7]-A22[jcall11,5])*(B22[jcall11,4]-B22[jcall11,6]) + \
                  6.0*(A22[jcall11,6]-A22[jcall11,4])*(B22[jcall11,5]-B22[jcall11,7]) - \
                  4.0*(A22[jcall11,5]-A22[jcall11,3])*(B22[jcall11,6]-B22[jcall11,8]) - \
                  4.0*(A22[jcall11,4]-A22[jcall11,2])*(B22[jcall11,7]-B22[jcall11,9]) + \
                  1.0*(A22[jcall11,3]-A22[jcall11,1])*(B22[jcall11,8]-B22[jcall11,10]) + \
                  (A22[jcall11,2]-A22[jcall11,0])*(B22[jcall11,9]-B22[jcall11,11])) \
                  /(9676800.0*th.sqrt(th.tensor(33.0)))









############# Here we do D,S overlap #############
    S311 = th.zeros_like(S111)
    A311, B311 = SET(rij, jcallds, zeta_a[...,2],zeta_b[...,0])

    S321 = th.zeros_like(S111)
    A321, B321 = SET(rij, jcallds, zeta_a[...,2],zeta_b[...,1])

    S322 = th.zeros_like(S111)

    jcallds431 = (jcallds==431)
    if(jcallds431.sum() != 0):
        S311[jcallds431] = th.pow(zeta_b[jcallds431,0],1.5)* \
                   th.pow(zeta_a[jcallds431,2],3.5)* \
                   rij[jcallds431]**5 * \
                  ((A311[jcallds431,2]*(3.0*B311[jcallds431,0]-B311[jcallds431,2])+A311[jcallds431,4]*(3.0*B311[jcallds431,2]-B311[jcallds431,0])+4.0*A311[jcallds431,3]*B311[jcallds431,1]) - \
                  (A311[jcallds431,0]*(3.0*B311[jcallds431,2]-B311[jcallds431,4])+A311[jcallds431,2]*(3.0*B311[jcallds431,4]-B311[jcallds431,2])+4.0*A311[jcallds431,1]*B311[jcallds431,3])) \
                  /(48.0*th.sqrt(th.tensor(2.0)))

    jcallds5 = (jcallds==5)

    if(jcallds5.sum() != 0):
        S311[jcallds5] = th.pow(zeta_b[jcallds5,0],2.5)* \
                   th.pow(zeta_a[jcallds5,2],3.5)* \
                   rij[jcallds5]**6 * \
                  ((A311[jcallds5,3]*(3.0*B311[jcallds5,0]-B311[jcallds5,2])+A311[jcallds5,5]*(3.0*B311[jcallds5,2]-B311[jcallds5,0])+4.0*A311[jcallds5,4]*B311[jcallds5,1]) + \
                  (-A311[jcallds5,2]*(3.0*B311[jcallds5,1]-B311[jcallds5,3])-A311[jcallds5,4]*(3.0*B311[jcallds5,3]-B311[jcallds5,1])-4.0*A311[jcallds5,3]*B311[jcallds5,2]) - \
                  (A311[jcallds5,1]*(3.0*B311[jcallds5,2]-B311[jcallds5,4])+A311[jcallds5,3]*(3.0*B311[jcallds5,4]-B311[jcallds5,2])+4.0*A311[jcallds5,2]*B311[jcallds5,3]) - \
                  (-A311[jcallds5,0]*(3.0*B311[jcallds5,3]-B311[jcallds5,5])-A311[jcallds5,2]*(3.0*B311[jcallds5,5]-B311[jcallds5,3])-4.0*A311[jcallds5,1]*B311[jcallds5,4])) \
                  /(96.0*th.sqrt(th.tensor(6.0)))
        S321[jcallds5] = th.pow(zeta_b[jcallds5,1],2.5)* \
                   th.pow(zeta_a[jcallds5,2],3.5)* \
                   rij[jcallds5]**6 * \
                  ((A321[jcallds5,2]*(3.0*B321[jcallds5,0]-B321[jcallds5,2])+A321[jcallds5,3]*(B321[jcallds5,1]+B321[jcallds5,3])-A321[jcallds5,4]*(B321[jcallds5,0]+B321[jcallds5,2])-A321[jcallds5,5]*(3.0*B321[jcallds5,3]-B321[jcallds5,1])) - \
                  (A321[jcallds5,0]*(3.0*B321[jcallds5,2]-B321[jcallds5,4])+A321[jcallds5,1]*(B321[jcallds5,3]+B321[jcallds5,5])-A321[jcallds5,2]*(B321[jcallds5,2]+B321[jcallds5,4])-A321[jcallds5,3]*(3.0*B321[jcallds5,5]-B321[jcallds5,3]))) \
                  /(96.0*th.sqrt(th.tensor(2.0)))
        S322[jcallds5] = th.pow(zeta_b[jcallds5,1],2.5)* \
                   th.pow(zeta_a[jcallds5,2],3.5)* \
                   rij[jcallds5]**6 * \
                  (((A321[jcallds5,4]-A321[jcallds5,2])*(B321[jcallds5,0]-B321[jcallds5,2])+(A321[jcallds5,3]-A321[jcallds5,5])*(-B321[jcallds5,1]+B321[jcallds5,3])) - \
                  ((A321[jcallds5,2]-A321[jcallds5,0])*(B321[jcallds5,2]-B321[jcallds5,4])+(A321[jcallds5,1]-A321[jcallds5,3])*(-B321[jcallds5,3]+B321[jcallds5,5]))) \
                  /(32.0*th.sqrt(th.tensor(6.0)))



    jcallds6 = (jcallds==6)


    if(jcallds6.sum() != 0):
        S311[jcallds6] = th.pow(zeta_b[jcallds6,0],3.5)* \
                   th.pow(zeta_a[jcallds6,2],3.5)* \
                   rij[jcallds6]**7 * \
                  ((A311[jcallds6,4]*(3.0*B311[jcallds6,0]-B311[jcallds6,2])+A311[jcallds6,6]*(3.0*B311[jcallds6,2]-B311[jcallds6,0])+4.0*A311[jcallds6,5]*B311[jcallds6,1]) + \
                  2.0*(-A311[jcallds6,3]*(3.0*B311[jcallds6,1]-B311[jcallds6,3])-A311[jcallds6,5]*(3.0*B311[jcallds6,3]-B311[jcallds6,1])-4.0*A311[jcallds6,4]*B311[jcallds6,2]) - \
                  2.0*(-A311[jcallds6,1]*(3.0*B311[jcallds6,3]-B311[jcallds6,5])-A311[jcallds6,3]*(3.0*B311[jcallds6,5]-B311[jcallds6,3])-4.0*A311[jcallds6,2]*B311[jcallds6,4]) - \
                  (A311[jcallds6,0]*(3.0*B311[jcallds6,4]-B311[jcallds6,6])+A311[jcallds6,2]*(3.0*B311[jcallds6,6]-B311[jcallds6,4])+4.0*A311[jcallds6,1]*B311[jcallds6,5])) \
                  /(576.0*th.sqrt(th.tensor(5.0)))
        S321[jcallds6] = th.pow(zeta_b[jcallds6,1],3.5)* \
                   th.pow(zeta_a[jcallds6,2],3.5)* \
                   rij[jcallds6]**7 * \
                  ((A321[jcallds6,3]*(3.0*B321[jcallds6,0]-B321[jcallds6,2])+A321[jcallds6,4]*(B321[jcallds6,1]+B321[jcallds6,3])-A321[jcallds6,5]*(B321[jcallds6,0]+B321[jcallds6,2])-A321[jcallds6,6]*(3.0*B321[jcallds6,3]-B321[jcallds6,1])) + \
                  (-A321[jcallds6,2]*(3.0*B321[jcallds6,1]-B321[jcallds6,3])-A321[jcallds6,3]*(B321[jcallds6,2]+B321[jcallds6,4])+A321[jcallds6,4]*(B321[jcallds6,1]+B321[jcallds6,3])+A321[jcallds6,5]*(3.0*B321[jcallds6,4]-B321[jcallds6,2]))  - \
                  (A321[jcallds6,1]*(3.0*B321[jcallds6,2]-B321[jcallds6,4])+A321[jcallds6,2]*(B321[jcallds6,3]+B321[jcallds6,5])-A321[jcallds6,3]*(B321[jcallds6,2]+B321[jcallds6,4])-A321[jcallds6,4]*(3.0*B321[jcallds6,5]-B321[jcallds6,3]))  - \
                  (-A321[jcallds6,0]*(3.0*B321[jcallds6,3]-B321[jcallds6,5])-A321[jcallds6,1]*(B321[jcallds6,4]+B321[jcallds6,6])+A321[jcallds6,2]*(B321[jcallds6,3]+B321[jcallds6,5])+A321[jcallds6,3]*(3.0*B321[jcallds6,6]-B321[jcallds6,4]))) \
                 /(192.0*th.sqrt(th.tensor(15.0)))
        S322[jcallds6] = th.pow(zeta_b[jcallds6,1],3.5)* \
                   th.pow(zeta_a[jcallds6,2],3.5)* \
                   rij[jcallds6]**7 * \
                 (((A321[jcallds6,5]-A321[jcallds6,3])*(B321[jcallds6,0]-B321[jcallds6,2])+(A321[jcallds6,4]-A321[jcallds6,6])*-(B321[jcallds6,1]-B321[jcallds6,3])) + \
                  ((A321[jcallds6,4]-A321[jcallds6,2])*-(B321[jcallds6,1]-B321[jcallds6,3])+(A321[jcallds6,3]-A321[jcallds6,5])*(B321[jcallds6,2]-B321[jcallds6,4]))  - \
                  ((A321[jcallds6,3]-A321[jcallds6,1])*(B321[jcallds6,2]-B321[jcallds6,4])+(A321[jcallds6,2]-A321[jcallds6,4])*-(B321[jcallds6,3]-B321[jcallds6,5]))  - \
                  ((A321[jcallds6,2]-A321[jcallds6,0])*-(B321[jcallds6,3]-B321[jcallds6,5])+(A321[jcallds6,1]-A321[jcallds6,3])*(B321[jcallds6,4]-B321[jcallds6,6]))) \
                  /(192.0*th.sqrt(th.tensor(5.0)))




    jcallds734 = (jcallds==734)


    if(jcallds734.sum() != 0):
        S311[jcallds734] = th.pow(zeta_b[jcallds734,0],4.5)* \
                   th.pow(zeta_a[jcallds734,2],3.5)* \
                   rij[jcallds734]**8 * \
                   ((A311[jcallds734,5]*(3.0*B311[jcallds734,0]-B311[jcallds734,2])+A311[jcallds734,7]*(3.0*B311[jcallds734,2]-B311[jcallds734,0])+4.0*A311[jcallds734,6]*B311[jcallds734,1]) + \
                   3.0*(A311[jcallds734,4]*-(3.0*B311[jcallds734,1]-B311[jcallds734,3])+A311[jcallds734,6]*-(3.0*B311[jcallds734,3]-B311[jcallds734,1])-4.0*A311[jcallds734,5]*B311[jcallds734,2]) + \
                   2.0*(A311[jcallds734,3]*(3.0*B311[jcallds734,2]-B311[jcallds734,4])+A311[jcallds734,5]*(3.0*B311[jcallds734,4]-B311[jcallds734,2])+4.0*A311[jcallds734,4]*B311[jcallds734,3]) - \
                   2.0*(A311[jcallds734,2]*-(3.0*B311[jcallds734,3]-B311[jcallds734,5])+A311[jcallds734,4]*-(3.0*B311[jcallds734,5]-B311[jcallds734,3])-4.0*A311[jcallds734,3]*B311[jcallds734,4]) - \
                   3.0*(A311[jcallds734,1]*(3.0*B311[jcallds734,4]-B311[jcallds734,6])+A311[jcallds734,3]*(3.0*B311[jcallds734,6]-B311[jcallds734,4])+4.0*A311[jcallds734,2]*B311[jcallds734,5]) - \
                   1.0*(A311[jcallds734,0]*-(3.0*B311[jcallds734,5]-B311[jcallds734,7])+A311[jcallds734,2]*-(3.0*B311[jcallds734,7]-B311[jcallds734,5])-4.0*A311[jcallds734,1]*B311[jcallds734,6])) \
                   /(1152.0*th.sqrt(th.tensor(70.0)))
        S321[jcallds734] = th.pow(zeta_b[jcallds734,1],4.5)* \
                   th.pow(zeta_a[jcallds734,2],3.5)* \
                   rij[jcallds734]**8 * \
                  ((A321[jcallds734,4]*(3.0*B321[jcallds734,0]-B321[jcallds734,2])-A321[jcallds734,5]*-(B321[jcallds734,1]+B321[jcallds734,3])-A321[jcallds734,6]*(B321[jcallds734,0]+B321[jcallds734,2])+A321[jcallds734,7]*-(3.0*B321[jcallds734,3]-B321[jcallds734,1])) + \
                  2.0*(A321[jcallds734,3]*-(3.0*B321[jcallds734,1]-B321[jcallds734,3])-A321[jcallds734,4]*(B321[jcallds734,2]+B321[jcallds734,4])-A321[jcallds734,5]*-(B321[jcallds734,1]+B321[jcallds734,3])+A321[jcallds734,6]*(3.0*B321[jcallds734,4]-B321[jcallds734,2])) - \
                  2.0*(A321[jcallds734,1]*-(3.0*B321[jcallds734,3]-B321[jcallds734,5])-A321[jcallds734,2]*(B321[jcallds734,4]+B321[jcallds734,6])-A321[jcallds734,3]*-(B321[jcallds734,3]+B321[jcallds734,5])+A321[jcallds734,4]*(3.0*B321[jcallds734,6]-B321[jcallds734,4])) - \
                  1.0*(A321[jcallds734,0]*(3.0*B321[jcallds734,4]-B321[jcallds734,6])-A321[jcallds734,1]*-(B321[jcallds734,5]+B321[jcallds734,7])-A321[jcallds734,2]*(B321[jcallds734,4]+B321[jcallds734,6])+A321[jcallds734,3]*-(3.0*B321[jcallds734,7]-B321[jcallds734,5])))\
                 /(384.0*th.sqrt(th.tensor(210.0)))
        S322[jcallds734] = th.pow(zeta_b[jcallds734,1],4.5)* \
                   th.pow(zeta_a[jcallds734,2],3.5)* \
                   rij[jcallds734]**8 * \
                 (((A321[jcallds734,6]-A321[jcallds734,4])*(B321[jcallds734,0]-B321[jcallds734,2])-(A321[jcallds734,5]-A321[jcallds734,7])*(B321[jcallds734,1]-B321[jcallds734,3])) + \
                 2.0*(-(A321[jcallds734,5]-A321[jcallds734,3])*(B321[jcallds734,1]-B321[jcallds734,3])+(A321[jcallds734,4]-A321[jcallds734,6])*(B321[jcallds734,2]-B321[jcallds734,4])) - \
                 2.0*(-(A321[jcallds734,3]-A321[jcallds734,1])*(B321[jcallds734,3]-B321[jcallds734,5])+(A321[jcallds734,2]-A321[jcallds734,4])*(B321[jcallds734,4]-B321[jcallds734,6])) - \
                 1.0*((A321[jcallds734,2]-A321[jcallds734,0])*(B321[jcallds734,4]-B321[jcallds734,6])-(A321[jcallds734,1]-A321[jcallds734,3])*(B321[jcallds734,5]-B321[jcallds734,7]))) \
                  /(384.0*th.sqrt(th.tensor(70.0)))






    jcallds541 = (jcallds==541)
    if(jcallds541.sum() != 0):
        S311[jcallds541] = th.pow(zeta_b[jcallds541,0],1.5)* \
                   th.pow(zeta_a[jcallds541,2],4.5)* \
                   rij[jcallds541]**6 * \
                  ((A311[jcallds541,3]*(3.0*B311[jcallds541,0]-B311[jcallds541,2])+A311[jcallds541,5]*(3.0*B311[jcallds541,2]-B311[jcallds541,0])+4.0*A311[jcallds541,4]*B311[jcallds541,1]) - \
                  (-A311[jcallds541,2]*(3.0*B311[jcallds541,1]-B311[jcallds541,3])-A311[jcallds541,4]*(3.0*B311[jcallds541,3]-B311[jcallds541,1])-4.0*A311[jcallds541,3]*B311[jcallds541,2]) - \
                  (A311[jcallds541,1]*(3.0*B311[jcallds541,2]-B311[jcallds541,4])+A311[jcallds541,3]*(3.0*B311[jcallds541,4]-B311[jcallds541,2])+4.0*A311[jcallds541,2]*B311[jcallds541,3]) + \
                  (-A311[jcallds541,0]*(3.0*B311[jcallds541,3]-B311[jcallds541,5])-A311[jcallds541,2]*(3.0*B311[jcallds541,5]-B311[jcallds541,3])-4.0*A311[jcallds541,1]*B311[jcallds541,4])) \
                  /(192.0*th.sqrt(th.tensor(7.0)))

    jcallds642 = (jcallds==642)
    if(jcallds642.sum() != 0):
        S311[jcallds642] = th.pow(zeta_b[jcallds642,0],2.5)* \
                   th.pow(zeta_a[jcallds642,2],4.5)* \
                   rij[jcallds642]**7 * \
                  ((A311[jcallds642,4]*(3.0*B311[jcallds642,0]-B311[jcallds642,2])+A311[jcallds642,6]*(3.0*B311[jcallds642,2]-B311[jcallds642,0])+4.0*A311[jcallds642,5]*B311[jcallds642,1]) - \
                  2.0*(A311[jcallds642,2]*(3.0*B311[jcallds642,2]-B311[jcallds642,4])+A311[jcallds642,4]*(3.0*B311[jcallds642,4]-B311[jcallds642,2])+4.0*A311[jcallds642,3]*B311[jcallds642,3]) + \
                  (A311[jcallds642,0]*(3.0*B311[jcallds642,4]-B311[jcallds642,6])+A311[jcallds642,2]*(3.0*B311[jcallds642,6]-B311[jcallds642,4])+4.0*A311[jcallds642,1]*B311[jcallds642,5])) \
                  /(384.0*th.sqrt(th.tensor(21.0)))
        S321[jcallds642] = th.pow(zeta_b[jcallds642,1],2.5)* \
                   th.pow(zeta_a[jcallds642,2],4.5)* \
                   rij[jcallds642]**7 * \
                  ((A321[jcallds642,3]*(3.0*B321[jcallds642,0]-B321[jcallds642,2])+A321[jcallds642,4]*(B321[jcallds642,1]+B321[jcallds642,3])-A321[jcallds642,5]*(B321[jcallds642,0]+B321[jcallds642,2])-A321[jcallds642,6]*(3.0*B321[jcallds642,3]-B321[jcallds642,1])) - \
                  (-A321[jcallds642,2]*(3.0*B321[jcallds642,1]-B321[jcallds642,3])-A321[jcallds642,3]*(B321[jcallds642,2]+B321[jcallds642,4])+A321[jcallds642,4]*(B321[jcallds642,1]+B321[jcallds642,3])+A321[jcallds642,5]*(3.0*B321[jcallds642,4]-B321[jcallds642,2]))  - \
                  (A321[jcallds642,1]*(3.0*B321[jcallds642,2]-B321[jcallds642,4])+A321[jcallds642,2]*(B321[jcallds642,3]+B321[jcallds642,5])-A321[jcallds642,3]*(B321[jcallds642,2]+B321[jcallds642,4])-A321[jcallds642,4]*(3.0*B321[jcallds642,5]-B321[jcallds642,3]))  + \
                  (-A321[jcallds642,0]*(3.0*B321[jcallds642,3]-B321[jcallds642,5])-A321[jcallds642,1]*(B321[jcallds642,4]+B321[jcallds642,6])+A321[jcallds642,2]*(B321[jcallds642,3]+B321[jcallds642,5])+A321[jcallds642,3]*(3.0*B321[jcallds642,6]-B321[jcallds642,4]))) \
                  /(384.0*th.sqrt(th.tensor(7.0)))
        S322[jcallds642] = th.pow(zeta_b[jcallds642,1],2.5)* \
                   th.pow(zeta_a[jcallds642,2],4.5)* \
                   rij[jcallds642]**7 * \
                  (((A321[jcallds642,5]-A321[jcallds642,3])*(B321[jcallds642,0]-B321[jcallds642,2])+(A321[jcallds642,4]-A321[jcallds642,6])*(-B321[jcallds642,1]+B321[jcallds642,3])) - \
                  ((A321[jcallds642,4]-A321[jcallds642,2])*(-B321[jcallds642,1]+B321[jcallds642,3])+(A321[jcallds642,3]-A321[jcallds642,5])*(B321[jcallds642,2]-B321[jcallds642,4]))  - \
                  ((A321[jcallds642,3]-A321[jcallds642,1])*(B321[jcallds642,2]-B321[jcallds642,4])+(A321[jcallds642,2]-A321[jcallds642,4])*(-B321[jcallds642,3]+B321[jcallds642,5]))  + \
                  ((A321[jcallds642,2]-A321[jcallds642,0])*(-B321[jcallds642,3]+B321[jcallds642,5])+(A321[jcallds642,1]-A321[jcallds642,3])*(B321[jcallds642,4]-B321[jcallds642,6]))) \
                  /(128.0*th.sqrt(th.tensor(21.0)))



    jcallds7 = (jcallds==7)
    if(jcallds7.sum() != 0):
        S311[jcallds7] = th.pow(zeta_b[jcallds7,0],3.5)* \
                   th.pow(zeta_a[jcallds7,2],4.5)* \
                   rij[jcallds7]**8 * \
                  ((A311[jcallds7,5]*(3.0*B311[jcallds7,0]-B311[jcallds7,2])+A311[jcallds7,7]*(3.0*B311[jcallds7,2]-B311[jcallds7,0])+4.0*A311[jcallds7,6]*B311[jcallds7,1]) + \
                  (-A311[jcallds7,4]*(3.0*B311[jcallds7,1]-B311[jcallds7,3])-A311[jcallds7,6]*(3.0*B311[jcallds7,3]-B311[jcallds7,1])-4.0*A311[jcallds7,5]*B311[jcallds7,2]) - \
                  2.0*(A311[jcallds7,3]*(3.0*B311[jcallds7,2]-B311[jcallds7,4])+A311[jcallds7,5]*(3.0*B311[jcallds7,4]-B311[jcallds7,2])+4.0*A311[jcallds7,4]*B311[jcallds7,3]) - \
                  2.0*(-A311[jcallds7,2]*(3.0*B311[jcallds7,3]-B311[jcallds7,5])-A311[jcallds7,4]*(3.0*B311[jcallds7,5]-B311[jcallds7,3])-4.0*A311[jcallds7,3]*B311[jcallds7,4]) + \
                  (A311[jcallds7,1]*(3.0*B311[jcallds7,4]-B311[jcallds7,6])+A311[jcallds7,3]*(3.0*B311[jcallds7,6]-B311[jcallds7,4])+4.0*A311[jcallds7,2]*B311[jcallds7,5]) + \
                  (-A311[jcallds7,0]*(3.0*B311[jcallds7,5]-B311[jcallds7,7])-A311[jcallds7,2]*(3.0*B311[jcallds7,7]-B311[jcallds7,5])-4.0*A311[jcallds7,1]*B311[jcallds7,6])) \
                  /(1152.0*th.sqrt(th.tensor(70.0)))
        S321[jcallds7] = th.pow(zeta_b[jcallds7,1],3.5)* \
                   th.pow(zeta_a[jcallds7,2],4.5)* \
                   rij[jcallds7]**8 * \
                  ((A321[jcallds7,4]*(3.0*B321[jcallds7,0]-B321[jcallds7,2])+A321[jcallds7,5]*(B321[jcallds7,1]+B321[jcallds7,3])-A321[jcallds7,6]*(B321[jcallds7,0]+B321[jcallds7,2])-A321[jcallds7,7]*(3.0*B321[jcallds7,3]-B321[jcallds7,1])) - \
                  2.0*(A321[jcallds7,2]*(3.0*B321[jcallds7,2]-B321[jcallds7,4])+A321[jcallds7,3]*(B321[jcallds7,3]+B321[jcallds7,5])-A321[jcallds7,4]*(B321[jcallds7,2]+B321[jcallds7,4])-A321[jcallds7,5]*(3.0*B321[jcallds7,5]-B321[jcallds7,3]))  + \
                  (A321[jcallds7,0]*(3.0*B321[jcallds7,4]-B321[jcallds7,6])+A321[jcallds7,1]*(B321[jcallds7,5]+B321[jcallds7,7])-A321[jcallds7,2]*(B321[jcallds7,4]+B321[jcallds7,6])-A321[jcallds7,3]*(3.0*B321[jcallds7,7]-B321[jcallds7,5]))) \
                  /(384.0*th.sqrt(th.tensor(210.0)))
        S322[jcallds7] = th.pow(zeta_b[jcallds7,1],3.5)* \
                   th.pow(zeta_a[jcallds7,2],4.5)* \
                   rij[jcallds7]**8 * \
                  (((A321[jcallds7,6]-A321[jcallds7,4])*(B321[jcallds7,0]-B321[jcallds7,2])+(A321[jcallds7,5]-A321[jcallds7,7])*(-B321[jcallds7,1]+B321[jcallds7,3])) - \
                  2.0*((A321[jcallds7,4]-A321[jcallds7,2])*(B321[jcallds7,2]-B321[jcallds7,4])+(A321[jcallds7,3]-A321[jcallds7,5])*(-B321[jcallds7,3]+B321[jcallds7,5]))  + \
                  ((A321[jcallds7,2]-A321[jcallds7,0])*(B321[jcallds7,4]-B321[jcallds7,6])+(A321[jcallds7,1]-A321[jcallds7,3])*(-B321[jcallds7,5]+B321[jcallds7,7]))) \
                  /(384.0*th.sqrt(th.tensor(70.0)))



    jcallds8 = (jcallds==8)
    if(jcallds8.sum() != 0):
        S311[jcallds8] = th.pow(zeta_b[jcallds8,0],4.5)* \
                   th.pow(zeta_a[jcallds8,2],4.5)* \
                   rij[jcallds8]**9 * \
                  ((A311[jcallds8,6]*(3.0*B311[jcallds8,0]-B311[jcallds8,2])+A311[jcallds8,8]*(3.0*B311[jcallds8,2]-B311[jcallds8,0])+4.0*A311[jcallds8,7]*B311[jcallds8,1]) + \
                  2.0*(-A311[jcallds8,5]*(3.0*B311[jcallds8,1]-B311[jcallds8,3])-A311[jcallds8,7]*(3.0*B311[jcallds8,3]-B311[jcallds8,1])-4.0*A311[jcallds8,6]*B311[jcallds8,2]) - \
                  1.0*(A311[jcallds8,4]*(3.0*B311[jcallds8,2]-B311[jcallds8,4])+A311[jcallds8,6]*(3.0*B311[jcallds8,4]-B311[jcallds8,2])+4.0*A311[jcallds8,5]*B311[jcallds8,3]) - \
                  4.0*(-A311[jcallds8,3]*(3.0*B311[jcallds8,3]-B311[jcallds8,5])-A311[jcallds8,5]*(3.0*B311[jcallds8,5]-B311[jcallds8,3])-4.0*A311[jcallds8,4]*B311[jcallds8,4]) - \
                  1.0*(A311[jcallds8,2]*(3.0*B311[jcallds8,4]-B311[jcallds8,6])+A311[jcallds8,4]*(3.0*B311[jcallds8,6]-B311[jcallds8,4])+4.0*A311[jcallds8,3]*B311[jcallds8,5]) + \
                  2.0*(-A311[jcallds8,1]*(3.0*B311[jcallds8,5]-B311[jcallds8,7])-A311[jcallds8,3]*(3.0*B311[jcallds8,7]-B311[jcallds8,5])-4.0*A311[jcallds8,2]*B311[jcallds8,6]) + \
                  (A311[jcallds8,0]*(3.0*B311[jcallds8,6]-B311[jcallds8,8])+A311[jcallds8,2]*(3.0*B311[jcallds8,8]-B311[jcallds8,6])+4.0*A311[jcallds8,1]*B311[jcallds8,7])) \
                  /(32256.0*th.sqrt(th.tensor(5.0)))
        S321[jcallds8] = th.pow(zeta_b[jcallds8,1],4.5)* \
                   th.pow(zeta_a[jcallds8,2],4.5)* \
                   rij[jcallds8]**9 * \
                  ((A321[jcallds8,5]*(3.0*B321[jcallds8,0]-B321[jcallds8,2])+A321[jcallds8,6]*(B321[jcallds8,1]+B321[jcallds8,3])-A321[jcallds8,7]*(B321[jcallds8,0]+B321[jcallds8,2])-A321[jcallds8,8]*(3.0*B321[jcallds8,3]-B321[jcallds8,1])) + \
                  (-A321[jcallds8,4]*(3.0*B321[jcallds8,1]-B321[jcallds8,3])-A321[jcallds8,5]*(B321[jcallds8,2]+B321[jcallds8,4])+A321[jcallds8,6]*(B321[jcallds8,1]+B321[jcallds8,3])+A321[jcallds8,7]*(3.0*B321[jcallds8,4]-B321[jcallds8,2]))  - \
                  2.0*(A321[jcallds8,3]*(3.0*B321[jcallds8,2]-B321[jcallds8,4])+A321[jcallds8,4]*(B321[jcallds8,3]+B321[jcallds8,5])-A321[jcallds8,5]*(B321[jcallds8,2]+B321[jcallds8,4])-A321[jcallds8,6]*(3.0*B321[jcallds8,5]-B321[jcallds8,3]))  - \
                  2.0*(-A321[jcallds8,2]*(3.0*B321[jcallds8,3]-B321[jcallds8,5])-A321[jcallds8,3]*(B321[jcallds8,4]+B321[jcallds8,6])+A321[jcallds8,4]*(B321[jcallds8,3]+B321[jcallds8,5])+A321[jcallds8,5]*(3.0*B321[jcallds8,6]-B321[jcallds8,4]))  + \
                  (A321[jcallds8,1]*(3.0*B321[jcallds8,4]-B321[jcallds8,6])+A321[jcallds8,2]*(B321[jcallds8,5]+B321[jcallds8,7])-A321[jcallds8,3]*(B321[jcallds8,4]+B321[jcallds8,6])-A321[jcallds8,4]*(3.0*B321[jcallds8,7]-B321[jcallds8,5]))  + \
                  (-A321[jcallds8,0]*(3.0*B321[jcallds8,5]-B321[jcallds8,7])-A321[jcallds8,1]*(B321[jcallds8,6]+B321[jcallds8,8])+A321[jcallds8,2]*(B321[jcallds8,5]+B321[jcallds8,7])+A321[jcallds8,3]*(3.0*B321[jcallds8,8]-B321[jcallds8,6]))) \
                  /(10752.0*th.sqrt(th.tensor(15.0)))
        S322[jcallds8] = th.pow(zeta_b[jcallds8,1],4.5)* \
                   th.pow(zeta_a[jcallds8,2],4.5)* \
                   rij[jcallds8]**9 * \
                  (((A321[jcallds8,7]-A321[jcallds8,5])*(B321[jcallds8,0]-B321[jcallds8,2])+(A321[jcallds8,6]-A321[jcallds8,8])*(-B321[jcallds8,1]+B321[jcallds8,3])) + \
                  ((A321[jcallds8,6]-A321[jcallds8,4])*(-B321[jcallds8,1]+B321[jcallds8,3])+(A321[jcallds8,5]-A321[jcallds8,7])*(B321[jcallds8,2]-B321[jcallds8,4]))  - \
                  2.0*((A321[jcallds8,5]-A321[jcallds8,3])*(B321[jcallds8,2]-B321[jcallds8,4])+(A321[jcallds8,4]-A321[jcallds8,6])*(-B321[jcallds8,3]+B321[jcallds8,5]))  - \
                  2.0*((A321[jcallds8,4]-A321[jcallds8,2])*(-B321[jcallds8,3]+B321[jcallds8,5])+(A321[jcallds8,3]-A321[jcallds8,5])*(B321[jcallds8,4]-B321[jcallds8,6]))  + \
                  ((A321[jcallds8,3]-A321[jcallds8,1])*(B321[jcallds8,4]-B321[jcallds8,6])+(A321[jcallds8,2]-A321[jcallds8,4])*(-B321[jcallds8,5]+B321[jcallds8,7]))  + \
                  ((A321[jcallds8,2]-A321[jcallds8,0])*(-B321[jcallds8,5]+B321[jcallds8,7])+(A321[jcallds8,1]-A321[jcallds8,3])*(B321[jcallds8,6]-B321[jcallds8,8]))) \
                  /(10752.0*th.sqrt(th.tensor(5.0)))




    jcallds945 = (jcallds==945)
    if(jcallds945.sum() != 0):
        S311[jcallds945] = th.pow(zeta_b[jcallds945,0],5.5)* \
                   th.pow(zeta_a[jcallds945,2],4.5)* \
                   rij[jcallds945]**10 * \
                  ((A311[jcallds945,7]*(3.0*B311[jcallds945,0]-B311[jcallds945,2])+A311[jcallds945,9]*(3.0*B311[jcallds945,2]-B311[jcallds945,0])-4.0*A311[jcallds945,8]*B311[jcallds945,1]) + \
                  3.0*(A311[jcallds945,6]*-(3.0*B311[jcallds945,1]-B311[jcallds945,3])+A311[jcallds945,8]*-(3.0*B311[jcallds945,3]-B311[jcallds945,1])+4.0*A311[jcallds945,7]*B311[jcallds945,2]) + \
                  1.0*(A311[jcallds945,5]*(3.0*B311[jcallds945,2]-B311[jcallds945,4])+A311[jcallds945,7]*(3.0*B311[jcallds945,4]-B311[jcallds945,2])-4.0*A311[jcallds945,6]*B311[jcallds945,3]) - \
                  5.0*(A311[jcallds945,4]*-(3.0*B311[jcallds945,3]-B311[jcallds945,5])+A311[jcallds945,6]*-(3.0*B311[jcallds945,5]-B311[jcallds945,3])+4.0*A311[jcallds945,5]*B311[jcallds945,4]) - \
                  5.0*(A311[jcallds945,3]*(3.0*B311[jcallds945,4]-B311[jcallds945,6])+A311[jcallds945,5]*(3.0*B311[jcallds945,6]-B311[jcallds945,4])-4.0*A311[jcallds945,4]*B311[jcallds945,5]) + \
                  1.0*(A311[jcallds945,2]*-(3.0*B311[jcallds945,5]-B311[jcallds945,7])+A311[jcallds945,4]*-(3.0*B311[jcallds945,7]-B311[jcallds945,5])+4.0*A311[jcallds945,3]*B311[jcallds945,6]) + \
                  3.0*(A311[jcallds945,1]*(3.0*B311[jcallds945,6]-B311[jcallds945,8])+A311[jcallds945,3]*(3.0*B311[jcallds945,8]-B311[jcallds945,6])-4.0*A311[jcallds945,2]*B311[jcallds945,7]) + \
                  1.0*(A311[jcallds945,0]*-(3.0*B311[jcallds945,7]-B311[jcallds945,9])+A311[jcallds945,2]*-(3.0*B311[jcallds945,9]-B311[jcallds945,7])+4.0*A311[jcallds945,1]*B311[jcallds945,8])) \
                  /(483840.0*th.sqrt(th.tensor(2.0)))
        S321[jcallds945] = th.pow(zeta_b[jcallds945,1],5.5)* \
                   th.pow(zeta_a[jcallds945,2],4.5)* \
                   rij[jcallds945]**10 * \
                  ((A321[jcallds945,6]*(3.0*B321[jcallds945,0]-B321[jcallds945,2])-A321[jcallds945,7]*-(B321[jcallds945,1]+B321[jcallds945,3])-A321[jcallds945,8]*(B321[jcallds945,0]+B321[jcallds945,2])+A321[jcallds945,9]*-(3.0*B321[jcallds945,3]-B321[jcallds945,1])) + \
                  2.0*(A321[jcallds945,5]*-(3.0*B321[jcallds945,1]-B321[jcallds945,3])-A321[jcallds945,6]*(B321[jcallds945,2]+B321[jcallds945,4])-A321[jcallds945,7]*-(B321[jcallds945,1]+B321[jcallds945,3])+A321[jcallds945,8]*(3.0*B321[jcallds945,4]-B321[jcallds945,2])) - \
                  1.0*(A321[jcallds945,4]*(3.0*B321[jcallds945,2]-B321[jcallds945,4])-A321[jcallds945,5]*-(B321[jcallds945,3]+B321[jcallds945,5])-A321[jcallds945,6]*(B321[jcallds945,2]+B321[jcallds945,4])+A321[jcallds945,7]*-(3.0*B321[jcallds945,5]-B321[jcallds945,3])) - \
                  4.0*(A321[jcallds945,3]*-(3.0*B321[jcallds945,3]-B321[jcallds945,5])-A321[jcallds945,4]*(B321[jcallds945,4]+B321[jcallds945,6])-A321[jcallds945,5]*-(B321[jcallds945,3]+B321[jcallds945,5])+A321[jcallds945,6]*(3.0*B321[jcallds945,6]-B321[jcallds945,4])) - \
                  1.0*(A321[jcallds945,2]*(3.0*B321[jcallds945,4]-B321[jcallds945,6])-A321[jcallds945,3]*-(B321[jcallds945,5]+B321[jcallds945,7])-A321[jcallds945,4]*(B321[jcallds945,4]+B321[jcallds945,6])+A321[jcallds945,5]*-(3.0*B321[jcallds945,7]-B321[jcallds945,5])) + \
                  2.0*(A321[jcallds945,1]*-(3.0*B321[jcallds945,5]-B321[jcallds945,7])-A321[jcallds945,2]*(B321[jcallds945,6]+B321[jcallds945,8])-A321[jcallds945,3]*-(B321[jcallds945,5]+B321[jcallds945,7])+A321[jcallds945,4]*(3.0*B321[jcallds945,8]-B321[jcallds945,6])) + \
                  1.0*(A321[jcallds945,0]*(3.0*B321[jcallds945,6]-B321[jcallds945,8])-A321[jcallds945,1]*-(B321[jcallds945,7]+B321[jcallds945,9])-A321[jcallds945,2]*(B321[jcallds945,6]+B321[jcallds945,8])+A321[jcallds945,3]*-(3.0*B321[jcallds945,9]-B321[jcallds945,8]))) \
                  /(161280.0*th.sqrt(th.tensor(6.0)))
        S322[jcallds945] = th.pow(zeta_b[jcallds945,1],5.5)* \
                   th.pow(zeta_a[jcallds945,2],4.5)* \
                   rij[jcallds945]**10 * \
                  (((A321[jcallds945,8]-A321[jcallds945,6])*(B321[jcallds945,0]-B321[jcallds945,2])+(A321[jcallds945,7]-A321[jcallds945,9])*-(B321[jcallds945,1]-B321[jcallds945,3])) + \
                  2.0*((A321[jcallds945,7]-A321[jcallds945,5])*-(B321[jcallds945,1]-B321[jcallds945,3])+(A321[jcallds945,6]-A321[jcallds945,8])*(B321[jcallds945,2]-B321[jcallds945,4])) - \
                  1.0*((A321[jcallds945,6]-A321[jcallds945,4])*(B321[jcallds945,2]-B321[jcallds945,4])+(A321[jcallds945,5]-A321[jcallds945,7])*-(B321[jcallds945,3]-B321[jcallds945,5])) - \
                  4.0*((A321[jcallds945,5]-A321[jcallds945,3])*-(B321[jcallds945,3]-B321[jcallds945,5])+(A321[jcallds945,4]-A321[jcallds945,6])*(B321[jcallds945,4]-B321[jcallds945,6])) - \
                  1.0*((A321[jcallds945,4]-A321[jcallds945,2])*(B321[jcallds945,4]-B321[jcallds945,6])+(A321[jcallds945,3]-A321[jcallds945,5])*-(B321[jcallds945,5]-B321[jcallds945,7])) + \
                  2.0*((A321[jcallds945,3]-A321[jcallds945,1])*-(B321[jcallds945,5]-B321[jcallds945,7])+(A321[jcallds945,2]-A321[jcallds945,4])*(B321[jcallds945,6]-B321[jcallds945,8])) + \
                  1.0*((A321[jcallds945,2]-A321[jcallds945,0])*(B321[jcallds945,6]-B321[jcallds945,8])+(A321[jcallds945,1]-A321[jcallds945,3])*-(B321[jcallds945,7]-B321[jcallds945,9]))) \
                  /(161280.0*th.sqrt(th.tensor(2.0)))







    jcallds651 = (jcallds==651)

    if(jcallds651.sum() != 0):
        S311[jcallds651] = th.pow(zeta_b[jcallds651,0],1.5)* \
                   th.pow(zeta_a[jcallds651,2],5.5)* \
                   rij[jcallds651]**7 * \
                  ((A311[jcallds651,4]*(3.0*B311[jcallds651,0]-B311[jcallds651,2])+A311[jcallds651,6]*(3.0*B311[jcallds651,2]-B311[jcallds651,0])+4.0*A311[jcallds651,5]*B311[jcallds651,1]) - \
                  2.0*(-A311[jcallds651,3]*(3.0*B311[jcallds651,1]-B311[jcallds651,3])-A311[jcallds651,5]*(3.0*B311[jcallds651,3]-B311[jcallds651,1])-4.0*A311[jcallds651,4]*B311[jcallds651,2]) - \
                  2.0*(-A311[jcallds651,1]*(3.0*B311[jcallds651,3]-B311[jcallds651,5])-A311[jcallds651,3]*(3.0*B311[jcallds651,5]-B311[jcallds651,3])-4.0*A311[jcallds651,2]*B311[jcallds651,4]) + \
                  (A311[jcallds651,0]*(3.0*B311[jcallds651,4]-B311[jcallds651,6])+A311[jcallds651,2]*(3.0*B311[jcallds651,6]-B311[jcallds651,4])+4.0*A311[jcallds651,1]*B311[jcallds651,5])) \
                  /(576.0*th.sqrt(th.tensor(70.0)))

    jcallds752 = (jcallds==752)
    if(jcallds752.sum() != 0):
        S311[jcallds752] = th.pow(zeta_b[jcallds752,0],2.5)* \
                   th.pow(zeta_a[jcallds752,2],5.5)* \
                   rij[jcallds752]**8 * \
                  ((A311[jcallds752,5]*(3.0*B311[jcallds752,0]-B311[jcallds752,2])+A311[jcallds752,7]*(3.0*B311[jcallds752,2]-B311[jcallds752,0])+4.0*A311[jcallds752,6]*B311[jcallds752,1]) - \
                  (-A311[jcallds752,4]*(3.0*B311[jcallds752,1]-B311[jcallds752,3])-A311[jcallds752,6]*(3.0*B311[jcallds752,3]-B311[jcallds752,1])-4.0*A311[jcallds752,5]*B311[jcallds752,2]) - \
                  2.0*(A311[jcallds752,3]*(3.0*B311[jcallds752,2]-B311[jcallds752,4])+A311[jcallds752,5]*(3.0*B311[jcallds752,4]-B311[jcallds752,2])+4.0*A311[jcallds752,4]*B311[jcallds752,3]) + \
                  2.0*(-A311[jcallds752,2]*(3.0*B311[jcallds752,3]-B311[jcallds752,5])-A311[jcallds752,4]*(3.0*B311[jcallds752,5]-B311[jcallds752,3])-4.0*A311[jcallds752,3]*B311[jcallds752,4]) + \
                  (A311[jcallds752,1]*(3.0*B311[jcallds752,4]-B311[jcallds752,6])+A311[jcallds752,3]*(3.0*B311[jcallds752,6]-B311[jcallds752,4])+4.0*A311[jcallds752,2]*B311[jcallds752,5]) - \
                  (-A311[jcallds752,0]*(3.0*B311[jcallds752,5]-B311[jcallds752,7])-A311[jcallds752,2]*(3.0*B311[jcallds752,7]-B311[jcallds752,5])-4.0*A311[jcallds752,1]*B311[jcallds752,6])) \
                  /(1152.0*th.sqrt(th.tensor(210.0)))
        S321[jcallds752] = th.pow(zeta_b[jcallds752,1],2.5)* \
                   th.pow(zeta_a[jcallds752,2],5.5)* \
                   rij[jcallds752]**8 * \
                  ((A321[jcallds752,4]*(3.0*B321[jcallds752,0]-B321[jcallds752,2])+A321[jcallds752,5]*(B321[jcallds752,1]+B321[jcallds752,3])-A321[jcallds752,6]*(B321[jcallds752,0]+B321[jcallds752,2])-A321[jcallds752,7]*(3.0*B321[jcallds752,3]-B321[jcallds752,1])) - \
                  2.0*(-A321[jcallds752,3]*(3.0*B321[jcallds752,1]-B321[jcallds752,3])-A321[jcallds752,4]*(B321[jcallds752,2]+B321[jcallds752,4])+A321[jcallds752,5]*(B321[jcallds752,1]+B321[jcallds752,3])+A321[jcallds752,6]*(3.0*B321[jcallds752,4]-B321[jcallds752,2]))  + \
                  2.0*(-A321[jcallds752,1]*(3.0*B321[jcallds752,3]-B321[jcallds752,5])-A321[jcallds752,2]*(B321[jcallds752,4]+B321[jcallds752,6])+A321[jcallds752,3]*(B321[jcallds752,3]+B321[jcallds752,5])+A321[jcallds752,4]*(3.0*B321[jcallds752,6]-B321[jcallds752,4]))  - \
                  (A321[jcallds752,0]*(3.0*B321[jcallds752,4]-B321[jcallds752,6])+A321[jcallds752,1]*(B321[jcallds752,5]+B321[jcallds752,7])-A321[jcallds752,2]*(B321[jcallds752,4]+B321[jcallds752,6])-A321[jcallds752,3]*(3.0*B321[jcallds752,7]-B321[jcallds752,5]))) \
                  /(1152.0*th.sqrt(th.tensor(70.0)))
        S322[jcallds752] = th.pow(zeta_b[jcallds752,1],2.5)* \
                   th.pow(zeta_a[jcallds752,2],5.5)* \
                   rij[jcallds752]**8 * \
                  (((A321[jcallds752,6]-A321[jcallds752,4])*(B321[jcallds752,0]-B321[jcallds752,2])+(A321[jcallds752,5]-A321[jcallds752,7])*(-B321[jcallds752,1]+B321[jcallds752,3])) - \
                  2.0*((A321[jcallds752,5]-A321[jcallds752,3])*(-B321[jcallds752,1]+B321[jcallds752,3])+(A321[jcallds752,4]-A321[jcallds752,6])*(B321[jcallds752,2]-B321[jcallds752,4]))  + \
                  2.0*((A321[jcallds752,3]-A321[jcallds752,1])*(-B321[jcallds752,3]+B321[jcallds752,5])+(A321[jcallds752,2]-A321[jcallds752,4])*(B321[jcallds752,4]-B321[jcallds752,6]))  - \
                  ((A321[jcallds752,2]-A321[jcallds752,0])*(B321[jcallds752,4]-B321[jcallds752,6])+(A321[jcallds752,1]-A321[jcallds752,3])*(-B321[jcallds752,5]+B321[jcallds752,7]))) \
                  /(384.0*th.sqrt(th.tensor(210.0)))



    jcallds853 = (jcallds==853)
    if(jcallds853.sum() != 0):
        S311[jcallds853] = th.pow(zeta_b[jcallds853,0],3.5)* \
                   th.pow(zeta_a[jcallds853,2],5.5)* \
                   rij[jcallds853]**9 * \
                  ((A311[jcallds853,6]*(3.0*B311[jcallds853,0]-B311[jcallds853,2])+A311[jcallds853,8]*(3.0*B311[jcallds853,2]-B311[jcallds853,0])+4.0*A311[jcallds853,7]*B311[jcallds853,1]) - \
                  3.0*(A311[jcallds853,4]*(3.0*B311[jcallds853,2]-B311[jcallds853,4])+A311[jcallds853,6]*(3.0*B311[jcallds853,4]-B311[jcallds853,2])+4.0*A311[jcallds853,5]*B311[jcallds853,3]) + \
                  3.0*(A311[jcallds853,2]*(3.0*B311[jcallds853,4]-B311[jcallds853,6])+A311[jcallds853,4]*(3.0*B311[jcallds853,6]-B311[jcallds853,4])+4.0*A311[jcallds853,3]*B311[jcallds853,5]) - \
                  (A311[jcallds853,0]*(3.0*B311[jcallds853,6]-B311[jcallds853,8])+A311[jcallds853,2]*(3.0*B311[jcallds853,8]-B311[jcallds853,6])+4.0*A311[jcallds853,1]*B311[jcallds853,7])) \
                  /(34560.0*th.sqrt(th.tensor(7.0)))
        S321[jcallds853] = th.pow(zeta_b[jcallds853,1],3.5)* \
                   th.pow(zeta_a[jcallds853,2],5.5)* \
                   rij[jcallds853]**9 * \
                  ((A321[jcallds853,5]*(3.0*B321[jcallds853,0]-B321[jcallds853,2])+A321[jcallds853,6]*(B321[jcallds853,1]+B321[jcallds853,3])-A321[jcallds853,7]*(B321[jcallds853,0]+B321[jcallds853,2])-A321[jcallds853,8]*(3.0*B321[jcallds853,3]-B321[jcallds853,1])) - \
                  (-A321[jcallds853,4]*(3.0*B321[jcallds853,1]-B321[jcallds853,3])-A321[jcallds853,5]*(B321[jcallds853,2]+B321[jcallds853,4])+A321[jcallds853,6]*(B321[jcallds853,1]+B321[jcallds853,3])+A321[jcallds853,7]*(3.0*B321[jcallds853,4]-B321[jcallds853,2]))  - \
                  2.0*(A321[jcallds853,3]*(3.0*B321[jcallds853,2]-B321[jcallds853,4])+A321[jcallds853,4]*(B321[jcallds853,3]+B321[jcallds853,5])-A321[jcallds853,5]*(B321[jcallds853,2]+B321[jcallds853,4])-A321[jcallds853,6]*(3.0*B321[jcallds853,5]-B321[jcallds853,3]))  + \
                  2.0*(-A321[jcallds853,2]*(3.0*B321[jcallds853,3]-B321[jcallds853,5])-A321[jcallds853,3]*(B321[jcallds853,4]+B321[jcallds853,6])+A321[jcallds853,4]*(B321[jcallds853,3]+B321[jcallds853,5])+A321[jcallds853,5]*(3.0*B321[jcallds853,6]-B321[jcallds853,4]))  + \
                  (A321[jcallds853,1]*(3.0*B321[jcallds853,4]-B321[jcallds853,6])+A321[jcallds853,2]*(B321[jcallds853,5]+B321[jcallds853,7])-A321[jcallds853,3]*(B321[jcallds853,4]+B321[jcallds853,6])-A321[jcallds853,4]*(3.0*B321[jcallds853,7]-B321[jcallds853,5]))  - \
                  (-A321[jcallds853,0]*(3.0*B321[jcallds853,5]-B321[jcallds853,7])-A321[jcallds853,1]*(B321[jcallds853,6]+B321[jcallds853,8])+A321[jcallds853,2]*(B321[jcallds853,5]+B321[jcallds853,7])+A321[jcallds853,3]*(3.0*B321[jcallds853,8]-B321[jcallds853,6]))) \
                  /(11520.0*th.sqrt(th.tensor(21.0)))
        S322[jcallds853] = th.pow(zeta_b[jcallds853,1],3.5)* \
                   th.pow(zeta_a[jcallds853,2],5.5)* \
                   rij[jcallds853]**9 * \
                  (((A321[jcallds853,7]-A321[jcallds853,5])*(B321[jcallds853,0]-B321[jcallds853,2])+(A321[jcallds853,6]-A321[jcallds853,8])*(-B321[jcallds853,1]+B321[jcallds853,3])) - \
                  ((A321[jcallds853,6]-A321[jcallds853,4])*(-B321[jcallds853,1]+B321[jcallds853,3])+(A321[jcallds853,5]-A321[jcallds853,7])*(B321[jcallds853,2]-B321[jcallds853,4]))  - \
                  2.0*((A321[jcallds853,5]-A321[jcallds853,3])*(B321[jcallds853,2]-B321[jcallds853,4])+(A321[jcallds853,4]-A321[jcallds853,6])*(-B321[jcallds853,3]+B321[jcallds853,5]))  + \
                  2.0*((A321[jcallds853,4]-A321[jcallds853,2])*(-B321[jcallds853,3]+B321[jcallds853,5])+(A321[jcallds853,3]-A321[jcallds853,5])*(B321[jcallds853,4]-B321[jcallds853,6]))  + \
                  ((A321[jcallds853,3]-A321[jcallds853,1])*(B321[jcallds853,4]-B321[jcallds853,6])+(A321[jcallds853,2]-A321[jcallds853,4])*(-B321[jcallds853,5]+B321[jcallds853,7]))  - \
                  ((A321[jcallds853,2]-A321[jcallds853,0])*(-B321[jcallds853,5]+B321[jcallds853,7])+(A321[jcallds853,1]-A321[jcallds853,3])*(B321[jcallds853,6]-B321[jcallds853,8]))) \
                  /(11520.0*th.sqrt(th.tensor(7.0)))



    jcallds9 = (jcallds==9)
    if(jcallds9.sum() != 0):
        S311[jcallds9] = th.pow(zeta_b[jcallds9,0],4.5)* \
                   th.pow(zeta_a[jcallds9,2],5.5)* \
                   rij[jcallds9]**10 * \
                  ((A311[jcallds9,7]*(3.0*B311[jcallds9,0]-B311[jcallds9,2])+A311[jcallds9,9]*(3.0*B311[jcallds9,2]-B311[jcallds9,0])+4.0*A311[jcallds9,8]*B311[jcallds9,1]) + \
                  (-A311[jcallds9,6]*(3.0*B311[jcallds9,1]-B311[jcallds9,3])-A311[jcallds9,8]*(3.0*B311[jcallds9,3]-B311[jcallds9,1])-4.0*A311[jcallds9,7]*B311[jcallds9,2]) - \
                  3.0*(A311[jcallds9,5]*(3.0*B311[jcallds9,2]-B311[jcallds9,4])+A311[jcallds9,7]*(3.0*B311[jcallds9,4]-B311[jcallds9,2])+4.0*A311[jcallds9,6]*B311[jcallds9,3]) - \
                  3.0*(-A311[jcallds9,4]*(3.0*B311[jcallds9,3]-B311[jcallds9,5])-A311[jcallds9,6]*(3.0*B311[jcallds9,5]-B311[jcallds9,3])-4.0*A311[jcallds9,5]*B311[jcallds9,4]) + \
                  3.0*(A311[jcallds9,3]*(3.0*B311[jcallds9,4]-B311[jcallds9,6])+A311[jcallds9,5]*(3.0*B311[jcallds9,6]-B311[jcallds9,4])+4.0*A311[jcallds9,4]*B311[jcallds9,5]) + \
                  3.0*(-A311[jcallds9,2]*(3.0*B311[jcallds9,5]-B311[jcallds9,7])-A311[jcallds9,4]*(3.0*B311[jcallds9,7]-B311[jcallds9,5])-4.0*A311[jcallds9,3]*B311[jcallds9,6]) - \
                  (A311[jcallds9,1]*(3.0*B311[jcallds9,6]-B311[jcallds9,8])+A311[jcallds9,3]*(3.0*B311[jcallds9,8]-B311[jcallds9,6])+4.0*A311[jcallds9,2]*B311[jcallds9,7]) - \
                  (-A311[jcallds9,0]*(3.0*B311[jcallds9,7]-B311[jcallds9,9])-A311[jcallds9,2]*(3.0*B311[jcallds9,9]-B311[jcallds9,7])-4.0*A311[jcallds9,1]*B311[jcallds9,8])) \
                  /(483840.0*th.sqrt(th.tensor(2.0)))
        S321[jcallds9] = th.pow(zeta_b[jcallds9,1],4.5)* \
                   th.pow(zeta_a[jcallds9,2],5.5)* \
                   rij[jcallds9]**10 * \
                  ((A321[jcallds9,6]*(3.0*B321[jcallds9,0]-B321[jcallds9,2])+A321[jcallds9,7]*(B321[jcallds9,1]+B321[jcallds9,3])-A321[jcallds9,8]*(B321[jcallds9,0]+B321[jcallds9,2])-A321[jcallds9,9]*(3.0*B321[jcallds9,3]-B321[jcallds9,1])) - \
                  3.0*(A321[jcallds9,4]*(3.0*B321[jcallds9,2]-B321[jcallds9,4])+A321[jcallds9,5]*(B321[jcallds9,3]+B321[jcallds9,5])-A321[jcallds9,6]*(B321[jcallds9,2]+B321[jcallds9,4])-A321[jcallds9,7]*(3.0*B321[jcallds9,5]-B321[jcallds9,3]))  + \
                  3.0*(A321[jcallds9,2]*(3.0*B321[jcallds9,4]-B321[jcallds9,6])+A321[jcallds9,3]*(B321[jcallds9,5]+B321[jcallds9,7])-A321[jcallds9,4]*(B321[jcallds9,4]+B321[jcallds9,6])-A321[jcallds9,5]*(3.0*B321[jcallds9,7]-B321[jcallds9,5]))  - \
                  1.0*(A321[jcallds9,0]*(3.0*B321[jcallds9,6]-B321[jcallds9,8])+A321[jcallds9,1]*(B321[jcallds9,7]+B321[jcallds9,9])-A321[jcallds9,2]*(B321[jcallds9,6]+B321[jcallds9,8])-A321[jcallds9,3]*(3.0*B321[jcallds9,9]-B321[jcallds9,7]))) \
                  /(161280.0*th.sqrt(th.tensor(6.0)))
        S322[jcallds9] = th.pow(zeta_b[jcallds9,1],4.5)* \
                   th.pow(zeta_a[jcallds9,2],5.5)* \
                   rij[jcallds9]**10 * \
                  (((A321[jcallds9,8]-A321[jcallds9,6])*(B321[jcallds9,0]-B321[jcallds9,2])+(A321[jcallds9,7]-A321[jcallds9,9])*(-B321[jcallds9,1]+B321[jcallds9,3])) - \
                  3.0*((A321[jcallds9,6]-A321[jcallds9,4])*(B321[jcallds9,2]-B321[jcallds9,4])+(A321[jcallds9,5]-A321[jcallds9,7])*(-B321[jcallds9,3]+B321[jcallds9,5]))  + \
                  3.0*((A321[jcallds9,4]-A321[jcallds9,2])*(B321[jcallds9,4]-B321[jcallds9,6])+(A321[jcallds9,3]-A321[jcallds9,5])*(-B321[jcallds9,5]+B321[jcallds9,7]))  - \
                  1.0*((A321[jcallds9,2]-A321[jcallds9,0])*(B321[jcallds9,6]-B321[jcallds9,8])+(A321[jcallds9,1]-A321[jcallds9,3])*(-B321[jcallds9,7]+B321[jcallds9,9]))) \
                  /(161280.0*th.sqrt(th.tensor(2.0)))



    jcallds10 = (jcallds==10)
    if(jcallds10.sum() != 0):
        S311[jcallds10] = th.pow(zeta_b[jcallds10,0],5.5)* \
                   th.pow(zeta_a[jcallds10,2],5.5)* \
                   rij[jcallds10]**11 * \
                  ((A311[jcallds10,8]*(3.0*B311[jcallds10,0]-B311[jcallds10,2])+A311[jcallds10,10]*(3.0*B311[jcallds10,2]-B311[jcallds10,0])+4.0*A311[jcallds10,9]*B311[jcallds10,1]) + \
                  2.0*(-A311[jcallds10,7]*(3.0*B311[jcallds10,1]-B311[jcallds10,3])-A311[jcallds10,9]*(3.0*B311[jcallds10,3]-B311[jcallds10,1])-4.0*A311[jcallds10,8]*B311[jcallds10,2]) - \
                  2.0*(A311[jcallds10,6]*(3.0*B311[jcallds10,2]-B311[jcallds10,4])+A311[jcallds10,8]*(3.0*B311[jcallds10,4]-B311[jcallds10,2])+4.0*A311[jcallds10,7]*B311[jcallds10,3]) - \
                  6.0*(-A311[jcallds10,5]*(3.0*B311[jcallds10,3]-B311[jcallds10,5])-A311[jcallds10,7]*(3.0*B311[jcallds10,5]-B311[jcallds10,3])-4.0*A311[jcallds10,6]*B311[jcallds10,4]) + \
                  6.0*(-A311[jcallds10,3]*(3.0*B311[jcallds10,5]-B311[jcallds10,7])-A311[jcallds10,5]*(3.0*B311[jcallds10,7]-B311[jcallds10,5])-4.0*A311[jcallds10,4]*B311[jcallds10,6]) + \
                  2.0*(A311[jcallds10,2]*(3.0*B311[jcallds10,6]-B311[jcallds10,8])+A311[jcallds10,4]*(3.0*B311[jcallds10,8]-B311[jcallds10,6])+4.0*A311[jcallds10,3]*B311[jcallds10,7]) - \
                  2.0*(-A311[jcallds10,1]*(3.0*B311[jcallds10,7]-B311[jcallds10,9])-A311[jcallds10,3]*(3.0*B311[jcallds10,9]-B311[jcallds10,7])-4.0*A311[jcallds10,2]*B311[jcallds10,8]) - \
                  (A311[jcallds10,0]*(3.0*B311[jcallds10,8]-B311[jcallds10,10])+A311[jcallds10,2]*(3.0*B311[jcallds10,10]-B311[jcallds10,8])+4.0*A311[jcallds10,1]*B311[jcallds10,9])) \
                  /(2903040.0*th.sqrt(th.tensor(5.0)))
        S321[jcallds10] = th.pow(zeta_b[jcallds10,1],5.5)* \
                   th.pow(zeta_a[jcallds10,2],5.5)* \
                   rij[jcallds10]**11 * \
                  ((A321[jcallds10,7]*(3.0*B321[jcallds10,0]-B321[jcallds10,2])+A321[jcallds10,8]*(B321[jcallds10,1]+B321[jcallds10,3])-A321[jcallds10,9]*(B321[jcallds10,0]+B321[jcallds10,2])-A321[jcallds10,10]*(3.0*B321[jcallds10,3]-B321[jcallds10,1])) + \
                  1.0*(-A321[jcallds10,6]*(3.0*B321[jcallds10,1]-B321[jcallds10,3])-A321[jcallds10,7]*(B321[jcallds10,2]+B321[jcallds10,4])+A321[jcallds10,8]*(B321[jcallds10,1]+B321[jcallds10,3])+A321[jcallds10,9]*(3.0*B321[jcallds10,4]-B321[jcallds10,2]))  - \
                  3.0*(A321[jcallds10,5]*(3.0*B321[jcallds10,2]-B321[jcallds10,4])+A321[jcallds10,6]*(B321[jcallds10,3]+B321[jcallds10,5])-A321[jcallds10,7]*(B321[jcallds10,2]+B321[jcallds10,4])-A321[jcallds10,8]*(3.0*B321[jcallds10,5]-B321[jcallds10,3]))  - \
                  3.0*(-A321[jcallds10,4]*(3.0*B321[jcallds10,3]-B321[jcallds10,5])-A321[jcallds10,5]*(B321[jcallds10,4]+B321[jcallds10,6])+A321[jcallds10,6]*(B321[jcallds10,3]+B321[jcallds10,5])+A321[jcallds10,7]*(3.0*B321[jcallds10,6]-B321[jcallds10,4]))  + \
                  3.0*(A321[jcallds10,3]*(3.0*B321[jcallds10,4]-B321[jcallds10,6])+A321[jcallds10,4]*(B321[jcallds10,5]+B321[jcallds10,7])-A321[jcallds10,5]*(B321[jcallds10,4]+B321[jcallds10,6])-A321[jcallds10,6]*(3.0*B321[jcallds10,7]-B321[jcallds10,5]))  + \
                  3.0*(-A321[jcallds10,2]*(3.0*B321[jcallds10,5]-B321[jcallds10,7])-A321[jcallds10,3]*(B321[jcallds10,6]+B321[jcallds10,8])+A321[jcallds10,4]*(B321[jcallds10,5]+B321[jcallds10,7])+A321[jcallds10,5]*(3.0*B321[jcallds10,8]-B321[jcallds10,6]))  - \
                  1.0*(A321[jcallds10,1]*(3.0*B321[jcallds10,6]-B321[jcallds10,8])+A321[jcallds10,2]*(B321[jcallds10,7]+B321[jcallds10,9])-A321[jcallds10,3]*(B321[jcallds10,6]+B321[jcallds10,8])-A321[jcallds10,4]*(3.0*B321[jcallds10,9]-B321[jcallds10,7]))  - \
                  1.0*(-A321[jcallds10,0]*(3.0*B321[jcallds10,7]-B321[jcallds10,9])-A321[jcallds10,1]*(B321[jcallds10,8]+B321[jcallds10,10])+A321[jcallds10,2]*(B321[jcallds10,7]+B321[jcallds10,9])+A321[jcallds10,3]*(3.0*B321[jcallds10,10]-B321[jcallds10,8]))) \
                  /(967680.0*th.sqrt(th.tensor(15.0)))
        S322[jcallds10] = th.pow(zeta_b[jcallds10,1],5.5)* \
                   th.pow(zeta_a[jcallds10,2],5.5)* \
                   rij[jcallds10]**11 * \
                  (((A321[jcallds10,9]-A321[jcallds10,7])*(B321[jcallds10,0]-B321[jcallds10,2])+(A321[jcallds10,8]-A321[jcallds10,10])*(-B321[jcallds10,1]+B321[jcallds10,3])) + \
                  1.0*((A321[jcallds10,8]-A321[jcallds10,6])*(-B321[jcallds10,1]+B321[jcallds10,3])+(A321[jcallds10,7]-A321[jcallds10,9])*(B321[jcallds10,2]-B321[jcallds10,4]))  - \
                  3.0*((A321[jcallds10,7]-A321[jcallds10,5])*(B321[jcallds10,2]-B321[jcallds10,4])+(A321[jcallds10,6]-A321[jcallds10,8])*(-B321[jcallds10,3]+B321[jcallds10,5]))  - \
                  3.0*((A321[jcallds10,6]-A321[jcallds10,4])*(-B321[jcallds10,3]+B321[jcallds10,5])+(A321[jcallds10,5]-A321[jcallds10,7])*(B321[jcallds10,4]-B321[jcallds10,6]))  + \
                  3.0*((A321[jcallds10,5]-A321[jcallds10,3])*(B321[jcallds10,4]-B321[jcallds10,6])+(A321[jcallds10,4]-A321[jcallds10,6])*(-B321[jcallds10,5]+B321[jcallds10,7]))  + \
                  3.0*((A321[jcallds10,4]-A321[jcallds10,2])*(-B321[jcallds10,5]+B321[jcallds10,7])+(A321[jcallds10,3]-A321[jcallds10,5])*(B321[jcallds10,6]-B321[jcallds10,8]))  - \
                  1.0*((A321[jcallds10,3]-A321[jcallds10,1])*(B321[jcallds10,6]-B321[jcallds10,8])+(A321[jcallds10,2]-A321[jcallds10,4])*(-B321[jcallds10,7]+B321[jcallds10,9]))  - \
                  1.0*((A321[jcallds10,2]-A321[jcallds10,0])*(-B321[jcallds10,7]+B321[jcallds10,9])+(A321[jcallds10,1]-A321[jcallds10,3])*(B321[jcallds10,8]-B321[jcallds10,10]))) \
                  /(967680.0*th.sqrt(th.tensor(5.0)))



    jcallds11 = (jcallds==11)
    if(jcallds11.sum() != 0):
        S311[jcallds11] = th.pow(zeta_b[jcallds11,0],6.5)* \
                   th.pow(zeta_a[jcallds11,2],5.5)* \
                   rij[jcallds11]**12 * \
                  ((A311[jcallds11,9]*(3.0*B311[jcallds11,0]-B311[jcallds11,2])+A311[jcallds11,11]*(3.0*B311[jcallds11,2]-B311[jcallds11,0])+4.0*A311[jcallds11,10]*B311[jcallds11,1]) + \
                  3.0*(A311[jcallds11,8]*-(3.0*B311[jcallds11,1]-B311[jcallds11,3])+A311[jcallds11,10]*-(3.0*B311[jcallds11,3]-B311[jcallds11,1])-4.0*A311[jcallds11,9]*B311[jcallds11,2]) - \
                  8.0*(A311[jcallds11,6]*-(3.0*B311[jcallds11,3]-B311[jcallds11,5])+A311[jcallds11,8]*-(3.0*B311[jcallds11,5]-B311[jcallds11,3])-4.0*A311[jcallds11,7]*B311[jcallds11,4]) - \
                  6.0*(A311[jcallds11,5]*(3.0*B311[jcallds11,4]-B311[jcallds11,6])+A311[jcallds11,7]*(3.0*B311[jcallds11,6]-B311[jcallds11,4])+4.0*A311[jcallds11,6]*B311[jcallds11,5]) + \
                  6.0*(A311[jcallds11,4]*-(3.0*B311[jcallds11,5]-B311[jcallds11,7])+A311[jcallds11,6]*-(3.0*B311[jcallds11,7]-B311[jcallds11,5])-4.0*A311[jcallds11,5]*B311[jcallds11,6]) + \
                  8.0*(A311[jcallds11,3]*(3.0*B311[jcallds11,6]-B311[jcallds11,8])+A311[jcallds11,5]*(3.0*B311[jcallds11,8]-B311[jcallds11,6])+4.0*A311[jcallds11,4]*B311[jcallds11,7]) - \
                  3.0*(A311[jcallds11,1]*(3.0*B311[jcallds11,8]-B311[jcallds11,10])+A311[jcallds11,3]*(3.0*B311[jcallds11,10]-B311[jcallds11,8])+4.0*A311[jcallds11,2]*B311[jcallds11,9]) - \
                  1.0*(A311[jcallds11,0]*-(3.0*B311[jcallds11,9]-B311[jcallds11,11])+A311[jcallds11,2]*-(3.0*B311[jcallds11,11]-B311[jcallds11,9])-4.0*A311[jcallds11,1]*B311[jcallds11,10])) \
                  /(5806080.0*th.sqrt(th.tensor(165.0)))
        S321[jcallds11] = th.pow(zeta_b[jcallds11,1],6.5)* \
                   th.pow(zeta_a[jcallds11,2],5.5)* \
                   rij[jcallds11]**12 * \
                  ((A321[jcallds11,8]*(3.0*B321[jcallds11,0]-B321[jcallds11,2])-A321[jcallds11,9]*-(B321[jcallds11,1]+B321[jcallds11,3])-A321[jcallds11,10]*(B321[jcallds11,0]+B321[jcallds11,2])+A321[jcallds11,11]*-(3.0*B321[jcallds11,3]-B321[jcallds11,1])) + \
                  2.0*(A321[jcallds11,7]*-(3.0*B321[jcallds11,1]-B321[jcallds11,3])-A321[jcallds11,8]*(B321[jcallds11,2]+B321[jcallds11,4])-A321[jcallds11,9]*-(B321[jcallds11,1]+B321[jcallds11,3])+A321[jcallds11,10]*(3.0*B321[jcallds11,4]-B321[jcallds11,2])) - \
                  2.0*(A321[jcallds11,6]*(3.0*B321[jcallds11,2]-B321[jcallds11,4])-A321[jcallds11,7]*-(B321[jcallds11,3]+B321[jcallds11,5])-A321[jcallds11,8]*(B321[jcallds11,2]+B321[jcallds11,4])+A321[jcallds11,9]*-(3.0*B321[jcallds11,5]-B321[jcallds11,3])) - \
                  6.0*(A321[jcallds11,5]*-(3.0*B321[jcallds11,3]-B321[jcallds11,5])-A321[jcallds11,6]*(B321[jcallds11,4]+B321[jcallds11,6])-A321[jcallds11,7]*-(B321[jcallds11,3]+B321[jcallds11,5])+A321[jcallds11,8]*(3.0*B321[jcallds11,6]-B321[jcallds11,4])) + \
                  6.0*(A321[jcallds11,3]*-(3.0*B321[jcallds11,5]-B321[jcallds11,7])-A321[jcallds11,4]*(B321[jcallds11,6]+B321[jcallds11,8])-A321[jcallds11,5]*-(B321[jcallds11,5]+B321[jcallds11,7])+A321[jcallds11,6]*(3.0*B321[jcallds11,8]-B321[jcallds11,6])) + \
                  2.0*(A321[jcallds11,2]*(3.0*B321[jcallds11,6]-B321[jcallds11,8])-A321[jcallds11,3]*-(B321[jcallds11,7]+B321[jcallds11,9])-A321[jcallds11,4]*(B321[jcallds11,6]+B321[jcallds11,8])+A321[jcallds11,5]*-(3.0*B321[jcallds11,9]-B321[jcallds11,7])) - \
                  2.0*(A321[jcallds11,1]*-(3.0*B321[jcallds11,7]-B321[jcallds11,9])-A321[jcallds11,2]*(B321[jcallds11,8]+B321[jcallds11,10])-A321[jcallds11,3]*-(B321[jcallds11,7]+B321[jcallds11,9])+A321[jcallds11,4]*(3.0*B321[jcallds11,10]-B321[jcallds11,8])) - \
                  1.0*(A321[jcallds11,0]*(3.0*B321[jcallds11,8]-B321[jcallds11,10])-A321[jcallds11,1]*-(B321[jcallds11,9]+B321[jcallds11,11])-A321[jcallds11,2]*(B321[jcallds11,8]+B321[jcallds11,10])+A321[jcallds11,3]*-(3.0*B321[jcallds11,11]-B321[jcallds11,9])))  \
                  /(5806080.0*th.sqrt(th.tensor(55.0)))
        S322[jcallds11] = th.pow(zeta_b[jcallds11,1],6.5)* \
                   th.pow(zeta_a[jcallds11,2],5.5)* \
                   rij[jcallds11]**12 * \
                  (((A321[jcallds11,10]-A321[jcallds11,8])*(B321[jcallds11,0]-B321[jcallds11,2])+(A321[jcallds11,9]-A321[jcallds11,11])*-(B321[jcallds11,1]-B321[jcallds11,3])) + \
                  2.0*((A321[jcallds11,9]-A321[jcallds11,7])*-(B321[jcallds11,1]-B321[jcallds11,3])+(A321[jcallds11,8]-A321[jcallds11,10])*(B321[jcallds11,2]-B321[jcallds11,4])) - \
                  2.0*((A321[jcallds11,8]-A321[jcallds11,6])*(B321[jcallds11,2]-B321[jcallds11,4])+(A321[jcallds11,7]-A321[jcallds11,9])*-(B321[jcallds11,3]-B321[jcallds11,5])) - \
                  6.0*((A321[jcallds11,7]-A321[jcallds11,5])*-(B321[jcallds11,3]-B321[jcallds11,5])+(A321[jcallds11,6]-A321[jcallds11,8])*(B321[jcallds11,4]-B321[jcallds11,6])) + \
                  6.0*((A321[jcallds11,5]-A321[jcallds11,3])*-(B321[jcallds11,5]-B321[jcallds11,7])+(A321[jcallds11,4]-A321[jcallds11,6])*(B321[jcallds11,6]-B321[jcallds11,8])) + \
                  2.0*((A321[jcallds11,4]-A321[jcallds11,2])*(B321[jcallds11,6]-B321[jcallds11,8])+(A321[jcallds11,3]-A321[jcallds11,5])*-(B321[jcallds11,5]-B321[jcallds11,7])) - \
                  2.0*((A321[jcallds11,3]-A321[jcallds11,1])*-(B321[jcallds11,7]-B321[jcallds11,9])+(A321[jcallds11,2]-A321[jcallds11,4])*(B321[jcallds11,6]-B321[jcallds11,8])) - \
                  1.0*((A321[jcallds11,2]-A321[jcallds11,0])*(B321[jcallds11,8]-B321[jcallds11,10])+(A321[jcallds11,1]-A321[jcallds11,3])*-(B321[jcallds11,7]-B321[jcallds11,9]))) \
                  /(1935360.0*th.sqrt(th.tensor(165.0)))







############# Here we do S,D overlap #############
    S131 = th.zeros_like(S111)
    A131, B131 = SET(rij, jcallsd, zeta_a[...,0],zeta_b[...,2])

    S231 = th.zeros_like(S111)
    A231, B231 = SET(rij, jcallsd, zeta_a[...,1],zeta_b[...,2])

    S232 = th.zeros_like(S111)



    jcallsd6 = (jcallsd==6)
    if(jcallsd6.sum() != 0):
        S131[jcallsd6] = th.pow(zeta_b[jcallsd6,2],3.5)* \
                   th.pow(zeta_a[jcallsd6,0],3.5)* \
                   rij[jcallsd6]**7 * \
                  ((A131[jcallsd6,4]*(3.0*B131[jcallsd6,0]-B131[jcallsd6,2])+A131[jcallsd6,6]*(3.0*B131[jcallsd6,2]-B131[jcallsd6,0])-4.0*A131[jcallsd6,5]*B131[jcallsd6,1]) + \
                  2.0*(A131[jcallsd6,3]*(3.0*B131[jcallsd6,1]-B131[jcallsd6,3])+A131[jcallsd6,5]*(3.0*B131[jcallsd6,3]-B131[jcallsd6,1])-4.0*A131[jcallsd6,4]*B131[jcallsd6,2]) - \
                  2.0*(A131[jcallsd6,1]*(3.0*B131[jcallsd6,3]-B131[jcallsd6,5])+A131[jcallsd6,3]*(3.0*B131[jcallsd6,5]-B131[jcallsd6,3])-4.0*A131[jcallsd6,2]*B131[jcallsd6,4]) - \
                  (A131[jcallsd6,0]*(3.0*B131[jcallsd6,4]-B131[jcallsd6,6])+A131[jcallsd6,2]*(3.0*B131[jcallsd6,6]-B131[jcallsd6,4])-4.0*A131[jcallsd6,1]*B131[jcallsd6,5])) \
                  /(576.0*th.sqrt(th.tensor(5.0)))
        S231[jcallsd6] = th.pow(zeta_b[jcallsd6,2],3.5)* \
                   th.pow(zeta_a[jcallsd6,1],3.5)* \
                   rij[jcallsd6]**7 * \
                  ((A231[jcallsd6,3]*(3.0*B231[jcallsd6,0]-B231[jcallsd6,2])-A231[jcallsd6,4]*(B231[jcallsd6,1]+B231[jcallsd6,3])-A231[jcallsd6,5]*(B231[jcallsd6,0]+B231[jcallsd6,2])+A231[jcallsd6,6]*(3.0*B231[jcallsd6,3]-B231[jcallsd6,1])) + \
                  (A231[jcallsd6,2]*(3.0*B231[jcallsd6,1]-B231[jcallsd6,3])-A231[jcallsd6,3]*(B231[jcallsd6,2]+B231[jcallsd6,4])-A231[jcallsd6,4]*(B231[jcallsd6,1]+B231[jcallsd6,3])+A231[jcallsd6,5]*(3.0*B231[jcallsd6,4]-B231[jcallsd6,2]))  - \
                  (A231[jcallsd6,1]*(3.0*B231[jcallsd6,2]-B231[jcallsd6,4])-A231[jcallsd6,2]*(B231[jcallsd6,3]+B231[jcallsd6,5])-A231[jcallsd6,3]*(B231[jcallsd6,2]+B231[jcallsd6,4])+A231[jcallsd6,4]*(3.0*B231[jcallsd6,5]-B231[jcallsd6,3]))  - \
                  (A231[jcallsd6,0]*(3.0*B231[jcallsd6,3]-B231[jcallsd6,5])-A231[jcallsd6,1]*(B231[jcallsd6,4]+B231[jcallsd6,6])-A231[jcallsd6,2]*(B231[jcallsd6,3]+B231[jcallsd6,5])+A231[jcallsd6,3]*(3.0*B231[jcallsd6,6]-B231[jcallsd6,4]))) \
                  /(192.0*th.sqrt(th.tensor(15.0)))
        S232[jcallsd6] = th.pow(zeta_b[jcallsd6,2],3.5)* \
                   th.pow(zeta_a[jcallsd6,1],3.5)* \
                   rij[jcallsd6]**7 * \
                  (((A231[jcallsd6,5]-A231[jcallsd6,3])*(B231[jcallsd6,0]-B231[jcallsd6,2])+(A231[jcallsd6,4]-A231[jcallsd6,6])*(B231[jcallsd6,1]-B231[jcallsd6,3])) + \
                  ((A231[jcallsd6,4]-A231[jcallsd6,2])*(B231[jcallsd6,1]-B231[jcallsd6,3])+(A231[jcallsd6,3]-A231[jcallsd6,5])*(B231[jcallsd6,2]-B231[jcallsd6,4]))  - \
                  ((A231[jcallsd6,3]-A231[jcallsd6,1])*(B231[jcallsd6,2]-B231[jcallsd6,4])+(A231[jcallsd6,2]-A231[jcallsd6,4])*(B231[jcallsd6,3]-B231[jcallsd6,5]))  - \
                  ((A231[jcallsd6,2]-A231[jcallsd6,0])*(B231[jcallsd6,3]-B231[jcallsd6,5])+(A231[jcallsd6,1]-A231[jcallsd6,3])*(B231[jcallsd6,4]-B231[jcallsd6,6]))) \
                  /(192.0*th.sqrt(th.tensor(5.0)))



    jcallsd7 = (jcallsd==7)
    if(jcallsd7.sum() != 0):
        S131[jcallsd7] = th.pow(zeta_b[jcallsd7,2],3.5)* \
                   th.pow(zeta_a[jcallsd7,0],4.5)* \
                   rij[jcallsd7]**8 * \
                  ((A131[jcallsd7,5]*(3.0*B131[jcallsd7,0]-B131[jcallsd7,2])+A131[jcallsd7,7]*(3.0*B131[jcallsd7,2]-B131[jcallsd7,0])-4.0*A131[jcallsd7,6]*B131[jcallsd7,1]) + \
                  3.0*(A131[jcallsd7,4]*(3.0*B131[jcallsd7,1]-B131[jcallsd7,3])+A131[jcallsd7,6]*(3.0*B131[jcallsd7,3]-B131[jcallsd7,1])-4.0*A131[jcallsd7,5]*B131[jcallsd7,2]) + \
                  2.0*(A131[jcallsd7,3]*(3.0*B131[jcallsd7,2]-B131[jcallsd7,4])+A131[jcallsd7,5]*(3.0*B131[jcallsd7,4]-B131[jcallsd7,2])-4.0*A131[jcallsd7,4]*B131[jcallsd7,3]) - \
                  2.0*(A131[jcallsd7,2]*(3.0*B131[jcallsd7,3]-B131[jcallsd7,5])+A131[jcallsd7,4]*(3.0*B131[jcallsd7,5]-B131[jcallsd7,3])-4.0*A131[jcallsd7,3]*B131[jcallsd7,4]) - \
                  3.0*(A131[jcallsd7,1]*(3.0*B131[jcallsd7,4]-B131[jcallsd7,6])+A131[jcallsd7,3]*(3.0*B131[jcallsd7,6]-B131[jcallsd7,4])-4.0*A131[jcallsd7,2]*B131[jcallsd7,5]) - \
                  (A131[jcallsd7,0]*(3.0*B131[jcallsd7,5]-B131[jcallsd7,7])+A131[jcallsd7,2]*(3.0*B131[jcallsd7,7]-B131[jcallsd7,5])-4.0*A131[jcallsd7,1]*B131[jcallsd7,6])) \
                  /(1152.0*th.sqrt(th.tensor(70.0)))
        S231[jcallsd7] = th.pow(zeta_b[jcallsd7,2],3.5)* \
                   th.pow(zeta_a[jcallsd7,1],4.5)* \
                   rij[jcallsd7]**8 * \
                  ((A231[jcallsd7,4]*(3.0*B231[jcallsd7,0]-B231[jcallsd7,2])-A231[jcallsd7,5]*(B231[jcallsd7,1]+B231[jcallsd7,3])-A231[jcallsd7,6]*(B231[jcallsd7,0]+B231[jcallsd7,2])+A231[jcallsd7,7]*(3.0*B231[jcallsd7,3]-B231[jcallsd7,1])) + \
                  2.0*(A231[jcallsd7,3]*(3.0*B231[jcallsd7,1]-B231[jcallsd7,3])-A231[jcallsd7,4]*(B231[jcallsd7,2]+B231[jcallsd7,4])-A231[jcallsd7,5]*(B231[jcallsd7,1]+B231[jcallsd7,3])+A231[jcallsd7,6]*(3.0*B231[jcallsd7,4]-B231[jcallsd7,2]))  - \
                  2.0*(A231[jcallsd7,1]*(3.0*B231[jcallsd7,3]-B231[jcallsd7,5])-A231[jcallsd7,2]*(B231[jcallsd7,4]+B231[jcallsd7,6])-A231[jcallsd7,3]*(B231[jcallsd7,3]+B231[jcallsd7,5])+A231[jcallsd7,4]*(3.0*B231[jcallsd7,6]-B231[jcallsd7,4]))  - \
                  (A231[jcallsd7,0]*(3.0*B231[jcallsd7,4]-B231[jcallsd7,6])-A231[jcallsd7,1]*(B231[jcallsd7,5]+B231[jcallsd7,7])-A231[jcallsd7,2]*(B231[jcallsd7,4]+B231[jcallsd7,6])+A231[jcallsd7,3]*(3.0*B231[jcallsd7,7]-B231[jcallsd7,5]))) \
                  /(384.0*th.sqrt(th.tensor(210.0)))
        S232[jcallsd7] = th.pow(zeta_b[jcallsd7,2],3.5)* \
                   th.pow(zeta_a[jcallsd7,1],4.5)* \
                   rij[jcallsd7]**8 * \
                  (((A231[jcallsd7,6]-A231[jcallsd7,4])*(B231[jcallsd7,0]-B231[jcallsd7,2])+(A231[jcallsd7,5]-A231[jcallsd7,7])*(B231[jcallsd7,1]-B231[jcallsd7,3])) + \
                  2.0*((A231[jcallsd7,5]-A231[jcallsd7,3])*(B231[jcallsd7,1]-B231[jcallsd7,3])+(A231[jcallsd7,4]-A231[jcallsd7,6])*(B231[jcallsd7,2]-B231[jcallsd7,4]))  - \
                  2.0*((A231[jcallsd7,3]-A231[jcallsd7,1])*(B231[jcallsd7,3]-B231[jcallsd7,5])+(A231[jcallsd7,2]-A231[jcallsd7,4])*(B231[jcallsd7,4]-B231[jcallsd7,6]))  - \
                  ((A231[jcallsd7,2]-A231[jcallsd7,0])*(B231[jcallsd7,4]-B231[jcallsd7,6])+(A231[jcallsd7,1]-A231[jcallsd7,3])*(B231[jcallsd7,5]-B231[jcallsd7,7]))) \
                  /(384.0*th.sqrt(th.tensor(70.0)))



    jcallsd8 = (jcallsd==8)
    if(jcallsd8.sum() != 0):
        S131[jcallsd8] = th.pow(zeta_b[jcallsd8,2],4.5)* \
                   th.pow(zeta_a[jcallsd8,0],4.5)* \
                   rij[jcallsd8]**9 * \
                  ((A131[jcallsd8,6]*(3.0*B131[jcallsd8,0]-B131[jcallsd8,2])+A131[jcallsd8,8]*(3.0*B131[jcallsd8,2]-B131[jcallsd8,0])-4.0*A131[jcallsd8,7]*B131[jcallsd8,1]) + \
                  2.0*(A131[jcallsd8,5]*(3.0*B131[jcallsd8,1]-B131[jcallsd8,3])+A131[jcallsd8,7]*(3.0*B131[jcallsd8,3]-B131[jcallsd8,1])-4.0*A131[jcallsd8,6]*B131[jcallsd8,2]) - \
                  1.0*(A131[jcallsd8,4]*(3.0*B131[jcallsd8,2]-B131[jcallsd8,4])+A131[jcallsd8,6]*(3.0*B131[jcallsd8,4]-B131[jcallsd8,2])-4.0*A131[jcallsd8,5]*B131[jcallsd8,3]) - \
                  4.0*(A131[jcallsd8,3]*(3.0*B131[jcallsd8,3]-B131[jcallsd8,5])+A131[jcallsd8,5]*(3.0*B131[jcallsd8,5]-B131[jcallsd8,3])-4.0*A131[jcallsd8,4]*B131[jcallsd8,4]) - \
                  1.0*(A131[jcallsd8,2]*(3.0*B131[jcallsd8,4]-B131[jcallsd8,6])+A131[jcallsd8,4]*(3.0*B131[jcallsd8,6]-B131[jcallsd8,4])-4.0*A131[jcallsd8,3]*B131[jcallsd8,5]) + \
                  2.0*(A131[jcallsd8,1]*(3.0*B131[jcallsd8,5]-B131[jcallsd8,7])+A131[jcallsd8,3]*(3.0*B131[jcallsd8,7]-B131[jcallsd8,5])-4.0*A131[jcallsd8,2]*B131[jcallsd8,6]) + \
                  (A131[jcallsd8,0]*(3.0*B131[jcallsd8,6]-B131[jcallsd8,8])+A131[jcallsd8,2]*(3.0*B131[jcallsd8,8]-B131[jcallsd8,6])-4.0*A131[jcallsd8,1]*B131[jcallsd8,7])) \
                  /(32256.0*th.sqrt(th.tensor(5.0)))
        S231[jcallsd8] = th.pow(zeta_b[jcallsd8,2],4.5)* \
                   th.pow(zeta_a[jcallsd8,1],4.5)* \
                   rij[jcallsd8]**9 * \
                  ((A231[jcallsd8,5]*(3.0*B231[jcallsd8,0]-B231[jcallsd8,2])-A231[jcallsd8,6]*(B231[jcallsd8,1]+B231[jcallsd8,3])-A231[jcallsd8,7]*(B231[jcallsd8,0]+B231[jcallsd8,2])+A231[jcallsd8,8]*(3.0*B231[jcallsd8,3]-B231[jcallsd8,1])) + \
                  (A231[jcallsd8,4]*(3.0*B231[jcallsd8,1]-B231[jcallsd8,3])-A231[jcallsd8,5]*(B231[jcallsd8,2]+B231[jcallsd8,4])-A231[jcallsd8,6]*(B231[jcallsd8,1]+B231[jcallsd8,3])+A231[jcallsd8,7]*(3.0*B231[jcallsd8,4]-B231[jcallsd8,2]))  - \
                  2.0*(A231[jcallsd8,3]*(3.0*B231[jcallsd8,2]-B231[jcallsd8,4])-A231[jcallsd8,4]*(B231[jcallsd8,3]+B231[jcallsd8,5])-A231[jcallsd8,5]*(B231[jcallsd8,2]+B231[jcallsd8,4])+A231[jcallsd8,6]*(3.0*B231[jcallsd8,5]-B231[jcallsd8,3]))  - \
                  2.0*(A231[jcallsd8,2]*(3.0*B231[jcallsd8,3]-B231[jcallsd8,5])-A231[jcallsd8,3]*(B231[jcallsd8,4]+B231[jcallsd8,6])-A231[jcallsd8,4]*(B231[jcallsd8,3]+B231[jcallsd8,5])+A231[jcallsd8,5]*(3.0*B231[jcallsd8,6]-B231[jcallsd8,4]))  + \
                  (A231[jcallsd8,1]*(3.0*B231[jcallsd8,4]-B231[jcallsd8,6])-A231[jcallsd8,2]*(B231[jcallsd8,5]+B231[jcallsd8,7])-A231[jcallsd8,3]*(B231[jcallsd8,4]+B231[jcallsd8,6])+A231[jcallsd8,4]*(3.0*B231[jcallsd8,7]-B231[jcallsd8,5]))  + \
                  (A231[jcallsd8,0]*(3.0*B231[jcallsd8,5]-B231[jcallsd8,7])-A231[jcallsd8,1]*(B231[jcallsd8,6]+B231[jcallsd8,8])-A231[jcallsd8,2]*(B231[jcallsd8,5]+B231[jcallsd8,7])+A231[jcallsd8,3]*(3.0*B231[jcallsd8,8]-B231[jcallsd8,6]))) \
                  /(10752.0*th.sqrt(th.tensor(15.0)))
        S232[jcallsd8] = th.pow(zeta_b[jcallsd8,2],4.5)* \
                   th.pow(zeta_a[jcallsd8,1],4.5)* \
                   rij[jcallsd8]**9 * \
                  (((A231[jcallsd8,7]-A231[jcallsd8,5])*(B231[jcallsd8,0]-B231[jcallsd8,2])+(A231[jcallsd8,6]-A231[jcallsd8,8])*(B231[jcallsd8,1]-B231[jcallsd8,3])) + \
                  ((A231[jcallsd8,6]-A231[jcallsd8,4])*(B231[jcallsd8,1]-B231[jcallsd8,3])+(A231[jcallsd8,5]-A231[jcallsd8,7])*(B231[jcallsd8,2]-B231[jcallsd8,4]))  - \
                  2.0*((A231[jcallsd8,5]-A231[jcallsd8,3])*(B231[jcallsd8,2]-B231[jcallsd8,4])+(A231[jcallsd8,4]-A231[jcallsd8,6])*(B231[jcallsd8,3]-B231[jcallsd8,5]))  - \
                  2.0*((A231[jcallsd8,4]-A231[jcallsd8,2])*(B231[jcallsd8,3]-B231[jcallsd8,5])+(A231[jcallsd8,3]-A231[jcallsd8,5])*(B231[jcallsd8,4]-B231[jcallsd8,6]))  + \
                  ((A231[jcallsd8,3]-A231[jcallsd8,1])*(B231[jcallsd8,4]-B231[jcallsd8,6])+(A231[jcallsd8,2]-A231[jcallsd8,4])*(B231[jcallsd8,5]-B231[jcallsd8,7]))  + \
                  ((A231[jcallsd8,2]-A231[jcallsd8,0])*(B231[jcallsd8,5]-B231[jcallsd8,7])+(A231[jcallsd8,1]-A231[jcallsd8,3])*(B231[jcallsd8,6]-B231[jcallsd8,8]))) \
                  /(10752.0*th.sqrt(th.tensor(5.0)))



    jcallsd853 = (jcallsd==853)

    if(jcallsd853.sum() != 0):
        S131[jcallsd853] = th.pow(zeta_b[jcallsd853,2],3.5)* \
                   th.pow(zeta_a[jcallsd853,0],5.5)* \
                   rij[jcallsd853]**9 * \
                  ((A131[jcallsd853,6]*(3.0*B131[jcallsd853,0]-B131[jcallsd853,2])+A131[jcallsd853,8]*(3.0*B131[jcallsd853,2]-B131[jcallsd853,0])-4.0*A131[jcallsd853,7]*B131[jcallsd853,1]) + \
                  4.0*(A131[jcallsd853,5]*(3.0*B131[jcallsd853,1]-B131[jcallsd853,3])+A131[jcallsd853,7]*(3.0*B131[jcallsd853,3]-B131[jcallsd853,1])-4.0*A131[jcallsd853,6]*B131[jcallsd853,2]) + \
                  5.0*(A131[jcallsd853,4]*(3.0*B131[jcallsd853,2]-B131[jcallsd853,4])+A131[jcallsd853,6]*(3.0*B131[jcallsd853,4]-B131[jcallsd853,2])-4.0*A131[jcallsd853,5]*B131[jcallsd853,3]) - \
                  5.0*(A131[jcallsd853,2]*(3.0*B131[jcallsd853,4]-B131[jcallsd853,6])+A131[jcallsd853,4]*(3.0*B131[jcallsd853,6]-B131[jcallsd853,4])-4.0*A131[jcallsd853,3]*B131[jcallsd853,5]) - \
                  4.0*(A131[jcallsd853,1]*(3.0*B131[jcallsd853,5]-B131[jcallsd853,7])+A131[jcallsd853,3]*(3.0*B131[jcallsd853,7]-B131[jcallsd853,5])-4.0*A131[jcallsd853,2]*B131[jcallsd853,6]) - \
                  (A131[jcallsd853,0]*(3.0*B131[jcallsd853,6]-B131[jcallsd853,8])+A131[jcallsd853,2]*(3.0*B131[jcallsd853,8]-B131[jcallsd853,6])-4.0*A131[jcallsd853,1]*B131[jcallsd853,7])) \
                  /(34560.0*th.sqrt(th.tensor(7.0)))
        S231[jcallsd853] = th.pow(zeta_b[jcallsd853,2],3.5)* \
                   th.pow(zeta_a[jcallsd853,1],5.5)* \
                   rij[jcallsd853]**9 * \
                  ((A231[jcallsd853,5]*(3.0*B231[jcallsd853,0]-B231[jcallsd853,2])-A231[jcallsd853,6]*(B231[jcallsd853,1]+B231[jcallsd853,3])-A231[jcallsd853,7]*(B231[jcallsd853,0]+B231[jcallsd853,2])+A231[jcallsd853,8]*(3.0*B231[jcallsd853,3]-B231[jcallsd853,1])) + \
                  3.0*(A231[jcallsd853,4]*(3.0*B231[jcallsd853,1]-B231[jcallsd853,3])-A231[jcallsd853,5]*(B231[jcallsd853,2]+B231[jcallsd853,4])-A231[jcallsd853,6]*(B231[jcallsd853,1]+B231[jcallsd853,3])+A231[jcallsd853,7]*(3.0*B231[jcallsd853,4]-B231[jcallsd853,2]))  + \
                  2.0*(A231[jcallsd853,3]*(3.0*B231[jcallsd853,2]-B231[jcallsd853,4])-A231[jcallsd853,4]*(B231[jcallsd853,3]+B231[jcallsd853,5])-A231[jcallsd853,5]*(B231[jcallsd853,2]+B231[jcallsd853,4])+A231[jcallsd853,6]*(3.0*B231[jcallsd853,5]-B231[jcallsd853,3]))  - \
                  2.0*(A231[jcallsd853,2]*(3.0*B231[jcallsd853,3]-B231[jcallsd853,5])-A231[jcallsd853,3]*(B231[jcallsd853,4]+B231[jcallsd853,6])-A231[jcallsd853,4]*(B231[jcallsd853,3]+B231[jcallsd853,5])+A231[jcallsd853,5]*(3.0*B231[jcallsd853,6]-B231[jcallsd853,4]))  - \
                  3.0*(A231[jcallsd853,1]*(3.0*B231[jcallsd853,4]-B231[jcallsd853,6])-A231[jcallsd853,2]*(B231[jcallsd853,5]+B231[jcallsd853,7])-A231[jcallsd853,3]*(B231[jcallsd853,4]+B231[jcallsd853,6])+A231[jcallsd853,4]*(3.0*B231[jcallsd853,7]-B231[jcallsd853,5]))  - \
                  (A231[jcallsd853,0]*(3.0*B231[jcallsd853,5]-B231[jcallsd853,7])-A231[jcallsd853,1]*(B231[jcallsd853,6]+B231[jcallsd853,8])-A231[jcallsd853,2]*(B231[jcallsd853,5]+B231[jcallsd853,7])+A231[jcallsd853,3]*(3.0*B231[jcallsd853,8]-B231[jcallsd853,6]))) \
                  /(11520.0*th.sqrt(th.tensor(21.0)))
        S232[jcallsd853] = th.pow(zeta_b[jcallsd853,2],3.5)* \
                   th.pow(zeta_a[jcallsd853,1],5.5)* \
                   rij[jcallsd853]**9 * \
                  (((A231[jcallsd853,7]-A231[jcallsd853,5])*(B231[jcallsd853,0]-B231[jcallsd853,2])+(A231[jcallsd853,6]-A231[jcallsd853,8])*(B231[jcallsd853,1]-B231[jcallsd853,3])) + \
                  3.0*((A231[jcallsd853,6]-A231[jcallsd853,4])*(B231[jcallsd853,1]-B231[jcallsd853,3])+(A231[jcallsd853,5]-A231[jcallsd853,7])*(B231[jcallsd853,2]-B231[jcallsd853,4]))  + \
                  2.0*((A231[jcallsd853,5]-A231[jcallsd853,3])*(B231[jcallsd853,2]-B231[jcallsd853,4])+(A231[jcallsd853,4]-A231[jcallsd853,6])*(B231[jcallsd853,3]-B231[jcallsd853,5]))  - \
                  2.0*((A231[jcallsd853,4]-A231[jcallsd853,2])*(B231[jcallsd853,3]-B231[jcallsd853,5])+(A231[jcallsd853,3]-A231[jcallsd853,5])*(B231[jcallsd853,4]-B231[jcallsd853,6]))  - \
                  3.0*((A231[jcallsd853,3]-A231[jcallsd853,1])*(B231[jcallsd853,4]-B231[jcallsd853,6])+(A231[jcallsd853,2]-A231[jcallsd853,4])*(B231[jcallsd853,5]-B231[jcallsd853,7]))  - \
                  ((A231[jcallsd853,2]-A231[jcallsd853,0])*(B231[jcallsd853,5]-B231[jcallsd853,7])+(A231[jcallsd853,1]-A231[jcallsd853,3])*(B231[jcallsd853,6]-B231[jcallsd853,8]))) \
                  /(11520.0*th.sqrt(th.tensor(7.0)))




    jcallsd9 = (jcallsd==9)
    if(jcallsd9.sum() != 0):
        S131[jcallsd9] = th.pow(zeta_b[jcallsd9,2],4.5)* \
                   th.pow(zeta_a[jcallsd9,0],5.5)* \
                   rij[jcallsd9]**10 * \
                  ((A131[jcallsd9,7]*(3.0*B131[jcallsd9,0]-B131[jcallsd9,2])+A131[jcallsd9,9]*(3.0*B131[jcallsd9,2]-B131[jcallsd9,0])-4.0*A131[jcallsd9,8]*B131[jcallsd9,1]) + \
                  3.0*(A131[jcallsd9,6]*(3.0*B131[jcallsd9,1]-B131[jcallsd9,3])+A131[jcallsd9,8]*(3.0*B131[jcallsd9,3]-B131[jcallsd9,1])-4.0*A131[jcallsd9,7]*B131[jcallsd9,2]) + \
                  1.0*(A131[jcallsd9,5]*(3.0*B131[jcallsd9,2]-B131[jcallsd9,4])+A131[jcallsd9,7]*(3.0*B131[jcallsd9,4]-B131[jcallsd9,2])-4.0*A131[jcallsd9,6]*B131[jcallsd9,3]) - \
                  5.0*(A131[jcallsd9,4]*(3.0*B131[jcallsd9,3]-B131[jcallsd9,5])+A131[jcallsd9,6]*(3.0*B131[jcallsd9,5]-B131[jcallsd9,3])-4.0*A131[jcallsd9,5]*B131[jcallsd9,4]) - \
                  5.0*(A131[jcallsd9,3]*(3.0*B131[jcallsd9,4]-B131[jcallsd9,6])+A131[jcallsd9,5]*(3.0*B131[jcallsd9,6]-B131[jcallsd9,4])-4.0*A131[jcallsd9,4]*B131[jcallsd9,5]) + \
                  1.0*(A131[jcallsd9,2]*(3.0*B131[jcallsd9,5]-B131[jcallsd9,7])+A131[jcallsd9,4]*(3.0*B131[jcallsd9,7]-B131[jcallsd9,5])-4.0*A131[jcallsd9,3]*B131[jcallsd9,6]) + \
                  3.0*(A131[jcallsd9,1]*(3.0*B131[jcallsd9,6]-B131[jcallsd9,8])+A131[jcallsd9,3]*(3.0*B131[jcallsd9,8]-B131[jcallsd9,6])-4.0*A131[jcallsd9,2]*B131[jcallsd9,7]) + \
                  (A131[jcallsd9,0]*(3.0*B131[jcallsd9,7]-B131[jcallsd9,9])+A131[jcallsd9,2]*(3.0*B131[jcallsd9,9]-B131[jcallsd9,7])-4.0*A131[jcallsd9,1]*B131[jcallsd9,8])) \
                  /(483840.0*th.sqrt(th.tensor(2.0)))
        S231[jcallsd9] = th.pow(zeta_b[jcallsd9,2],4.5)* \
                   th.pow(zeta_a[jcallsd9,1],5.5)* \
                   rij[jcallsd9]**10 * \
                  ((A231[jcallsd9,6]*(3.0*B231[jcallsd9,0]-B231[jcallsd9,2])-A231[jcallsd9,7]*(B231[jcallsd9,1]+B231[jcallsd9,3])-A231[jcallsd9,8]*(B231[jcallsd9,0]+B231[jcallsd9,2])+A231[jcallsd9,9]*(3.0*B231[jcallsd9,3]-B231[jcallsd9,1])) + \
                  2.0*(A231[jcallsd9,5]*(3.0*B231[jcallsd9,1]-B231[jcallsd9,3])-A231[jcallsd9,6]*(B231[jcallsd9,2]+B231[jcallsd9,4])-A231[jcallsd9,7]*(B231[jcallsd9,1]+B231[jcallsd9,3])+A231[jcallsd9,8]*(3.0*B231[jcallsd9,4]-B231[jcallsd9,2]))  - \
                  1.0*(A231[jcallsd9,4]*(3.0*B231[jcallsd9,2]-B231[jcallsd9,4])-A231[jcallsd9,5]*(B231[jcallsd9,3]+B231[jcallsd9,5])-A231[jcallsd9,6]*(B231[jcallsd9,2]+B231[jcallsd9,4])+A231[jcallsd9,7]*(3.0*B231[jcallsd9,5]-B231[jcallsd9,3]))  - \
                  4.0*(A231[jcallsd9,3]*(3.0*B231[jcallsd9,3]-B231[jcallsd9,5])-A231[jcallsd9,4]*(B231[jcallsd9,4]+B231[jcallsd9,6])-A231[jcallsd9,5]*(B231[jcallsd9,3]+B231[jcallsd9,5])+A231[jcallsd9,6]*(3.0*B231[jcallsd9,6]-B231[jcallsd9,4]))  - \
                  1.0*(A231[jcallsd9,2]*(3.0*B231[jcallsd9,4]-B231[jcallsd9,6])-A231[jcallsd9,3]*(B231[jcallsd9,5]+B231[jcallsd9,7])-A231[jcallsd9,4]*(B231[jcallsd9,4]+B231[jcallsd9,6])+A231[jcallsd9,5]*(3.0*B231[jcallsd9,7]-B231[jcallsd9,5]))  + \
                  2.0*(A231[jcallsd9,1]*(3.0*B231[jcallsd9,5]-B231[jcallsd9,7])-A231[jcallsd9,2]*(B231[jcallsd9,6]+B231[jcallsd9,8])-A231[jcallsd9,3]*(B231[jcallsd9,5]+B231[jcallsd9,7])+A231[jcallsd9,4]*(3.0*B231[jcallsd9,8]-B231[jcallsd9,6]))  + \
                  1.0*(A231[jcallsd9,0]*(3.0*B231[jcallsd9,6]-B231[jcallsd9,8])-A231[jcallsd9,1]*(B231[jcallsd9,7]+B231[jcallsd9,9])-A231[jcallsd9,2]*(B231[jcallsd9,6]+B231[jcallsd9,8])+A231[jcallsd9,3]*(3.0*B231[jcallsd9,9]-B231[jcallsd9,7]))) \
                  /(161280.0*th.sqrt(th.tensor(6.0)))
        S232[jcallsd9] = th.pow(zeta_b[jcallsd9,2],4.5)* \
                   th.pow(zeta_a[jcallsd9,1],5.5)* \
                   rij[jcallsd9]**10 * \
                  (((A231[jcallsd9,8]-A231[jcallsd9,6])*(B231[jcallsd9,0]-B231[jcallsd9,2])+(A231[jcallsd9,7]-A231[jcallsd9,9])*(B231[jcallsd9,1]-B231[jcallsd9,3])) + \
                  2.0*((A231[jcallsd9,7]-A231[jcallsd9,5])*(B231[jcallsd9,1]-B231[jcallsd9,3])+(A231[jcallsd9,6]-A231[jcallsd9,8])*(B231[jcallsd9,2]-B231[jcallsd9,4]))  - \
                  1.0*((A231[jcallsd9,6]-A231[jcallsd9,4])*(B231[jcallsd9,2]-B231[jcallsd9,4])+(A231[jcallsd9,5]-A231[jcallsd9,7])*(B231[jcallsd9,3]-B231[jcallsd9,5]))  - \
                  4.0*((A231[jcallsd9,5]-A231[jcallsd9,3])*(B231[jcallsd9,3]-B231[jcallsd9,5])+(A231[jcallsd9,4]-A231[jcallsd9,6])*(B231[jcallsd9,4]-B231[jcallsd9,6]))  - \
                  1.0*((A231[jcallsd9,4]-A231[jcallsd9,2])*(B231[jcallsd9,4]-B231[jcallsd9,6])+(A231[jcallsd9,3]-A231[jcallsd9,5])*(B231[jcallsd9,5]-B231[jcallsd9,7]))  + \
                  2.0*((A231[jcallsd9,3]-A231[jcallsd9,1])*(B231[jcallsd9,5]-B231[jcallsd9,7])+(A231[jcallsd9,2]-A231[jcallsd9,4])*(B231[jcallsd9,6]-B231[jcallsd9,8]))  + \
                  1.0*((A231[jcallsd9,2]-A231[jcallsd9,0])*(B231[jcallsd9,6]-B231[jcallsd9,8])+(A231[jcallsd9,1]-A231[jcallsd9,3])*(B231[jcallsd9,7]-B231[jcallsd9,9]))) \
                  /(161280.0*th.sqrt(th.tensor(2.0)))



    jcallsd10 = (jcallsd==10)

    if(jcallsd10.sum() != 0):
        S131[jcallsd10] = th.pow(zeta_b[jcallsd10,2],5.5)* \
                   th.pow(zeta_a[jcallsd10,0],5.5)* \
                   rij[jcallsd10]**11 * \
                  ((A131[jcallsd10,8]*(3.0*B131[jcallsd10,0]-B131[jcallsd10,2])+A131[jcallsd10,10]*(3.0*B131[jcallsd10,2]-B131[jcallsd10,0])-4.0*A131[jcallsd10,9]*B131[jcallsd10,1]) + \
                  2.0*(A131[jcallsd10,7]*(3.0*B131[jcallsd10,1]-B131[jcallsd10,3])+A131[jcallsd10,9]*(3.0*B131[jcallsd10,3]-B131[jcallsd10,1])-4.0*A131[jcallsd10,8]*B131[jcallsd10,2]) - \
                  2.0*(A131[jcallsd10,6]*(3.0*B131[jcallsd10,2]-B131[jcallsd10,4])+A131[jcallsd10,8]*(3.0*B131[jcallsd10,4]-B131[jcallsd10,2])-4.0*A131[jcallsd10,7]*B131[jcallsd10,3]) - \
                  6.0*(A131[jcallsd10,5]*(3.0*B131[jcallsd10,3]-B131[jcallsd10,5])+A131[jcallsd10,7]*(3.0*B131[jcallsd10,5]-B131[jcallsd10,3])-4.0*A131[jcallsd10,6]*B131[jcallsd10,4]) - \
                  6.0*(A131[jcallsd10,3]*(3.0*B131[jcallsd10,5]-B131[jcallsd10,7])+A131[jcallsd10,5]*(3.0*B131[jcallsd10,7]-B131[jcallsd10,5])-4.0*A131[jcallsd10,4]*B131[jcallsd10,6]) - \
                  2.0*(A131[jcallsd10,2]*(3.0*B131[jcallsd10,6]-B131[jcallsd10,8])+A131[jcallsd10,4]*(3.0*B131[jcallsd10,8]-B131[jcallsd10,6])-4.0*A131[jcallsd10,3]*B131[jcallsd10,7]) + \
                  2.0*(A131[jcallsd10,1]*(3.0*B131[jcallsd10,7]-B131[jcallsd10,9])+A131[jcallsd10,3]*(3.0*B131[jcallsd10,9]-B131[jcallsd10,7])-4.0*A131[jcallsd10,2]*B131[jcallsd10,8]) + \
                  (A131[jcallsd10,0]*(3.0*B131[jcallsd10,8]-B131[jcallsd10,10])+A131[jcallsd10,2]*(3.0*B131[jcallsd10,10]-B131[jcallsd10,8])-4.0*A131[jcallsd10,1]*B131[jcallsd10,9])) \
                  /(2903040.0*th.sqrt(th.tensor(5.0)))
        S231[jcallsd10] = th.pow(zeta_b[jcallsd10,2],5.5)* \
                   th.pow(zeta_a[jcallsd10,1],5.5)* \
                   rij[jcallsd10]**11 * \
                  ((A231[jcallsd10,7]*(3.0*B231[jcallsd10,0]-B231[jcallsd10,2])-A231[jcallsd10,8]*(B231[jcallsd10,1]+B231[jcallsd10,3])-A231[jcallsd10,9]*(B231[jcallsd10,0]+B231[jcallsd10,2])+A231[jcallsd10,10]*(3.0*B231[jcallsd10,3]-B231[jcallsd10,1])) + \
                  1.0*(A231[jcallsd10,6]*(3.0*B231[jcallsd10,1]-B231[jcallsd10,3])-A231[jcallsd10,7]*(B231[jcallsd10,2]+B231[jcallsd10,4])-A231[jcallsd10,8]*(B231[jcallsd10,1]+B231[jcallsd10,3])+A231[jcallsd10,9]*(3.0*B231[jcallsd10,4]-B231[jcallsd10,2]))  - \
                  3.0*(A231[jcallsd10,5]*(3.0*B231[jcallsd10,2]-B231[jcallsd10,4])-A231[jcallsd10,6]*(B231[jcallsd10,3]+B231[jcallsd10,5])-A231[jcallsd10,7]*(B231[jcallsd10,2]+B231[jcallsd10,4])+A231[jcallsd10,8]*(3.0*B231[jcallsd10,5]-B231[jcallsd10,3]))  - \
                  3.0*(A231[jcallsd10,4]*(3.0*B231[jcallsd10,3]-B231[jcallsd10,5])-A231[jcallsd10,5]*(B231[jcallsd10,4]+B231[jcallsd10,6])-A231[jcallsd10,6]*(B231[jcallsd10,3]+B231[jcallsd10,5])+A231[jcallsd10,7]*(3.0*B231[jcallsd10,6]-B231[jcallsd10,4]))  + \
                  3.0*(A231[jcallsd10,3]*(3.0*B231[jcallsd10,4]-B231[jcallsd10,6])-A231[jcallsd10,4]*(B231[jcallsd10,5]+B231[jcallsd10,7])-A231[jcallsd10,5]*(B231[jcallsd10,4]+B231[jcallsd10,6])+A231[jcallsd10,6]*(3.0*B231[jcallsd10,7]-B231[jcallsd10,5]))  + \
                  3.0*(A231[jcallsd10,2]*(3.0*B231[jcallsd10,5]-B231[jcallsd10,7])-A231[jcallsd10,3]*(B231[jcallsd10,6]+B231[jcallsd10,8])-A231[jcallsd10,4]*(B231[jcallsd10,5]+B231[jcallsd10,7])+A231[jcallsd10,5]*(3.0*B231[jcallsd10,8]-B231[jcallsd10,6]))  - \
                  1.0*(A231[jcallsd10,1]*(3.0*B231[jcallsd10,6]-B231[jcallsd10,8])-A231[jcallsd10,2]*(B231[jcallsd10,7]+B231[jcallsd10,9])-A231[jcallsd10,3]*(B231[jcallsd10,6]+B231[jcallsd10,8])+A231[jcallsd10,4]*(3.0*B231[jcallsd10,9]-B231[jcallsd10,7]))  - \
                  1.0*(A231[jcallsd10,0]*(3.0*B231[jcallsd10,7]-B231[jcallsd10,9])-A231[jcallsd10,1]*(B231[jcallsd10,8]+B231[jcallsd10,10])-A231[jcallsd10,2]*(B231[jcallsd10,7]+B231[jcallsd10,9])+A231[jcallsd10,3]*(3.0*B231[jcallsd10,10]-B231[jcallsd10,8]))) \
                  /(967680.0*th.sqrt(th.tensor(15.0)))
        S232[jcallsd10] = th.pow(zeta_b[jcallsd10,2],5.5)* \
                   th.pow(zeta_a[jcallsd10,1],5.5)* \
                   rij[jcallsd10]**11 * \
                  (((A231[jcallsd10,9]-A231[jcallsd10,7])*(B231[jcallsd10,0]-B231[jcallsd10,2])+(A231[jcallsd10,8]-A231[jcallsd10,10])*(B231[jcallsd10,1]-B231[jcallsd10,3])) + \
                  1.0*((A231[jcallsd10,8]-A231[jcallsd10,6])*(B231[jcallsd10,1]-B231[jcallsd10,3])+(A231[jcallsd10,7]-A231[jcallsd10,9])*(B231[jcallsd10,2]-B231[jcallsd10,4]))  - \
                  3.0*((A231[jcallsd10,7]-A231[jcallsd10,5])*(B231[jcallsd10,2]-B231[jcallsd10,4])+(A231[jcallsd10,6]-A231[jcallsd10,8])*(B231[jcallsd10,3]-B231[jcallsd10,5]))  - \
                  3.0*((A231[jcallsd10,6]-A231[jcallsd10,4])*(B231[jcallsd10,3]-B231[jcallsd10,5])+(A231[jcallsd10,5]-A231[jcallsd10,7])*(B231[jcallsd10,4]-B231[jcallsd10,6]))  + \
                  3.0*((A231[jcallsd10,5]-A231[jcallsd10,3])*(B231[jcallsd10,4]-B231[jcallsd10,6])+(A231[jcallsd10,4]-A231[jcallsd10,6])*(B231[jcallsd10,5]-B231[jcallsd10,7]))  + \
                  3.0*((A231[jcallsd10,4]-A231[jcallsd10,2])*(B231[jcallsd10,5]-B231[jcallsd10,7])+(A231[jcallsd10,3]-A231[jcallsd10,5])*(B231[jcallsd10,6]-B231[jcallsd10,8]))  - \
                  1.0*((A231[jcallsd10,3]-A231[jcallsd10,1])*(B231[jcallsd10,6]-B231[jcallsd10,8])+(A231[jcallsd10,2]-A231[jcallsd10,4])*(B231[jcallsd10,7]-B231[jcallsd10,9]))  - \
                  1.0*((A231[jcallsd10,2]-A231[jcallsd10,0])*(B231[jcallsd10,7]-B231[jcallsd10,9])+(A231[jcallsd10,1]-A231[jcallsd10,3])*(B231[jcallsd10,8]-B231[jcallsd10,10]))) \
                  /(967680.0*th.sqrt(th.tensor(5.0)))




    jcallsd963 = (jcallsd==963)
    if(jcallsd963.sum() != 0):
        S131[jcallsd963] = th.pow(zeta_b[jcallsd963,2],3.5)* \
                   th.pow(zeta_a[jcallsd963,0],6.5)* \
                   rij[jcallsd963]**10 * \
                  ((A131[jcallsd963,7]*(3.0*B131[jcallsd963,0]-B131[jcallsd963,2])+A131[jcallsd963,9]*(3.0*B131[jcallsd963,2]-B131[jcallsd963,0])-4.0*A131[jcallsd963,8]*B131[jcallsd963,1]) + \
                  5.0*(A131[jcallsd963,6]*(3.0*B131[jcallsd963,1]-B131[jcallsd963,3])+A131[jcallsd963,8]*(3.0*B131[jcallsd963,3]-B131[jcallsd963,1])-4.0*A131[jcallsd963,7]*B131[jcallsd963,2]) + \
                  9.0*(A131[jcallsd963,5]*(3.0*B131[jcallsd963,2]-B131[jcallsd963,4])+A131[jcallsd963,7]*(3.0*B131[jcallsd963,4]-B131[jcallsd963,2])-4.0*A131[jcallsd963,6]*B131[jcallsd963,3]) + \
                  5.0*(A131[jcallsd963,4]*(3.0*B131[jcallsd963,3]-B131[jcallsd963,5])+A131[jcallsd963,6]*(3.0*B131[jcallsd963,5]-B131[jcallsd963,3])-4.0*A131[jcallsd963,5]*B131[jcallsd963,4]) - \
                  5.0*(A131[jcallsd963,3]*(3.0*B131[jcallsd963,4]-B131[jcallsd963,6])+A131[jcallsd963,5]*(3.0*B131[jcallsd963,6]-B131[jcallsd963,4])-4.0*A131[jcallsd963,4]*B131[jcallsd963,5]) - \
                  9.0*(A131[jcallsd963,2]*(3.0*B131[jcallsd963,5]-B131[jcallsd963,7])+A131[jcallsd963,4]*(3.0*B131[jcallsd963,7]-B131[jcallsd963,5])-4.0*A131[jcallsd963,3]*B131[jcallsd963,6]) - \
                  5.0*(A131[jcallsd963,1]*(3.0*B131[jcallsd963,6]-B131[jcallsd963,8])+A131[jcallsd963,3]*(3.0*B131[jcallsd963,8]-B131[jcallsd963,6])-4.0*A131[jcallsd963,2]*B131[jcallsd963,7]) - \
                  (A131[jcallsd963,0]*(3.0*B131[jcallsd963,7]-B131[jcallsd963,9])+A131[jcallsd963,2]*(3.0*B131[jcallsd963,9]-B131[jcallsd963,7])-4.0*A131[jcallsd963,1]*B131[jcallsd963,8])) \
                  /(69120.0*th.sqrt(th.tensor(231.0)))
        S231[jcallsd963] = th.pow(zeta_b[jcallsd963,2],3.5)* \
                   th.pow(zeta_a[jcallsd963,1],6.5)* \
                   rij[jcallsd963]**10 * \
                  ((A231[jcallsd963,6]*(3.0*B231[jcallsd963,0]-B231[jcallsd963,2])-A231[jcallsd963,7]*(B231[jcallsd963,1]+B231[jcallsd963,3])-A231[jcallsd963,8]*(B231[jcallsd963,0]+B231[jcallsd963,2])+A231[jcallsd963,9]*(3.0*B231[jcallsd963,3]-B231[jcallsd963,1])) + \
                  4.0*(A231[jcallsd963,5]*(3.0*B231[jcallsd963,1]-B231[jcallsd963,3])-A231[jcallsd963,6]*(B231[jcallsd963,2]+B231[jcallsd963,4])-A231[jcallsd963,7]*(B231[jcallsd963,1]+B231[jcallsd963,3])+A231[jcallsd963,8]*(3.0*B231[jcallsd963,4]-B231[jcallsd963,2]))  + \
                  5.0*(A231[jcallsd963,4]*(3.0*B231[jcallsd963,2]-B231[jcallsd963,4])-A231[jcallsd963,5]*(B231[jcallsd963,3]+B231[jcallsd963,5])-A231[jcallsd963,6]*(B231[jcallsd963,2]+B231[jcallsd963,4])+A231[jcallsd963,7]*(3.0*B231[jcallsd963,5]-B231[jcallsd963,3]))  + \
                  5.0*(A231[jcallsd963,2]*(3.0*B231[jcallsd963,4]-B231[jcallsd963,6])-A231[jcallsd963,3]*(B231[jcallsd963,5]+B231[jcallsd963,7])-A231[jcallsd963,4]*(B231[jcallsd963,4]+B231[jcallsd963,6])+A231[jcallsd963,5]*(3.0*B231[jcallsd963,7]-B231[jcallsd963,5]))  - \
                  4.0*(A231[jcallsd963,1]*(3.0*B231[jcallsd963,5]-B231[jcallsd963,7])-A231[jcallsd963,2]*(B231[jcallsd963,6]+B231[jcallsd963,8])-A231[jcallsd963,3]*(B231[jcallsd963,5]+B231[jcallsd963,7])+A231[jcallsd963,4]*(3.0*B231[jcallsd963,8]-B231[jcallsd963,6]))  - \
                  1.0*(A231[jcallsd963,0]*(3.0*B231[jcallsd963,6]-B231[jcallsd963,8])-A231[jcallsd963,1]*(B231[jcallsd963,7]+B231[jcallsd963,9])-A231[jcallsd963,2]*(B231[jcallsd963,6]+B231[jcallsd963,8])+A231[jcallsd963,3]*(3.0*B231[jcallsd963,9]-B231[jcallsd963,7]))) \
                  /(69120.0*th.sqrt(th.tensor(77.0)))
        S232[jcallsd963] = th.pow(zeta_b[jcallsd963,2],3.5)* \
                   th.pow(zeta_a[jcallsd963,1],6.5)* \
                   rij[jcallsd963]**10 * \
                  (((A231[jcallsd963,8]-A231[jcallsd963,6])*(B231[jcallsd963,0]-B231[jcallsd963,2])+(A231[jcallsd963,7]-A231[jcallsd963,9])*(B231[jcallsd963,1]-B231[jcallsd963,3])) + \
                  4.0*((A231[jcallsd963,7]-A231[jcallsd963,5])*(B231[jcallsd963,1]-B231[jcallsd963,3])+(A231[jcallsd963,6]-A231[jcallsd963,8])*(B231[jcallsd963,2]-B231[jcallsd963,4]))  + \
                  5.0*((A231[jcallsd963,6]-A231[jcallsd963,4])*(B231[jcallsd963,2]-B231[jcallsd963,4])+(A231[jcallsd963,5]-A231[jcallsd963,7])*(B231[jcallsd963,3]-B231[jcallsd963,5]))  + \
                  5.0*((A231[jcallsd963,4]-A231[jcallsd963,2])*(B231[jcallsd963,4]-B231[jcallsd963,6])+(A231[jcallsd963,3]-A231[jcallsd963,5])*(B231[jcallsd963,5]-B231[jcallsd963,7]))  - \
                  4.0*((A231[jcallsd963,3]-A231[jcallsd963,1])*(B231[jcallsd963,5]-B231[jcallsd963,7])+(A231[jcallsd963,2]-A231[jcallsd963,4])*(B231[jcallsd963,6]-B231[jcallsd963,8]))  - \
                  1.0*((A231[jcallsd963,2]-A231[jcallsd963,0])*(B231[jcallsd963,6]-B231[jcallsd963,8])+(A231[jcallsd963,1]-A231[jcallsd963,3])*(B231[jcallsd963,7]-B231[jcallsd963,9]))) \
                  /(23040.0*th.sqrt(th.tensor(231.0)))



    jcallsd1064 = (jcallsd==1064)
    if(jcallsd1064.sum() != 0):
        S131[jcallsd1064] = th.pow(zeta_b[jcallsd1064,2],4.5)* \
                   th.pow(zeta_a[jcallsd1064,0],6.5)* \
                   rij[jcallsd1064]**11 * \
                  ((A131[jcallsd1064,8]*(3.0*B131[jcallsd1064,0]-B131[jcallsd1064,2])+A131[jcallsd1064,10]*(3.0*B131[jcallsd1064,2]-B131[jcallsd1064,0])-4.0*A131[jcallsd1064,9]*B131[jcallsd1064,1]) + \
                  4.0*(A131[jcallsd1064,7]*(3.0*B131[jcallsd1064,1]-B131[jcallsd1064,3])+A131[jcallsd1064,9]*(3.0*B131[jcallsd1064,3]-B131[jcallsd1064,1])-4.0*A131[jcallsd1064,8]*B131[jcallsd1064,2]) + \
                  4.0*(A131[jcallsd1064,6]*(3.0*B131[jcallsd1064,2]-B131[jcallsd1064,4])+A131[jcallsd1064,8]*(3.0*B131[jcallsd1064,4]-B131[jcallsd1064,2])-4.0*A131[jcallsd1064,7]*B131[jcallsd1064,3]) - \
                  4.0*(A131[jcallsd1064,5]*(3.0*B131[jcallsd1064,3]-B131[jcallsd1064,5])+A131[jcallsd1064,7]*(3.0*B131[jcallsd1064,5]-B131[jcallsd1064,3])-4.0*A131[jcallsd1064,6]*B131[jcallsd1064,4]) - \
                  10.0*(A131[jcallsd1064,4]*(3.0*B131[jcallsd1064,4]-B131[jcallsd1064,6])+A131[jcallsd1064,6]*(3.0*B131[jcallsd1064,6]-B131[jcallsd1064,4])-4.0*A131[jcallsd1064,5]*B131[jcallsd1064,5]) - \
                  4.0*(A131[jcallsd1064,3]*(3.0*B131[jcallsd1064,5]-B131[jcallsd1064,7])+A131[jcallsd1064,5]*(3.0*B131[jcallsd1064,7]-B131[jcallsd1064,5])-4.0*A131[jcallsd1064,4]*B131[jcallsd1064,6]) + \
                  4.0*(A131[jcallsd1064,2]*(3.0*B131[jcallsd1064,6]-B131[jcallsd1064,8])+A131[jcallsd1064,4]*(3.0*B131[jcallsd1064,8]-B131[jcallsd1064,6])-4.0*A131[jcallsd1064,3]*B131[jcallsd1064,7]) + \
                  4.0*(A131[jcallsd1064,1]*(3.0*B131[jcallsd1064,7]-B131[jcallsd1064,9])+A131[jcallsd1064,3]*(3.0*B131[jcallsd1064,9]-B131[jcallsd1064,7])-4.0*A131[jcallsd1064,2]*B131[jcallsd1064,8]) + \
                  (A131[jcallsd1064,0]*(3.0*B131[jcallsd1064,8]-B131[jcallsd1064,10])+A131[jcallsd1064,2]*(3.0*B131[jcallsd1064,10]-B131[jcallsd1064,8])-4.0*A131[jcallsd1064,1]*B131[jcallsd1064,9])) \
                  /(967680.0*th.sqrt(th.tensor(66.0)))
        S231[jcallsd1064] = th.pow(zeta_b[jcallsd1064,2],4.5)* \
                   th.pow(zeta_a[jcallsd1064,1],6.5)* \
                   rij[jcallsd1064]**11 * \
                  ((A231[jcallsd1064,7]*(3.0*B231[jcallsd1064,0]-B231[jcallsd1064,2])-A231[jcallsd1064,8]*(B231[jcallsd1064,1]+B231[jcallsd1064,3])-A231[jcallsd1064,9]*(B231[jcallsd1064,0]+B231[jcallsd1064,2])+A231[jcallsd1064,10]*(3.0*B231[jcallsd1064,3]-B231[jcallsd1064,1])) + \
                  3.0*(A231[jcallsd1064,6]*(3.0*B231[jcallsd1064,1]-B231[jcallsd1064,3])-A231[jcallsd1064,7]*(B231[jcallsd1064,2]+B231[jcallsd1064,4])-A231[jcallsd1064,8]*(B231[jcallsd1064,1]+B231[jcallsd1064,3])+A231[jcallsd1064,9]*(3.0*B231[jcallsd1064,4]-B231[jcallsd1064,2]))  + \
                  1.0*(A231[jcallsd1064,5]*(3.0*B231[jcallsd1064,2]-B231[jcallsd1064,4])-A231[jcallsd1064,6]*(B231[jcallsd1064,3]+B231[jcallsd1064,5])-A231[jcallsd1064,7]*(B231[jcallsd1064,2]+B231[jcallsd1064,4])+A231[jcallsd1064,8]*(3.0*B231[jcallsd1064,5]-B231[jcallsd1064,3]))  - \
                  5.0*(A231[jcallsd1064,4]*(3.0*B231[jcallsd1064,3]-B231[jcallsd1064,5])-A231[jcallsd1064,5]*(B231[jcallsd1064,4]+B231[jcallsd1064,6])-A231[jcallsd1064,6]*(B231[jcallsd1064,3]+B231[jcallsd1064,5])+A231[jcallsd1064,7]*(3.0*B231[jcallsd1064,6]-B231[jcallsd1064,4]))  - \
                  5.0*(A231[jcallsd1064,3]*(3.0*B231[jcallsd1064,4]-B231[jcallsd1064,6])-A231[jcallsd1064,4]*(B231[jcallsd1064,5]+B231[jcallsd1064,7])-A231[jcallsd1064,5]*(B231[jcallsd1064,4]+B231[jcallsd1064,6])+A231[jcallsd1064,6]*(3.0*B231[jcallsd1064,7]-B231[jcallsd1064,5]))  + \
                  1.0*(A231[jcallsd1064,2]*(3.0*B231[jcallsd1064,5]-B231[jcallsd1064,7])-A231[jcallsd1064,3]*(B231[jcallsd1064,6]+B231[jcallsd1064,8])-A231[jcallsd1064,4]*(B231[jcallsd1064,5]+B231[jcallsd1064,7])+A231[jcallsd1064,5]*(3.0*B231[jcallsd1064,8]-B231[jcallsd1064,6]))  + \
                  3.0*(A231[jcallsd1064,1]*(3.0*B231[jcallsd1064,6]-B231[jcallsd1064,8])-A231[jcallsd1064,2]*(B231[jcallsd1064,7]+B231[jcallsd1064,9])-A231[jcallsd1064,3]*(B231[jcallsd1064,6]+B231[jcallsd1064,8])+A231[jcallsd1064,4]*(3.0*B231[jcallsd1064,9]-B231[jcallsd1064,7]))  + \
                  1.0*(A231[jcallsd1064,0]*(3.0*B231[jcallsd1064,7]-B231[jcallsd1064,9])-A231[jcallsd1064,1]*(B231[jcallsd1064,8]+B231[jcallsd1064,10])-A231[jcallsd1064,2]*(B231[jcallsd1064,7]+B231[jcallsd1064,9])+A231[jcallsd1064,3]*(3.0*B231[jcallsd1064,10]-B231[jcallsd1064,8]))) \
                  /(967680.0*th.sqrt(th.tensor(22.0)))
        S232[jcallsd1064] = th.pow(zeta_b[jcallsd1064,2],4.5)* \
                   th.pow(zeta_a[jcallsd1064,1],6.5)* \
                   rij[jcallsd1064]**11 * \
                  (((A231[jcallsd1064,9]-A231[jcallsd1064,7])*(B231[jcallsd1064,0]-B231[jcallsd1064,2])+(A231[jcallsd1064,8]-A231[jcallsd1064,10])*(B231[jcallsd1064,1]-B231[jcallsd1064,3])) + \
                  3.0*((A231[jcallsd1064,8]-A231[jcallsd1064,6])*(B231[jcallsd1064,1]-B231[jcallsd1064,3])+(A231[jcallsd1064,7]-A231[jcallsd1064,9])*(B231[jcallsd1064,2]-B231[jcallsd1064,4]))  + \
                  1.0*((A231[jcallsd1064,7]-A231[jcallsd1064,5])*(B231[jcallsd1064,2]-B231[jcallsd1064,4])+(A231[jcallsd1064,6]-A231[jcallsd1064,8])*(B231[jcallsd1064,3]-B231[jcallsd1064,5]))  - \
                  5.0*((A231[jcallsd1064,6]-A231[jcallsd1064,4])*(B231[jcallsd1064,3]-B231[jcallsd1064,5])+(A231[jcallsd1064,5]-A231[jcallsd1064,7])*(B231[jcallsd1064,4]-B231[jcallsd1064,6]))  - \
                  5.0*((A231[jcallsd1064,5]-A231[jcallsd1064,3])*(B231[jcallsd1064,4]-B231[jcallsd1064,6])+(A231[jcallsd1064,4]-A231[jcallsd1064,6])*(B231[jcallsd1064,5]-B231[jcallsd1064,7]))  + \
                  1.0*((A231[jcallsd1064,4]-A231[jcallsd1064,2])*(B231[jcallsd1064,5]-B231[jcallsd1064,7])+(A231[jcallsd1064,3]-A231[jcallsd1064,5])*(B231[jcallsd1064,6]-B231[jcallsd1064,8]))  + \
                  3.0*((A231[jcallsd1064,3]-A231[jcallsd1064,1])*(B231[jcallsd1064,6]-B231[jcallsd1064,8])+(A231[jcallsd1064,2]-A231[jcallsd1064,4])*(B231[jcallsd1064,7]-B231[jcallsd1064,9]))  + \
                  1.0*((A231[jcallsd1064,2]-A231[jcallsd1064,0])*(B231[jcallsd1064,7]-B231[jcallsd1064,9])+(A231[jcallsd1064,1]-A231[jcallsd1064,3])*(B231[jcallsd1064,8]-B231[jcallsd1064,10]))) \
                  /(322560.0*th.sqrt(th.tensor(66.0)))




    jcallsd11 = (jcallsd==11)
    if(jcallsd11.sum() != 0):
        S131[jcallsd11] = th.pow(zeta_b[jcallsd11,2],5.5)* \
                   th.pow(zeta_a[jcallsd11,0],6.5)* \
                   rij[jcallsd11]**12 * \
                  ((A131[jcallsd11,9]*(3.0*B131[jcallsd11,0]-B131[jcallsd11,2])+A131[jcallsd11,11]*(3.0*B131[jcallsd11,2]-B131[jcallsd11,0])-4.0*A131[jcallsd11,10]*B131[jcallsd11,1]) + \
                  3.0*(A131[jcallsd11,8]*(3.0*B131[jcallsd11,1]-B131[jcallsd11,3])+A131[jcallsd11,10]*(3.0*B131[jcallsd11,3]-B131[jcallsd11,1])-4.0*A131[jcallsd11,9]*B131[jcallsd11,2]) - \
                  8.0*(A131[jcallsd11,6]*(3.0*B131[jcallsd11,3]-B131[jcallsd11,5])+A131[jcallsd11,8]*(3.0*B131[jcallsd11,5]-B131[jcallsd11,3])-4.0*A131[jcallsd11,7]*B131[jcallsd11,4]) - \
                  6.0*(A131[jcallsd11,5]*(3.0*B131[jcallsd11,4]-B131[jcallsd11,6])+A131[jcallsd11,7]*(3.0*B131[jcallsd11,6]-B131[jcallsd11,4])-4.0*A131[jcallsd11,6]*B131[jcallsd11,5]) + \
                  6.0*(A131[jcallsd11,4]*(3.0*B131[jcallsd11,5]-B131[jcallsd11,7])+A131[jcallsd11,6]*(3.0*B131[jcallsd11,7]-B131[jcallsd11,5])-4.0*A131[jcallsd11,5]*B131[jcallsd11,6]) + \
                  8.0*(A131[jcallsd11,3]*(3.0*B131[jcallsd11,6]-B131[jcallsd11,8])+A131[jcallsd11,5]*(3.0*B131[jcallsd11,8]-B131[jcallsd11,6])-4.0*A131[jcallsd11,4]*B131[jcallsd11,7]) - \
                  3.0*(A131[jcallsd11,1]*(3.0*B131[jcallsd11,8]-B131[jcallsd11,10])+A131[jcallsd11,3]*(3.0*B131[jcallsd11,10]-B131[jcallsd11,8])-4.0*A131[jcallsd11,2]*B131[jcallsd11,9]) - \
                  (A131[jcallsd11,0]*(3.0*B131[jcallsd11,9]-B131[jcallsd11,11])+A131[jcallsd11,2]*(3.0*B131[jcallsd11,11]-B131[jcallsd11,9])-4.0*A131[jcallsd11,1]*B131[jcallsd11,10])) \
                  /(5806080.0*th.sqrt(th.tensor(165.0)))
        S231[jcallsd11] = th.pow(zeta_b[jcallsd11,2],5.5)* \
                   th.pow(zeta_a[jcallsd11,1],6.5)* \
                   rij[jcallsd11]**12 * \
                  ((A231[jcallsd11,8]*(3.0*B231[jcallsd11,0]-B231[jcallsd11,2])-A231[jcallsd11,9]*(B231[jcallsd11,1]+B231[jcallsd11,3])-A231[jcallsd11,10]*(B231[jcallsd11,0]+B231[jcallsd11,2])+A231[jcallsd11,11]*(3.0*B231[jcallsd11,3]-B231[jcallsd11,1])) + \
                  2.0*(A231[jcallsd11,7]*(3.0*B231[jcallsd11,1]-B231[jcallsd11,3])-A231[jcallsd11,8]*(B231[jcallsd11,2]+B231[jcallsd11,4])-A231[jcallsd11,9]*(B231[jcallsd11,1]+B231[jcallsd11,3])+A231[jcallsd11,10]*(3.0*B231[jcallsd11,4]-B231[jcallsd11,2]))  - \
                  2.0*(A231[jcallsd11,6]*(3.0*B231[jcallsd11,2]-B231[jcallsd11,4])-A231[jcallsd11,7]*(B231[jcallsd11,3]+B231[jcallsd11,5])-A231[jcallsd11,8]*(B231[jcallsd11,2]+B231[jcallsd11,4])+A231[jcallsd11,9]*(3.0*B231[jcallsd11,5]-B231[jcallsd11,3]))  - \
                  6.0*(A231[jcallsd11,5]*(3.0*B231[jcallsd11,3]-B231[jcallsd11,5])-A231[jcallsd11,6]*(B231[jcallsd11,4]+B231[jcallsd11,6])-A231[jcallsd11,7]*(B231[jcallsd11,3]+B231[jcallsd11,5])+A231[jcallsd11,8]*(3.0*B231[jcallsd11,6]-B231[jcallsd11,4]))  + \
                  6.0*(A231[jcallsd11,3]*(3.0*B231[jcallsd11,5]-B231[jcallsd11,7])-A231[jcallsd11,4]*(B231[jcallsd11,6]+B231[jcallsd11,8])-A231[jcallsd11,5]*(B231[jcallsd11,5]+B231[jcallsd11,7])+A231[jcallsd11,6]*(3.0*B231[jcallsd11,8]-B231[jcallsd11,6]))  + \
                  2.0*(A231[jcallsd11,2]*(3.0*B231[jcallsd11,6]-B231[jcallsd11,8])-A231[jcallsd11,3]*(B231[jcallsd11,7]+B231[jcallsd11,9])-A231[jcallsd11,4]*(B231[jcallsd11,6]+B231[jcallsd11,8])+A231[jcallsd11,5]*(3.0*B231[jcallsd11,9]-B231[jcallsd11,7]))  - \
                  2.0*(A231[jcallsd11,1]*(3.0*B231[jcallsd11,7]-B231[jcallsd11,9])-A231[jcallsd11,2]*(B231[jcallsd11,8]+B231[jcallsd11,10])-A231[jcallsd11,3]*(B231[jcallsd11,7]+B231[jcallsd11,9])+A231[jcallsd11,4]*(3.0*B231[jcallsd11,10]-B231[jcallsd11,8]))  - \
                  1.0*(A231[jcallsd11,0]*(3.0*B231[jcallsd11,8]-B231[jcallsd11,10])-A231[jcallsd11,1]*(B231[jcallsd11,9]+B231[jcallsd11,11])-A231[jcallsd11,2]*(B231[jcallsd11,8]+B231[jcallsd11,10])+A231[jcallsd11,3]*(3.0*B231[jcallsd11,11]-B231[jcallsd11,9]))) \
                  /(5806080.0*th.sqrt(th.tensor(55.0)))
        S232[jcallsd11] = th.pow(zeta_b[jcallsd11,2],5.5)* \
                   th.pow(zeta_a[jcallsd11,1],6.5)* \
                   rij[jcallsd11]**12 * \
                  (((A231[jcallsd11,10]-A231[jcallsd11,8])*(B231[jcallsd11,0]-B231[jcallsd11,2])+(A231[jcallsd11,9]-A231[jcallsd11,11])*(B231[jcallsd11,1]-B231[jcallsd11,3])) + \
                  2.0*((A231[jcallsd11,9]-A231[jcallsd11,7])*(B231[jcallsd11,1]-B231[jcallsd11,3])+(A231[jcallsd11,8]-A231[jcallsd11,10])*(B231[jcallsd11,2]-B231[jcallsd11,4]))  - \
                  2.0*((A231[jcallsd11,8]-A231[jcallsd11,6])*(B231[jcallsd11,2]-B231[jcallsd11,4])+(A231[jcallsd11,7]-A231[jcallsd11,9])*(B231[jcallsd11,3]-B231[jcallsd11,5]))  - \
                  6.0*((A231[jcallsd11,7]-A231[jcallsd11,5])*(B231[jcallsd11,3]-B231[jcallsd11,5])+(A231[jcallsd11,6]-A231[jcallsd11,8])*(B231[jcallsd11,4]-B231[jcallsd11,6]))  + \
                  6.0*((A231[jcallsd11,5]-A231[jcallsd11,3])*(B231[jcallsd11,5]-B231[jcallsd11,7])+(A231[jcallsd11,4]-A231[jcallsd11,6])*(B231[jcallsd11,6]-B231[jcallsd11,8]))  + \
                  2.0*((A231[jcallsd11,4]-A231[jcallsd11,2])*(B231[jcallsd11,6]-B231[jcallsd11,8])+(A231[jcallsd11,3]-A231[jcallsd11,5])*(B231[jcallsd11,7]-B231[jcallsd11,9]))  - \
                  2.0*((A231[jcallsd11,3]-A231[jcallsd11,1])*(B231[jcallsd11,7]-B231[jcallsd11,9])+(A231[jcallsd11,2]-A231[jcallsd11,4])*(B231[jcallsd11,8]-B231[jcallsd11,10]))  - \
                  1.0*((A231[jcallsd11,2]-A231[jcallsd11,0])*(B231[jcallsd11,8]-B231[jcallsd11,10])+(A231[jcallsd11,1]-A231[jcallsd11,3])*(B231[jcallsd11,9]-B231[jcallsd11,11]))) \
                  /(1935360.0*th.sqrt(th.tensor(165.0)))






############# Here we do Dgamma,Dgamma overlap #############
    S333 = th.zeros_like(S111)
    S332 = th.zeros_like(S111)
    S331 = th.zeros_like(S111)

    A33, B33 = SET(rij, jcallsd, zeta_a[...,2],zeta_b[...,2])

    jcalldd6 = (jcalldd==6)
    if(jcalldd6.sum() != 0):
        A0 = A33[jcalldd6,0]
        A2 = A33[jcalldd6,2]
        A4 = A33[jcalldd6,4]
        A6 = A33[jcalldd6,6]

        B0 = B33[jcalldd6,0]
        B2 = B33[jcalldd6,2]
        B4 = B33[jcalldd6,4]
        B6 = B33[jcalldd6,6]

        w = th.pow(zeta_b[jcalldd6,2],3.5)* \
                   th.pow(zeta_a[jcalldd6,2],3.5)* \
                   rij[jcalldd6]**7

        S333[jcalldd6] = w * \
                  (((A2-2.0*A4+A6)*(B0-2.0*B2+B4)) - \
                  ((A0-2.0*A2+A4)*(B2-2.0*B4+B6)))  \
                  /(768.0)
        S332[jcalldd6] = -w * \
                  ((A2*(B2-B0)+A4*(B0-B4)-A6*(B2-B4)) - \
                  (A0*(B4-B2)+A2*(B2-B6)-A4*(B4-B6)))  \
                  /(192.0)
        S331[jcalldd6] = w * \
                  ((A2*(9.0*B0-6.0*B2+B4)-2.0*A4*(3.0*B0-2.0*B2+3.0*B4)+A6*(B0-6.0*B2+9.0*B4)) - \
                  (A0*(9.0*B2-6.0*B4+B6)-2.0*A2*(3.0*B2-2.0*B4+3.0*B6)+A4*(B2-6.0*B4+9.0*B6)))  \
                  /(1152.0)



    jcalldd7 = (jcalldd==7)

    if(jcalldd7.sum() != 0):
        A0 = A33[jcalldd7,0]
        A1 = A33[jcalldd7,1]
        A2 = A33[jcalldd7,2]
        A3 = A33[jcalldd7,3]
        A4 = A33[jcalldd7,4]
        A5 = A33[jcalldd7,5]
        A6 = A33[jcalldd7,6]
        A7 = A33[jcalldd7,7]

        B0 = B33[jcalldd7,0]
        B1 = B33[jcalldd7,1]
        B2 = B33[jcalldd7,2]
        B3 = B33[jcalldd7,3]
        B4 = B33[jcalldd7,4]
        B5 = B33[jcalldd7,5]
        B6 = B33[jcalldd7,6]
        B7 = B33[jcalldd7,7]

        w = th.pow(zeta_b[jcalldd7,2],3.5)* \
                   th.pow(zeta_a[jcalldd7,2],4.5)* \
                   rij[jcalldd7]**8


        S333[jcalldd7] = w * \
                  (((A3-2.0*A5+A7)*(B0-2.0*B2+B4)) + \
                  ((A2-2.0*A4+A6)*(B1-2.0*B3+B5)) - \
                  ((A1-2.0*A3+A5)*(B2-2.0*B4+B6)) - \
                  ((A0-2.0*A2+A4)*(B3-2.0*B5+B7)))  \
                  /(1536.0*th.sqrt(th.tensor(14.0)))
        S332[jcalldd7] = -w * \
                  ((A3*(B2-B0)+A5*(B0-B4)-A7*(B2-B4)) + \
                  (A2*(B3-B1)+A4*(B1-B5)-A6*(B3-B5)) - \
                  (A1*(B4-B2)+A3*(B2-B6)-A5*(B4-B6)) - \
                  (A0*(B5-B3)+A2*(B3-B7)-A4*(B5-B7)))  \
                  /(384.0*th.sqrt(th.tensor(14.0)))
        S331[jcalldd7] = w * \
                  ((A3*(9.0*B0-6.0*B2+B4)-2.0*A5*(3.0*B0-2.0*B2+3.0*B4)+A7*(B0-6.0*B2+9.0*B4)) + \
                  (A2*(9.0*B1-6.0*B3+B5)-2.0*A4*(3.0*B1-2.0*B3+3.0*B5)+A6*(B1-6.0*B3+9.0*B5)) - \
                  (A1*(9.0*B2-6.0*B4+B6)-2.0*A3*(3.0*B2-2.0*B4+3.0*B6)+A5*(B2-6.0*B4+9.0*B6)) - \
                  (A0*(9.0*B3-6.0*B5+B7)-2.0*A2*(3.0*B3-2.0*B5+3.0*B7)+A4*(B3-6.0*B5+9.0*B7)))  \
                  /(2304.0*th.sqrt(th.tensor(14.0)))



    jcalldd8 = (jcalldd==8)

    if(jcalldd8.sum() != 0):
        A0 = A33[jcalldd8,0]
        A2 = A33[jcalldd8,2]
        A4 = A33[jcalldd8,4]
        A6 = A33[jcalldd8,6]
        A8 = A33[jcalldd8,8]

        B0 = B33[jcalldd8,0]
        B2 = B33[jcalldd8,2]
        B4 = B33[jcalldd8,4]
        B6 = B33[jcalldd8,6]
        B8 = B33[jcalldd8,8]

        w = th.pow(zeta_b[jcalldd8,2],4.5)* \
                   th.pow(zeta_a[jcalldd8,2],4.5)* \
                   rij[jcalldd8]**9

        S333[jcalldd8] = w * \
                  (((A4-2.0*A6+A8)*(B0-2.0*B2+B4)) - \
                  2.0*((A2-2.0*A4+A6)*(B2-2.0*B4+B6)) + \
                  ((A0-2.0*A2+A4)*(B4-2.0*B6+B8)))  \
                  /(43008.0)
        S332[jcalldd8] = -w * \
                  ((A4*(B2-B0)+A6*(B0-B4)-A8*(B2-B4)) - \
                  2.0*(A2*(B4-B2)+A4*(B2-B6)-A6*(B4-B6)) + \
                  (A0*(B6-B4)+A2*(B4-B8)-A4*(B6-B8)))  \
                  /(10752.0)
        S331[jcalldd8] = w * \
                  ((A4*(9.0*B0-6.0*B2+B4)-2.0*A6*(3.0*B0-2.0*B2+3.0*B4)+A8*(B0-6.0*B2+9.0*B4)) - \
                  2.0*(A2*(9.0*B2-6.0*B4+B6)-2.0*A4*(3.0*B2-2.0*B4+3.0*B6)+A6*(B2-6.0*B4+9.0*B6)) + \
                  (A0*(9.0*B4-6.0*B6+B8)-2.0*A2*(3.0*B4-2.0*B6+3.0*B8)+A4*(B4-6.0*B6+9.0*B8)))  \
                  /(64512.0)



    jcalldd853 = (jcalldd==853)
    if(jcalldd853.sum() != 0):
        A0 = A33[jcalldd853,0]
        A1 = A33[jcalldd853,1]
        A2 = A33[jcalldd853,2]
        A3 = A33[jcalldd853,3]
        A4 = A33[jcalldd853,4]
        A5 = A33[jcalldd853,5]
        A6 = A33[jcalldd853,6]
        A7 = A33[jcalldd853,7]
        A8 = A33[jcalldd853,8]

        B0 = B33[jcalldd853,0]
        B1 = B33[jcalldd853,1]
        B2 = B33[jcalldd853,2]
        B3 = B33[jcalldd853,3]
        B4 = B33[jcalldd853,4]
        B5 = B33[jcalldd853,5]
        B6 = B33[jcalldd853,6]
        B7 = B33[jcalldd853,7]
        B8 = B33[jcalldd853,8]

        w = th.pow(zeta_b[jcalldd853,2],3.5)* \
                   th.pow(zeta_a[jcalldd853,2],5.5)* \
                   rij[jcalldd853]**9

        S333[jcalldd853] = w * \
                  (((A4-2.0*A6+A8)*(B0-2.0*B2+B4)) + \
                  2.0*((A3-2.0*A5+A7)*(B1-2.0*B3+B5)) - \
                  2.0*((A1-2.0*A3+A5)*(B3-2.0*B5+B7)) - \
                  ((A0-2.0*A2+A4)*(B4-2.0*B6+B8)))  \
                  /(9216.0*th.sqrt(th.tensor(35.0)))
        S332[jcalldd853] = -w * \
                  ((A4*(B2-B0)+A6*(B0-B4)-A8*(B2-B4)) + \
                  2.0*(A3*(B3-B1)+A5*(B1-B5)-A7*(B3-B5)) - \
                  2.0*(A1*(B5-B3)+A3*(B3-B7)-A5*(B5-B7)) - \
                  (A0*(B6-B4)+A2*(B4-B8)-A4*(B6-B8)))  \
                  /(2304.0*th.sqrt(th.tensor(35.0)))
        S331[jcalldd853] = w * \
                  ((A4*(9.0*B0-6.0*B2+B4)-2.0*A6*(3.0*B0-2.0*B2+3.0*B4)+A8*(B0-6.0*B2+9.0*B4)) + \
                  2.0*(A3*(9.0*B1-6.0*B3+B5)-2.0*A5*(3.0*B1-2.0*B3+3.0*B5)+A7*(B1-6.0*B3+9.0*B5)) - \
                  2.0*(A1*(9.0*B3-6.0*B5+B7)-2.0*A3*(3.0*B3-2.0*B5+3.0*B7)+A5*(B3-6.0*B5+9.0*B7)) - \
                  (A0*(9.0*B4-6.0*B6+B8)-2.0*A2*(3.0*B4-2.0*B6+3.0*B8)+A4*(B4-6.0*B6+9.0*B8)))  \
                  /(13824.0*th.sqrt(th.tensor(35.0)))



    jcalldd9 = (jcalldd==9)

    if(jcalldd9.sum() != 0):
        A0 = A33[jcalldd9,0]
        A1 = A33[jcalldd9,1]
        A2 = A33[jcalldd9,2]
        A3 = A33[jcalldd9,3]
        A4 = A33[jcalldd9,4]
        A5 = A33[jcalldd9,5]
        A6 = A33[jcalldd9,6]
        A7 = A33[jcalldd9,7]
        A8 = A33[jcalldd9,8]
        A9 = A33[jcalldd9,9]


        B0 = B33[jcalldd9,0]
        B1 = B33[jcalldd9,1]
        B2 = B33[jcalldd9,2]
        B3 = B33[jcalldd9,3]
        B4 = B33[jcalldd9,4]
        B5 = B33[jcalldd9,5]
        B6 = B33[jcalldd9,6]
        B7 = B33[jcalldd9,7]
        B8 = B33[jcalldd9,8]
        B9 = B33[jcalldd9,9]
        w = th.pow(zeta_b[jcalldd9,2],4.5)* \
                   th.pow(zeta_a[jcalldd9,2],5.5)* \
                   rij[jcalldd9]**10

        S333[jcalldd9] = w * \
                  (((A5-2.0*A7+A9)*(B0-2.0*B2+B4)) + \
                  1.0*((A4-2.0*A6+A8)*(B1-2.0*B3+B5)) - \
                  2.0*((A3-2.0*A5+A7)*(B2-2.0*B4+B6)) - \
                  2.0*((A2-2.0*A4+A6)*(B3-2.0*B5+B7)) + \
                  1.0*((A1-2.0*A3+A5)*(B4-2.0*B6+B8)) + \
                  ((A0-2.0*A2+A4)*(B5-2.0*B7+B9)))  \
                  /(129024.0*th.sqrt(th.tensor(10.0)))
        S332[jcalldd9] = -w * \
                  ((A5*(B2-B0)+A7*(B0-B4)-A9*(B2-B4)) + \
                  1.0*(A4*(B3-B1)+A6*(B1-B5)-A8*(B3-B5)) - \
                  2.0*(A3*(B4-B2)+A5*(B2-B6)-A7*(B4-B6)) - \
                  2.0*(A2*(B5-B3)+A4*(B3-B7)-A6*(B5-B7)) + \
                  1.0*(A1*(B6-B4)+A3*(B4-B8)-A5*(B6-B8)) + \
                  (A0*(B7-B5)+A2*(B5-B9)-A4*(B7-B9)))  \
                  /(32256.0*th.sqrt(th.tensor(10.0)))
        S331[jcalldd9] = w  * \
                  ((A5*(9.0*B0-6.0*B2+B4)-2.0*A7*(3.0*B0-2.0*B2+3.0*B4)+A9*(B0-6.0*B2+9.0*B4)) + \
                  1.0*(A4*(9.0*B1-6.0*B3+B5)-2.0*A6*(3.0*B1-2.0*B3+3.0*B5)+A8*(B1-6.0*B3+9.0*B5)) - \
                  2.0*(A3*(9.0*B2-6.0*B4+B6)-2.0*A5*(3.0*B2-2.0*B4+3.0*B6)+A7*(B2-6.0*B4+9.0*B6)) - \
                  2.0*(A2*(9.0*B3-6.0*B5+B7)-2.0*A4*(3.0*B3-2.0*B5+3.0*B7)+A6*(B3-6.0*B5+9.0*B7)) + \
                  1.0*(A1*(9.0*B4-6.0*B6+B8)-2.0*A3*(3.0*B4-2.0*B6+3.0*B8)+A5*(B4-6.0*B6+9.0*B8)) + \
                  (A0*(9.0*B5-6.0*B7+B9)-2.0*A2*(3.0*B5-2.0*B7+3.0*B9)+A4*(B5-6.0*B7+9.0*B9)))  \
                  /(193536.0*th.sqrt(th.tensor(10.0)))



    jcalldd10 = (jcalldd==10)
    if(jcalldd10.sum() != 0):
        A0 = A33[jcalldd10,0]
        A2 = A33[jcalldd10,2]
        A4 = A33[jcalldd10,4]
        A6 = A33[jcalldd10,6]
        A8 = A33[jcalldd10,8]
        A10 = A33[jcalldd10,10]


        B0 = B33[jcalldd10,0]
        B2 = B33[jcalldd10,2]
        B4 = B33[jcalldd10,4]
        B6 = B33[jcalldd10,6]
        B8 = B33[jcalldd10,8]
        B10 = B33[jcalldd10,10]

        w = th.pow(zeta_b[jcalldd10,2],5.5)* \
                   th.pow(zeta_a[jcalldd10,2],5.5)* \
                   rij[jcalldd10]**11

        S333[jcalldd10] = w * \
                  (((A6-2.0*A8+A10)*(B0-2.0*B2+B4)) - \
                  3.0*((A4-2.0*A6+A8)*(B2-2.0*B4+B6)) + \
                  3.0*((A2-2.0*A4+A6)*(B4-2.0*B6+B8)) - \
                  ((A0-2.0*A2+A4)*(B6-2.0*B8+B10)))  \
                  /(3870720)
        S332[jcalldd10] = -w * \
                  ((A6*(B2-B0)+A8*(B0-B4)-A10*(B2-B4)) - \
                  3.0*(A4*(B4-B2)+A6*(B2-B6)-A8*(B4-B6)) + \
                  3.0*(A2*(B6-B4)+A4*(B4-B8)-A6*(B6-B8)) - \
                  (A0*(B8-B6)+A2*(B6-B10)-A4*(B8-B10)))  \
                  /(967680)
        S331[jcalldd10] = w * \
                  ((A6*(9.0*B0-6.0*B2+B4)-2.0*A8*(3.0*B0-2.0*B2+3.0*B4)+A10*(B0-6.0*B2+9.0*B4)) - \
                  3.0*(A4*(9.0*B2-6.0*B4+B6)-2.0*A6*(3.0*B2-2.0*B4+3.0*B6)+A8*(B2-6.0*B4+9.0*B6)) + \
                  3.0*(A2*(9.0*B4-6.0*B6+B8)-2.0*A4*(3.0*B4-2.0*B6+3.0*B8)+A6*(B4-6.0*B6+9.0*B8)) - \
                  (A0*(9.0*B6-6.0*B8+B10)-2.0*A2*(3.0*B6-2.0*B8+3.0*B10)+A4*(B6-6.0*B8+9.0*B10)))  \
                  /(5806080)







    #
    #form di
    #check di_index.txt
    di=th.zeros((npairs,9,9),dtype=dtype, device=device)

    # diat coe
    # c : computed in coe, with shape (75)
    # c0 : used in diat, shape (3,5,5)
    # index relation
    # c with index 56, 41, 26, 53, 38, 50, 37, 20
    #  c, c0
    # 37, 1 3 3  1.0
    # 20, 2 2 2  ca
    # 35, 2 2 3  sa*sb
    # 50, 2 2 4  sa*cb
    # 23, 2 3 2  0.0
    # 38, 2 3 3  cb
    # 53, 2 3 4  -sb
    # 26, 2 4 2  -sa
    # 41, 2 4 3  ca*sb
    # 56, 2 4 4  ca*cb

    ## c[37] = 1.0
    ## c[23] = 0.0
    #all not listed are 0.0

    #check di_index.txt for details

    sasb = sa*sb
    sacb = sa*cb
    casb = ca*sb
    cacb = ca*cb
    di[...,0,0] = S111
    #jcallg2 = (jcall>2)
    #di[jcallg2,1,0] = S211[jcallg2,0]*ca[jcallg2,0]*sb[jcallg2,0]
    #the fraction of H-H is low, may not necessary to use indexing
    di[...,1,0] = S211*ca*sb
    di[...,2,0] = S211*sa*sb
    di[...,3,0] = S211*cb
    #jcall==4, pq1=pq2=2, ii=4, second - second row
    #di[jcall4,0,1] = -S121[jcall4]*ca[jcall4]*sb[jcall4]
    di[...,0,1] = -S121*casb
    #di[jcall4,0,2] = -S121[jcall4]*sa[jcall4]*sb[jcall4]
    di[...,0,2] = -S121*sasb
    di[...,0,3] = -S121*cb
    #di[jcall4,1,1] = -S221[jcall4]*ca[jcall4]**2*sb[jcall4]**2
    #                 +S222[jcall4]*(ca[jcall4]**2*cb[jcall4]**2+sa[jcall4]**2)
    di[...,1,1] = -S221*casb**2 \
                     +S222*(cacb**2+sa**2)
    di[...,1,2] = -S221*casb*sasb \
                     +S222*(cacb*sacb-sa*ca)
    di[...,1,3] = -S221*casb*cb \
                     -S222*cacb*sb
    di[...,2,1] = -S221*sasb*casb \
                     +S222*(sacb*cacb-ca*sa)
    di[...,2,2] = -S221*sasb**2 \
                     +S222*(sacb**2+ca**2)
    di[...,2,3] = -S221*sasb*cb \
                     -S222*sacb*sb
    di[...,3,1] = -S221*cb*casb \
                     -S222*sb*cacb
    di[...,3,2] = -S221*cb*sasb \
                     -S222*sb*sacb
    di[...,3,3] = -S221*cb**2 \
                     +S222*sb**2
    #on pairs with same atom, diagonal part
    #di[jcall==0,:,:] = th.diag(th.ones(4,dtype=dtype)).reshape((-1,4,4))
    

    #d/s
    di[...,4,0] = S311*(2*ca**2-1)*sb**2*th.sqrt(th.tensor(3.0/4.0))
    di[...,5,0] = S311*ca*sb*cb*th.sqrt(th.tensor(3.0))
    di[...,6,0] = S311*(cb**2-1/2*sb**2)
    di[...,7,0] = S311*th.sqrt(th.tensor(3.0))*sa*sb*cb
    di[...,8,0] = S311*th.sqrt(th.tensor(3.0))*sa*ca*sb**2

    di[...,0,4] = S131*(2*ca**2-1)*sb**2*th.sqrt(th.tensor(3.0/4.0))
    di[...,0,5] = S131*ca*sb*cb*th.sqrt(th.tensor(3.0))
    di[...,0,6] = S131*(cb**2-1/2*sb**2)
    di[...,0,7] = S131*th.sqrt(th.tensor(3.0))*sa*sb*cb
    di[...,0,8] = S131*th.sqrt(th.tensor(3.0))*sa*ca*sb**2

    #d/p

    di[...,4,1] =   -(S321*th.sqrt(th.tensor(3.0/4.0))*(2.0*ca**2-1.0)*sb**2*casb \
                   - S322*((2.0*ca**2-1.0)*sb*cb*ca*cb+2.0*sa*ca*sb*sa))  ##

    di[...,4,2] =  -(S321*th.sqrt(th.tensor(3.0/4.0))*(2.0*ca**2-1.0)*sb**2*sa*sb \
                   - S322*((2.0*ca**2-1.0)*sb*cb*sa*cb-2.0*sa*ca*sb*ca))  ##


    di[...,4,3] =  -(S321*th.sqrt(th.tensor(3.0/4.0))*(2.0*ca**2-1.0)*sb**2*cb \
                   + S322*((2.0*ca**2-1.0)*sb*cb*sb))  ##



    di[...,1,4] =  (S231*th.sqrt(th.tensor(3.0/4.0))*(2.0*ca**2-1.0)*sb**2*casb \
                   - S232*((2.0*ca**2-1.0)*sb*cb*ca*cb+2.0*sa*ca*sb*sa))  ##

    di[...,2,4] =  (S231*th.sqrt(th.tensor(3.0/4.0))*(2.0*ca**2-1.0)*sb**2*sa*sb \
                   - S232*((2.0*ca**2-1.0)*sb*cb*sa*cb-2.0*sa*ca*sb*ca))  ##


    di[...,3,4] =  (S231*th.sqrt(th.tensor(3.0/4.0))*(2.0*ca**2-1.0)*sb**2*cb \
                   + S232*((2.0*ca**2-1.0)*sb*cb*sb))  ##




    di[...,5,1] = -(S321*th.sqrt(th.tensor(3.0))*ca*sb*cb*ca*sb      \
                  -S322*(ca*(2.0*cb**2-1.0)*ca*cb+sa*cb*sa))   ##


    di[...,5,2] = -(S321*th.sqrt(th.tensor(3.0))*ca*sb*cb*sa*sb         \
                  -S322*(ca*(2.0*cb**2-1.0)*sacb-sa*cb*ca))    ##


    di[...,5,3] = -(S321*th.sqrt(th.tensor(3.0))*ca*sb*cb*cb         \
                  +S322*(ca*(2.0*cb**2-1.0)*sb))               ##

    di[...,1,5] = (S231*th.sqrt(th.tensor(3.0))*ca*sb*cb*ca*sb   \
                  -S232*(ca*(2.0*cb**2-1.0)*ca*cb+sa*cb*sa))   ##


    di[...,2,5] = (S231*th.sqrt(th.tensor(3.0))*ca*sb*cb*sa*sb         \
                  -S232*(ca*(2.0*cb**2-1.0)*sacb-sa*cb*ca))    ##


    di[...,3,5] = (S231*th.sqrt(th.tensor(3.0))*ca*sb*cb*cb         \
                  +S232*(ca*(2.0*cb**2-1.0)*sb))               ##





    di[...,6,1] = -(S321*(cb**2-0.5*sb**2)*ca*sb \
                  +S322*th.sqrt(th.tensor(3.0))*sb*cb*ca*cb)  ##

    di[...,6,2] = -(S321*(cb**2-0.5*sb**2)*sa*sb \
                  +S322*th.sqrt(th.tensor(3.0))*sb*cb*sa*cb)   ##

    di[...,6,3] = -(S321*(cb**2-0.5*sb**2)*cb \
                  -S322*th.sqrt(th.tensor(3.0))*sb*cb*sb)  ##

    di[...,1,6] = (S231*(cb**2-0.5*sb**2)*ca*sb \
                  +S232*th.sqrt(th.tensor(3.0))*sb*cb*ca*cb)  ##

    di[...,2,6] = (S231*(cb**2-0.5*sb**2)*sa*sb \
                  +S232*th.sqrt(th.tensor(3.0))*sb*cb*sa*cb)  ##

    di[...,3,6] = (S231*(cb**2-0.5*sb**2)*cb \
                  -S232*th.sqrt(th.tensor(3.0))*sb*cb*sb)  ##




    di[...,7,1] = -(S321*th.sqrt(th.tensor(3.0))*sa*sb*cb*ca*sb \
                 -S322*((sa*(2.0*cb**2-1.0))*ca*cb-ca*cb*sa))  ##

    di[...,7,2] = -(S321*th.sqrt(th.tensor(3.0))*sa*sb*cb*sa*sb \
                 -S322*((sa*(2.0*cb**2-1.0))*sa*cb+ca*cb*ca))  ##

    di[...,7,3] = -(S321*th.sqrt(th.tensor(3.0))*sa*sb*cb*cb \
                 +S322*((sa*(2.0*cb**2-1.0))*sb))  ##

    di[...,1,7] = (S231*th.sqrt(th.tensor(3.0))*sa*sb*cb*ca*sb \
                 -S232*((sa*(2.0*cb**2-1.0))*ca*cb-ca*cb*sa)) ##

    di[...,2,7] = (S231*th.sqrt(th.tensor(3.0))*sa*sb*cb*sa*sb \
                 -S232*((sa*(2.0*cb**2-1.0))*sa*cb+ca*cb*ca)) ##

    di[...,3,7] = (S231*th.sqrt(th.tensor(3.0))*sa*sb*cb*cb \
                 +S232*((sa*(2.0*cb**2-1.0))*sb)) ##




    di[...,1,8] = (S231*th.sqrt(th.tensor(3.0))*sa*ca*sb**2*ca*sb \
                  -S232*(2.0*sa*ca*sb*cb*ca*cb-sb*(2.0*ca**2-1.0)*sa))  ##

    di[...,2,8] = (S231*th.sqrt(th.tensor(3.0))*sa*ca*sb**2*sa*sb \
                  -S232*(2.0*sa*ca*sb*cb*sa*cb+sb*(2.0*ca**2-1.0)*ca))  ##

    di[...,3,8] = (S231*th.sqrt(th.tensor(3.0))*sa*ca*sb**2*cb \
                  +S232*(2.0*sa*ca*sb*cb*sb))   ##

    di[...,8,1] = -(S321*th.sqrt(th.tensor(3.0))*sa*ca*sb**2*ca*sb \
                  -S322*(2.0*sa*ca*sb*cb*ca*cb-sb*(2.0*ca**2-1.0)*sa))  ##

    di[...,8,2] = -(S321*th.sqrt(th.tensor(3.0))*sa*ca*sb**2*sa*sb \
                  -S322*(2.0*sa*ca*sb*cb*sa*cb+sb*(2.0*ca**2-1.0)*ca))  ##

    di[...,8,3] = -(S321*th.sqrt(th.tensor(3.0))*sa*ca*sb**2*cb \
                  +S322*(2.0*sa*ca*sb*cb*sb))   ##
    #d/d

    di[...,4,4] = S331*(th.sqrt(th.tensor(3.0/4.0))*(2.0*ca**2-1.0)*sb**2)**2 \
                  +S332*((sb*cb*(2.0*ca**2-1.0))**2+(2.0*sa*ca*sb)**2) \
                  +S333*(((2.0*ca**2-1.0)*cb**2+0.5*(2.0*ca**2-1.0)*sb**2)**2+(2.0*sa*ca*cb)**2)   ##

    di[...,4,5] = S331*(th.sqrt(th.tensor(3.0/4.0))*(2.0*ca**2-1.0)*sb**2)*th.sqrt(th.tensor(3.0))*ca*sb*cb \
                  +S332*((sb*cb*(2.0*ca**2-1.0)*ca*(2.0*cb**2-1.0)+(-2.0*sa*ca*sb)*(-sa*cb))) \
                  +S333*(((2.0*ca**2-1.0)*cb**2+0.5*(2.0*ca**2-1.0)*sb**2)*(-ca*sb*cb)-(2.0*sa*ca*cb*sa*sb))   ##

    di[...,4,6] = S331*(th.sqrt(th.tensor(3.0/4.0))*(2.0*ca**2-1.0)*sb**2)*(cb**2-0.5*sb**2) \
                  -S332*((sb*cb*(2.0*ca**2-1.0))*th.sqrt(th.tensor(3.0))*sb*cb) \
                  +S333*(((2.0*ca**2-1.0)*cb**2+0.5*(2.0*ca**2-1.0)*sb**2)*th.sqrt(th.tensor(3.0/4.0))*sb**2)  ##

    di[...,4,7] = S331*(th.sqrt(th.tensor(3.0/4.0))*(2.0*ca**2-1.0)*sb**2)*th.sqrt(th.tensor(3.0))*sa*sb*cb \
                  +S332*((sb*cb*(2.0*ca**2-1.0))*sa*(2.0*cb**2-1.0)-(2.0*sa*ca*sb)*ca*cb) \
                  +S333*(((2.0*ca**2-1.0)*cb**2+0.5*(2.0*ca**2-1.0)*sb**2)*(-sa*sb*cb)+(2.0*sa*ca*cb)*ca*sb)  ##

    di[...,4,8] = (S331*((th.sqrt(th.tensor(3.0/4.0))*(2.0*ca**2-1.0)*sb**2)*th.sqrt(th.tensor(3.0))*sa*ca*sb**2) \
                  +S332*((sb*cb*(2.0*ca**2-1.0))*2.0*sa*ca*sb*cb-(2.0*sa*ca*sb)*sb*(2.0*ca**2-1.0)) \
                  +S333*(((2.0*ca**2-1.0)*cb**2+0.5*(2.0*ca**2-1.0)*sb**2)*(2.0*sa*ca*cb**2+sa*ca*sb**2)-(2.0*sa*ca*cb)*cb*(2.0*ca**2-1.0)))  ##

    di[...,5,4] = (S331*(th.sqrt(th.tensor(3.0/4.0))*(2.0*ca**2-1.0)*sb**2)*th.sqrt(th.tensor(3.0))*ca*sb*cb \
                  +S332*((sb*cb*(2.0*ca**2-1.0)*ca*(2.0*cb**2-1.0)+(-2.0*sa*ca*sb)*(-sa*cb))) \
                  +S333*(((2.0*ca**2-1.0)*cb**2+0.5*(2.0*ca**2-1.0)*sb**2)*(-ca*sb*cb)-(2.0*sa*ca*cb*sa*sb)))  ##

    di[...,6,4] = (S331*(th.sqrt(th.tensor(3.0/4.0))*(2.0*ca**2-1.0)*sb**2)*(cb**2-0.5*sb**2) \
                  -S332*((sb*cb*(2.0*ca**2-1.0))*th.sqrt(th.tensor(3.0))*sb*cb) \
                  +S333*(((2.0*ca**2-1.0)*cb**2+0.5*(2.0*ca**2-1.0)*sb**2)*th.sqrt(th.tensor(3.0/4.0))*sb**2))  ##

    di[...,7,4] = (S331*(th.sqrt(th.tensor(3.0/4.0))*(2.0*ca**2-1.0)*sb**2)*th.sqrt(th.tensor(3.0))*sa*sb*cb \
                  +S332*((sb*cb*(2.0*ca**2-1.0))*sa*(2.0*cb**2-1.0)-(2.0*sa*ca*sb)*ca*cb) \
                  +S333*(((2.0*ca**2-1.0)*cb**2+0.5*(2.0*ca**2-1.0)*sb**2)*(-sa*sb*cb)+(2.0*sa*ca*cb)*ca*sb))  ##

    di[...,8,4] = (S331*((th.sqrt(th.tensor(3.0/4.0))*(2.0*ca**2-1.0)*sb**2)*th.sqrt(th.tensor(3.0))*sa*ca*sb**2) \
                  +S332*((sb*cb*(2.0*ca**2-1.0))*2.0*sa*ca*sb*cb-(2.0*sa*ca*sb)*sb*(2.0*ca**2-1.0)) \
                  +S333*(((2.0*ca**2-1.0)*cb**2+0.5*(2.0*ca**2-1.0)*sb**2)*(2.0*sa*ca*cb**2+sa*ca*sb**2)-(2.0*sa*ca*cb)*cb*(2.0*ca**2-1.0)))  ##

    di[...,5,5] = S331*(th.sqrt(th.tensor(3.0))*ca*sb*cb)**2 \
                  +S332*((ca*(2.0*cb**2-1.0))**2+(sa*cb)**2) \
                  +S333*((ca*sb*cb)**2+(sa*sb)**2)  ##

    di[...,5,6] = S331*(th.sqrt(th.tensor(3.0))*ca*sb*cb)*(cb**2-0.5*sb**2) \
                  +S332*((ca*(2.0*cb**2-1.0))*-(th.sqrt(th.tensor(3.0))*sb*cb)) \
                  -S333*(ca*sb*cb*th.sqrt(th.tensor(3.0/4.0))*sb**2)  ##

    di[...,5,7] = S331*(th.sqrt(th.tensor(3.0))*ca*sb*cb)*th.sqrt(th.tensor(3.0))*sa*sb*cb \
                  +S332*((ca*(2.0*cb**2-1.0))*sa*(2.0*cb**2-1.0)-sa*cb*ca*cb) \
                  +S333*((ca*sb*cb)*sa*sb*cb-sa*sb*ca*sb)  ##

    di[...,5,8] = S331*(th.sqrt(th.tensor(3.0))*ca*sb*cb)*(th.sqrt(th.tensor(3.0))*sa*ca*sb**2) \
                  +S332*((ca*(2.0*cb**2-1.0))*2.0*sa*ca*sb*cb-sa*cb*(2.0*ca**2-1.0)*sb) \
                  +S333*(-ca*sb*cb*(2.0*sa*ca*cb**2+sa*ca*sb**2)+sa*sb*cb*(2.0*ca**2-1.0)) ##

    di[...,6,5] = (S331*(th.sqrt(th.tensor(3.0))*ca*sb*cb)*(cb**2-0.5*sb**2) \
                  +S332*((ca*(2.0*cb**2-1.0))*-(th.sqrt(th.tensor(3.0))*sb*cb)) \
                  -S333*(ca*sb*cb*th.sqrt(th.tensor(3.0/4.0))*sb**2))  ##

    di[...,7,5] = (S331*(th.sqrt(th.tensor(3.0))*ca*sb*cb)*th.sqrt(th.tensor(3.0))*sa*sb*cb \
                  +S332*((ca*(2.0*cb**2-1.0))*sa*(2.0*cb**2-1.0)-sa*cb*ca*cb) \
                  +S333*((ca*sb*cb)*sa*sb*cb-sa*sb*ca*sb))  ##

    di[...,8,5] = (S331*(th.sqrt(th.tensor(3.0))*ca*sb*cb)*(th.sqrt(th.tensor(3.0))*sa*ca*sb**2) \
                  +S332*((ca*(2.0*cb**2-1.0))*2.0*sa*ca*sb*cb-sa*cb*(2.0*ca**2-1.0)*sb) \
                  +S333*(-ca*sb*cb*(2.0*sa*ca*cb**2+sa*ca*sb**2)+sa*sb*cb*(2.0*ca**2-1.0))) ##

    di[...,6,6] = S331*(cb**2-0.5*sb**2)**2 \
                  +S332*(th.sqrt(th.tensor(3.0))*sb*cb)**2 \
                  +S333*(th.sqrt(th.tensor(3.0/4.0))*sb**2)**2  ##

    di[...,6,7] =  S331*(cb**2-0.5*sb**2)*th.sqrt(th.tensor(3.0))*sa*sb*cb \
                  -S332*(th.sqrt(th.tensor(3.0))*sb*cb)*sa*(2.0*cb**2-1.0) \
                  -S333*(th.sqrt(th.tensor(3.0/4.0))*sb**2)*(sa*sb*cb)  ##

    di[...,6,8] =  S331*(cb**2-0.5*sb**2)*th.sqrt(th.tensor(3.0))*sa*ca*sb**2 \
                  -S332*(th.sqrt(th.tensor(3.0))*sb*cb)*2.0*sa*ca*sb*cb \
                  +S333*(th.sqrt(th.tensor(3.0/4.0))*sb**2)*(2.0*sa*ca*cb**2+sa*ca*sb**2)  ##

    di[...,7,6] = (S331*(cb**2-0.5*sb**2)*th.sqrt(th.tensor(3.0))*sa*sb*cb \
                  -S332*(th.sqrt(th.tensor(3.0))*sb*cb)*sa*(2.0*cb**2-1.0) \
                  -S333*(th.sqrt(th.tensor(3.0/4.0))*sb**2)*(sa*sb*cb))  ##

    di[...,8,6] = (S331*(cb**2-0.5*sb**2)*th.sqrt(th.tensor(3.0))*sa*ca*sb**2 \
                  -S332*(th.sqrt(th.tensor(3.0))*sb*cb)*2.0*sa*ca*sb*cb
                  +S333*(th.sqrt(th.tensor(3.0/4.0))*sb**2)*(2.0*sa*ca*cb**2+sa*ca*sb**2))  ##

    di[...,7,7] = S331*(th.sqrt(th.tensor(3.0))*sa*sb*cb)**2 \
                  +S332*((sa*(2.0*cb**2-1.0))**2+(ca*cb)**2) \
                  +S333*((sa*sb*cb)**2+(ca*sb)**2)  ##

    di[...,7,8] = (S331*(th.sqrt(th.tensor(3.0))*sa*sb*cb)*th.sqrt(th.tensor(3.0))*sa*ca*sb**2 \
                  +S332*((sa*(2.0*cb**2-1.0))*2.0*sa*ca*sb*cb+(ca*cb)*(2.0*ca**2-1.0)*sb) \
                  +S333*(-(sa*sb*cb)*(2.0*sa*ca*cb**2+sa*ca*sb**2)+ca*sb*cb*(2.0*ca**2-1.0)))  ##

    di[...,8,7] = (S331*(th.sqrt(th.tensor(3.0))*sa*sb*cb)*th.sqrt(th.tensor(3.0))*sa*ca*sb**2 \
                  +S332*((sa*(2.0*cb**2-1.0))*2.0*sa*ca*sb*cb+(ca*cb)*(2.0*ca**2-1.0)*sb) \
                  +S333*(-(sa*sb*cb)*(2.0*sa*ca*cb**2+sa*ca*sb**2)+ca*sb*cb*(2.0*ca**2-1.0)))  ##

    di[...,8,8] = S331*(th.sqrt(th.tensor(3.0))*sa*ca*sb**2)**2 \
                  +S332*((2.0*sa*ca*sb*cb)**2+((2.0*ca**2-1.0)*sb)**2) \
                  +S333*((2.0*sa*ca*cb**2+sa*ca*sb**2)**2+(cb*(2.0*ca**2-1.0))**2)  ##




    """
    return S111, S211, S121, S221, S222


    """
    """
    #  c, c0
    # 37, 1 3 3  1.0
    # 20, 2 2 2  ca
    # 35, 2 2 3  sa*sb
    # 50, 2 2 4  sa*cb
    # 23, 2 3 2  0.0
    # 38, 2 3 3  cb
    # 53, 2 3 4  -sb
    # 26, 2 4 2  -sa
    # 41, 2 4 3  ca*sb
    # 56, 2 4 4  ca*cb
    """
    #print("CALCULATION OF DIATOMIC OVERLAP: ", time.time()-t0, " (diat_overlapD.py, diatom_overlap_matrixD)")

    return di


#parameter_set = ['U_ss', 'U_pp', 'zeta_s', 'zeta_p','beta_s', 'beta_p',
#                 'g_ss', 'g_sp', 'g_pp', 'g_p2', 'h_sp', 'alpha']


def SET(rij, jcall, z1, z2):
    """
    get the A integrals and B integrals for diatom_overlap_matrix
    """
    #alpha, beta below is used in aintgs and bintgs, not the parameters for AM1/MNDO/PM3
    #rij: distance between atom i and j in atomic unit
    alpha = 0.5*rij*(z1[...,:]+z2[...,:])
    beta  = 0.5*rij*(z1[...,:]-z2[...,:])
    A = aintgs(alpha, jcall)
    B = bintgs(beta, jcall)
    return A,B


def aintgs(x0, jcall):
    """
    A integrals for diatom_overlap_matrix
    """
    dtype = x0.dtype
    device = x0.device
    #indxa, indxb, index to show which one to choose
    # idxa = parameter_set.index('zeta_X') X=s,p for atom a
    # idxb = parameter_set.index('zeta_X') for b
    #jcall : k in aintgs
    #alpha = 0.5*rab*(zeta_a + zeta_b)
    # c = exp(-alpha)
    #jcall >=2, and = 2,3,4 for first and second row elements

    #in same case x will be zero, which causes an issue when backpropagating
    #like zete_p from Hydrogen, H -  H Pair
    # or pairs for same atom, then rab = 0
    t=1.0/th.tensor(0.0,dtype=dtype, device=device)
    x=th.where(x0!=0,x0,t).reshape(-1,1)
    a1 = th.exp(-x)/x
    
    a2 = a1 +a1/x
    # jcall >= 2
    a3 = a1 + 2.0*a2/x
    # jcall >= 3
    jcallp3 = (jcall>=3).reshape((-1,1))
    a4 = th.where(jcallp3,a1+3.0*a3/x, th.tensor(0.0,dtype=dtype, device=device))
    # jcall >=4
    jcallp4 = (jcall>=4).reshape((-1,1))
    a5 = th.where(jcallp4,a1+4.0*a4/x, th.tensor(0.0,dtype=dtype, device=device))

    jcallp5 = (jcall>=5).reshape((-1,1))
    a6 = th.where(jcallp5,a1+5.0*a5/x, th.tensor(0.0,dtype=dtype, device=device))

    jcallp6 = (jcall>=6).reshape((-1,1))
    a7 = th.where(jcallp6,a1+6.0*a6/x, th.tensor(0.0,dtype=dtype, device=device))

    jcallp7 = (jcall>=7).reshape((-1,1))
    a8 = th.where(jcallp7,a1+7.0*a7/x, th.tensor(0.0,dtype=dtype, device=device))

    jcallp8 = (jcall>=8).reshape((-1,1))
    a9 = th.where(jcallp8,a1+8.0*a8/x, th.tensor(0.0,dtype=dtype, device=device))

    jcallp9 = (jcall>=9).reshape((-1,1))
    a10 = th.where(jcallp9,a1+9.0*a9/x, th.tensor(0.0,dtype=dtype, device=device))

    jcallp10 = (jcall>=10).reshape((-1,1))
    a11 = th.where(jcallp10,a1+10.0*a10/x, th.tensor(0.0,dtype=dtype, device=device))

    jcallp11 = (jcall>=11).reshape((-1,1))
    a12 = th.where(jcallp11,a1+11.0*a11/x, th.tensor(0.0,dtype=dtype, device=device))

    jcallp12 = (jcall>=12).reshape((-1,1))
    a13 = th.where(jcallp12,a1+12.0*a12/x, th.tensor(0.0,dtype=dtype, device=device))


    return th.cat((a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13),dim=1)

def bintgs(x0, jcall):
    """
    B integrals for diatom_overlap_matrix
    """
    #jcall not used here, but may used later for more element types
    # may ignore jcall, and just compute all the b1, b2, ..., b5 will be used
    #as na>=nb, and in the set in diat2.f
    # setc.sa = s2 =  zeta_b, setc.sb = s1 = zeta_a

    # beta = 0.5*rab*(setc.sb-setc.sa)
    # beta = 0.5*rab*(zeta_a - zeta_b)


    ## |x|<=0.5, last = 6, goto 60
    # else goto 40
    # for convenience, may use goto90 for x<1.0e-6
    x=x0.reshape(-1,1)
    absx = th.abs(x)

    cond1 = absx>0.5

    #tx = th.exp(x)/x    # exp(x)/x  #not working with backward
    #tx = th.where(cond1, th.exp(x)/x, th.tensor(0.0,dtype=dtype)) # not working with backward
    x_cond1 = x[cond1]
    tx_cond1 = th.exp(x_cond1)/x_cond1

    #tmx = -th.exp(-x)/x # -exp(-x)/x #not working with backward
    #tmx = th.where(cond1, -th.exp(-x)/x,  th.tensor(0.0,dtype=dtype)) #not working with backward
    tmx_cond1 = -th.exp(-x_cond1)/x_cond1
    #b1(x=0)=2, b2(x=0)=0,b3(x=0)=2/3,b4(x=0)=0,b5(x=0)=2/5

    #do some test to choose which one is faster
    #b1 = th.where(cond1,  tx+tmx         , th.tensor(2.0))
    #b2 = th.where(cond1, -tx+tmx+b1/x    , th.tensor(0.0))
    #b3 = th.where(cond1,  tx+tmx+2.0*b2/x, th.tensor(2.0/3.0))
    #b4 = th.where(cond1, -tx+tmx+3.0*b3/x, th.tensor(0.0))
    #b5 = th.where(cond1,  tx+tmx+4.0*b4/x, th.tensor(2.0/5.0))
    #b4 = th.where(cond1 & (jcall>=3), -tx+tmx+3.0*b3/x, th.tensor(0.0))
    #b5 = th.where(cond1 & (jcall>=4),  tx+tmx+4.0*b4/x, th.tensor(2.0/5.0))

    #do some test to choose which one is faster

    #can't use this way th.where to do backpropagating
    b1 = th.ones_like(x)*2.0
    b2 = th.zeros_like(x)
    b3 = th.ones_like(x)*(2.0/3.0)
    b4 = th.zeros_like(x)
    b5 = th.ones_like(x)*(2.0/5.0)
    b6 = th.zeros_like(x)
    b7 = th.ones_like(x)*(2.0/7.0)
    b8 = th.zeros_like(x)
    b9 = th.ones_like(x)*(2.0/9.0)
    b10 = th.zeros_like(x)
    b11 = th.ones_like(x)*(2.0/11.0)
    b12 = th.zeros_like(x)
    b13 = th.ones_like(x)*(2.0/13.0)





    b1_cond1 =  tx_cond1 + tmx_cond1
    b1[cond1] =  b1_cond1

    b2_cond1 = -tx_cond1 + tmx_cond1 +     b1_cond1/x_cond1
    b2[cond1] =  b2_cond1

    b3_cond1 =  tx_cond1 + tmx_cond1 + 2.0*b2_cond1/x_cond1
    b3[cond1] = b3_cond1

    b4_cond1 = -tx_cond1 + tmx_cond1 + 3.0*b3_cond1/x_cond1
    b4[cond1] = b4_cond1

    b5_cond1 =  tx_cond1 + tmx_cond1 + 4.0*b4_cond1/x_cond1
    b5[cond1] = b5_cond1

    b6_cond1 = -tx_cond1 + tmx_cond1 + 5.0*b5_cond1/x_cond1
    b6[cond1] = b6_cond1

    b7_cond1 =  tx_cond1 + tmx_cond1 + 6.0*b6_cond1/x_cond1
    b7[cond1] = b7_cond1

    b8_cond1 = -tx_cond1 + tmx_cond1 + 7.0*b7_cond1/x_cond1
    b8[cond1] = b8_cond1

    b9_cond1 =  tx_cond1 + tmx_cond1 + 8.0*b8_cond1/x_cond1
    b9[cond1] = b9_cond1

    b10_cond1 = -tx_cond1 + tmx_cond1 + 9.0*b9_cond1/x_cond1
    b10[cond1] = b10_cond1

    b11_cond1 =  tx_cond1 + tmx_cond1 + 10.0*b10_cond1/x_cond1
    b11[cond1] = b11_cond1

    b12_cond1 = -tx_cond1 + tmx_cond1 + 11.0*b11_cond1/x_cond1
    b12[cond1] = b12_cond1

    b13_cond1 =  tx_cond1 + tmx_cond1 + 12.0*b12_cond1/x_cond1
    b13[cond1] = b13_cond1









    #b1 = th.where(cond1,  tx + tmx           , th.tensor(2.0, dtype=dtype))
    #b2 = th.where(cond1, -tx + tmx +     b1/x, th.tensor(0.0, dtype=dtype))
    #b3 = th.where(cond1,  tx + tmx + 2.0*b2/x, th.tensor(2.0/3.0, dtype=dtype))
    #b4 = th.where(cond1, -tx + tmx + 3.0*b3/x, th.tensor(0.0, dtype=dtype))
    #b5 = th.where(cond1,  tx + tmx + 4.0*b4/x, th.tensor(2.0/5.0, dtype=dtype))


    #|x|<=0.5
    cond2 = (absx<=0.5) & (absx>1.0e-6)
    # b_{i+1}(x) = \sum_m (-x)^m * (2.0 * (m+i+1)%2 ) / m! / (m+i+1)
    # i is even, i=0,2,4, m = 0,2,4,6
    # b_{i+1} (x) = \sum_{m=0,2,4,6} x^m * 2.0/(m! * (m+i+1))
    # factors
    #      m =   0    2      4       6
    # i=0        2  1/3   1/60  1/2520     2/(m!*(m+1))
    # i=2      2/3  1/5   1/84  1/3240     2/(m!*(m+3))
    # i=4      2/5  1/7   1/108  1/3960     2/(m!*(m+5))
    # i=6      2/7  1/9   1/132  1/4680     2/(m!*(m+7))
    # i=8      2/9  1/11  1/156  1/5400     2/(m!*(m+9))
    # i=10     2/11 1/13  1/180  1/6120     2/(m!*(m+11))
    # i=12     2/13 1/15  1/204  1/6840     2/(m!*(m+13))
    x = x[cond2]
    b1[cond2] = 2.0     + x**2/3.0 + x**4/60.0 + x**6/2520.0
    b3[cond2] = 2.0/3.0 + x**2/5.0 + x**4/84.0 + x**6/3240.0
    b5[cond2] = 2.0/5.0 + x**2/7.0 + x**4/108.0 + x**6/3960.0
    b7[cond2] = 2.0/7.0 + x**2/9.0 + x**4/132.0 + x**6/4680.0
    b9[cond2] = 2.0/9.0 + x**2/11.0 + x**4/156.0 + x**6/5400.0
    b11[cond2] = 2.0/11.0 + x**2/13.0 + x**4/180.0 + x**6/6120.0
    b13[cond2] = 2.0/13.0 + x**2/15.0 + x**4/204.0 + x**6/6840.0

    #b5[cond2 & (jcall>=4)] = -x*2.0/7.0 - x**3/27.0 - x**5/660.0

    # b_{i+1}(x) = \sum_m (-x)^m * (2.0 * (m+i+1)%2 ) / m! / (m+i+1)
    # i is odd, i = 1,3, m= 1,3,5
    # b_{i+1} (x) = \sum_{m=1,3,5} -x^m * 2.0/(m! * (m+i+1))
    # factors
    #      m =    1     3      5
    # i=1       2/3  1/15  1/420  2/(m!*(m+2))
    # i=3       2/5  1/21  1/540  2/(m!*(m+4))
    # i=5       2/7  1/27  1/660  2/(m!*(m+6))
    # i=7       2/9  1/33  1/780  2/(m!*(m+8))
    # i=9       2/11 1/39  1/900  2/(m!*(m+10))
    # i=11      2/13 1/45  1/1020  2/(m!*(m+12))


    b2[cond2] = -2.0/3.0*x - x**3/15.0 - x**5/420.0
    b4[cond2] = -2.0/5.0*x - x**3/21.0 - x**5/540.0
    b6[cond2] = -2.0/7.0*x - x**3/27.0 - x**5/660.0
    b8[cond2] = -2.0/9.0*x - x**3/33.0 - x**5/780.0
    b10[cond2] = -2.0/11.0*x - x**3/39.0 - x**5/900.0
    b12[cond2] = -2.0/13.0*x - x**3/45.0 - x**5/1020.0


    #b4[cond2 & (jcall>=3)] = -2.0/5.0*x[cond2] - x[cond2]**3/21.0 - x[cond2]**5/540.0
    #print(b1)
    #print(b3)
    return th.cat((b1, b2, b3, b4, b5, b6, b7, b8, b9 ,b10, b11, b12, b13), dim=1)
