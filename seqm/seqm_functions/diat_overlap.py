import torch


def diatom_overlap_matrix(ni, nj, xij, rij, zeta_a, zeta_b, qn_int):
    """
    compute the overlap matrix for each pair
    """
    dtype = xij.dtype
    device = xij.device

    # xij unit vector points from i to j: xij = (xj-xi)/|xi-xj|, shape (npairs, 3)
    # rij : distance between atom i and j in atomic units, shape (npairs,)
    # ni, nj, atomic number of atom i and j, shape (npairs,)
    # x = x2-x1
    # y = y2-y1
    # z = z2-z1
    # for first atom index == second atom index, (same atom)
    # x = x1, y = y1, z = z1
    npairs = xij.shape[0]
    # zeta_a, zeta_b: zeta_s and zeta_p for atom pair a-b, shape(npairs, 2)
    # output di: overlap matrix between AOs from atom i an j, shape (npairs, 4,4)
    # 0,1,2,3: sigma, px, py, pz

    xy = torch.norm(xij[..., :2], dim=1)

    # ``self.where(condition, y)`` is equivalent to ``torch.where(condition, self, y)``.

    tmp = torch.where(xij[..., 2] < 0.0, torch.tensor(-1.0, dtype=dtype, device=device),
                      torch.where(xij[..., 2] > 0.0, torch.tensor(1.0, dtype=dtype, device=device),
                      torch.tensor(0.0, dtype=dtype, device=device)))

    # ca = torch.where(xy>=1.0e-10, xij[...,0]/xy, tmp)
    cond_xy = xy >= 1.0e-10
    ca = tmp.clone()
    ca[cond_xy] = xij[cond_xy, 0] / xy[cond_xy]

    cb = torch.where(xy >= 1.0e-10, xij[..., 2], tmp)  # xij is a unti vector already
    # cb = torch.where(xy>=1.0e-10, xij[...,2]/rij, tmp)
    # del tmp
    # sa = torch.where(xy>=1.0e-10, xij[...,1]/xy, torch.tensor(0.0,dtype=dtype, device=device))
    sa = torch.zeros_like(xy)

    sa[cond_xy] = xij[cond_xy, 1] / xy[cond_xy]
    # sb = torch.where(xy>=1.0e-10, xy/rij, torch.tensor(0.0,dtype=dtype))
    sb = torch.where(xy >= 1.0e-10, xy, torch.tensor(0.0, dtype=dtype, device=device))
    ################################
    # ok to use torch.where here as postion doesn't require grad
    # if update to do MD, ca, cb, sa, sb should be chaneged to the indexing version
    ################################

    # overlap matrix in the local frame
    # first row  - first row  : ii = 1, jcall = 2
    # first row  - second row : ii = 2, jcall = 3
    # second row - secpmd row : ii = 4, jcall = 4

    # only first and second row are included here
    # one-element slice to keep dim
    # jcall shape (npairs, )

    # jcall = torch.where((ni>1) & (nj>1), torch.tensor(4, dtype=dtypeint),
    #        torch.where((ni==1) & (nj==1), torch.tensor(2, dtype=dtypeint),
    #        torch.tensor(3, dtype=dtypeint)))

    jcall = torch.zeros_like(ni)
    qni = qn_int[ni]
    qnj = qn_int[nj]
    jcall[(qni == 1) & (qnj == 1)] = 2
    jcall[(qni == 2) & (qnj == 1)] = 3
    jcall[(qni == 2) & (qnj == 2)] = 4
    if torch.any(jcall == 0):
        raise ValueError("\nError from diat.py, overlap matrix\nSome elements are not supported yet")

    # na>=nb
    # setc.isp=2
    # setc.ips=1
    # setc.sa = s2 = zeta_B
    # setc.sb = s1 = zeta_A

    # change to atomic units, a0 here is taken from mopac, different from the standard value
    # r is already in unit of bohr radius, AU

    # parameter_set = ['U_ss', 'U_pp', 'zeta_s', 'zeta_p','beta_s', 'beta_p',
    #                 'g_ss', 'g_sp', 'g_pp', 'g_p2', 'h_sp', 'alpha']

    A111, B111 = SET(rij, jcall, zeta_a[..., 0], zeta_b[..., 0])
    S111 = torch.zeros((npairs,), dtype=dtype, device=device)
    # print(id(S111))
    # s-s
    jcall2 = (jcall == 2)  # ii=1

    S111[jcall2] = torch.pow(zeta_a[jcall2, 0]
                             * zeta_b[jcall2, 0]
                             * rij[jcall2]**2, 1.5) * \
        (A111[jcall2, 2] * B111[jcall2, 0] - B111[jcall2, 2] * A111[jcall2, 0]) / 4.0
    jcall3 = (jcall == 3)  # ii=2
    # print(id(S111))
    S111[jcall3] = torch.pow(zeta_b[jcall3, 0], 1.5) * \
        torch.pow(zeta_a[jcall3, 0], 2.5) * \
        rij[jcall3]**4 * \
        (A111[jcall3, 3] * B111[jcall3, 0] - B111[jcall3, 3] * A111[jcall3, 0]
         + A111[jcall3, 2] * B111[jcall3, 1] - B111[jcall3, 2] * A111[jcall3, 1]) / \
        (torch.sqrt(torch.tensor(3.0)) * 8.0)
    jcall4 = (jcall == 4)  # ii=4
    # print(id(S111))
    S111[jcall4] = torch.pow(zeta_b[jcall4, 0]
                             * zeta_a[jcall4, 0], 2.5) * \
        rij[jcall4]**5 * \
        (A111[jcall4, 4] * B111[jcall4, 0] + B111[jcall4, 4] * A111[jcall4, 0]
         - 2.0 * A111[jcall4, 2] * B111[jcall4, 2]) / 48.0
    # jcall4 = (jcall==4)
    # print(id(S111))
    S211 = torch.zeros_like(S111)
    # s-p
    A211, B211 = SET(rij, jcall, zeta_a[..., 1], zeta_b[..., 0])
    S211[jcall3] = torch.pow(zeta_b[jcall3, 0], 1.5) * \
        torch.pow(zeta_a[jcall3, 1], 2.5) * \
        rij[jcall3]**4 * \
        (A211[jcall3, 2] * B211[jcall3, 0] - B211[jcall3, 2] * A211[jcall3, 0]
         + A211[jcall3, 3] * B211[jcall3, 1] - B211[jcall3, 3] * A211[jcall3, 1]) / 8.0
    S211[jcall4] = torch.pow(zeta_b[jcall4, 0] * zeta_a[jcall4, 1], 2.5) * \
        rij[jcall4]**5 * \
        (A211[jcall4, 3] * (B211[jcall4, 0] - B211[jcall4, 2])
         - A211[jcall4, 1] * (B211[jcall4, 2] - B211[jcall4, 4])
         + B211[jcall4, 3] * (A211[jcall4, 0] - A211[jcall4, 2])
         - B211[jcall4, 1] * (A211[jcall4, 2] - A211[jcall4, 4])) \
        / (16.0 * torch.sqrt(torch.tensor(3.0)))
    S121 = torch.zeros_like(S111)
    A121, B121 = SET(rij, jcall, zeta_a[..., 0], zeta_b[..., 1])
    S121[jcall4] = torch.pow(zeta_b[jcall4, 1] * zeta_a[jcall4, 0], 2.5) * \
        rij[jcall4]**5 * \
        (A121[jcall4, 3] * (B121[jcall4, 0] - B121[jcall4, 2])
         - A121[jcall4, 1] * (B121[jcall4, 2] - B121[jcall4, 4])
         - B121[jcall4, 3] * (A121[jcall4, 0] - A121[jcall4, 2])
         + B121[jcall4, 1] * (A121[jcall4, 2] - A121[jcall4, 4])) \
        / (16.0 * torch.sqrt(torch.tensor(3.0)))
    S221 = torch.zeros_like(S111)
    A22, B22 = SET(rij, jcall, zeta_a[..., 1], zeta_b[..., 1])
    w = torch.pow(zeta_b[jcall4, 1]
                  * zeta_a[jcall4, 1], 2.5) * \
        rij[jcall4]**5 / 16.0
    S221[jcall4] = -w * \
        (B22[jcall4, 2] * (A22[jcall4, 4] + A22[jcall4, 0])
         - A22[jcall4, 2] * (B22[jcall4, 4] + B22[jcall4, 0]))
    S222 = torch.zeros_like(S111)
    S222[jcall4] = 0.5 * w * \
        (A22[jcall4, 4] * (B22[jcall4, 0] - B22[jcall4, 2])
         - B22[jcall4, 4] * (A22[jcall4, 0] - A22[jcall4, 2])
         - A22[jcall4, 2] * B22[jcall4, 0] + B22[jcall4, 2] * A22[jcall4, 0])
    #
    di = torch.zeros((npairs, 4, 4), dtype=dtype, device=device)

    sasb = sa * sb
    sacb = sa * cb
    casb = ca * sb
    cacb = ca * cb

    di[..., 0, 0] = S111
    # jcallg2 = (jcall>2)
    # di[jcallg2,1,0] = S211[jcallg2,0]*ca[jcallg2,0]*sb[jcallg2,0]
    # the fraction of H-H is low, may not necessary to use indexing
    di[..., 1, 0] = S211 * ca * sb
    di[..., 2, 0] = S211 * sa * sb
    di[..., 3, 0] = S211 * cb
    # jcall==4, pq1=pq2=2, ii=4, second - second row
    # di[jcall4,0,1] = -S121[jcall4]*ca[jcall4]*sb[jcall4]
    di[jcall4, 0, 1] = -S121[jcall4] * casb[jcall4]
    # di[jcall4,0,2] = -S121[jcall4]*sa[jcall4]*sb[jcall4]
    di[jcall4, 0, 2] = -S121[jcall4] * sasb[jcall4]
    di[jcall4, 0, 3] = -S121[jcall4] * cb[jcall4]
    # di[jcall4,1,1] = -S221[jcall4]*ca[jcall4]**2*sb[jcall4]**2
    #                 +S222[jcall4]*(ca[jcall4]**2*cb[jcall4]**2+sa[jcall4]**2)
    di[jcall4, 1, 1] = -S221[jcall4] * casb[jcall4]**2 \
        + S222[jcall4] * (cacb[jcall4]**2 + sa[jcall4]**2)
    di[jcall4, 1, 2] = -S221[jcall4] * casb[jcall4] * sasb[jcall4] \
        + S222[jcall4] * (cacb[jcall4] * sacb[jcall4] - sa[jcall4] * ca[jcall4])
    di[jcall4, 1, 3] = -S221[jcall4] * casb[jcall4] * cb[jcall4] \
        - S222[jcall4] * cacb[jcall4] * sb[jcall4]
    di[jcall4, 2, 1] = -S221[jcall4] * sasb[jcall4] * casb[jcall4] \
        + S222[jcall4] * (sacb[jcall4] * cacb[jcall4] - ca[jcall4] * sa[jcall4])
    di[jcall4, 2, 2] = -S221[jcall4] * sasb[jcall4]**2 \
        + S222[jcall4] * (sacb[jcall4]**2 + ca[jcall4]**2)
    di[jcall4, 2, 3] = -S221[jcall4] * sasb[jcall4] * cb[jcall4] \
        - S222[jcall4] * sacb[jcall4] * sb[jcall4]
    di[jcall4, 3, 1] = -S221[jcall4] * cb[jcall4] * casb[jcall4] \
        - S222[jcall4] * sb[jcall4] * cacb[jcall4]
    di[jcall4, 3, 2] = -S221[jcall4] * cb[jcall4] * sasb[jcall4] \
        - S222[jcall4] * sb[jcall4] * sacb[jcall4]
    di[jcall4, 3, 3] = -S221[jcall4] * cb[jcall4]**2 \
        + S222[jcall4] * sb[jcall4]**2
    # on pairs with same atom, diagonal part
    # di[jcall==0,:,:] = torch.diag(torch.ones(4,dtype=dtype)).reshape((-1,4,4))

    return di


def SET(rij, jcall, z1, z2):
    """
    get the A integrals and B integrals for diatom_overlap_matrix
    """
    # alpha, beta below is used in aintgs and bintgs, not the parameters for AM1/MNDO/PM3
    # rij: distance between atom i and j in atomic unit
    alpha = 0.5 * rij * (z1[..., :] + z2[..., :])
    beta = 0.5 * rij * (z1[..., :] - z2[..., :])
    A = aintgs(alpha, jcall)
    B = bintgs(beta, jcall)

    return A, B


def aintgs(x0, jcall):
    """
    A integrals for diatom_overlap_matrix
    """
    dtype = x0.dtype
    device = x0.device
    # indxa, indxb, index to show which one to choose
    # idxa = parameter_set.index('zeta_X') X=s,p for atom a
    # idxb = parameter_set.index('zeta_X') for b
    # jcall : k in aintgs
    # alpha = 0.5*rab*(zeta_a + zeta_b)
    # c = exp(-alpha)
    # jcall >=2, and = 2,3,4 for first and second row elements

    # in same case x will be zero, which causes an issue when backpropagating
    # like zete_p from Hydrogen, H -  H Pair
    # or pairs for same atom, then rab = 0
    t = 1.0 / torch.tensor(0.0, dtype=dtype, device=device)
    x = torch.where(x0 != 0, x0, t).reshape(-1, 1)

    a1 = torch.exp(-x) / x
    a2 = a1 + a1 / x
    # jcall >= 2
    a3 = a1 + 2.0 * a2 / x
    # jcall >= 3
    jcallp3 = (jcall >= 3).reshape((-1, 1))
    a4 = torch.where(jcallp3, a1 + 3.0 * a3 / x, torch.tensor(0.0, dtype=dtype, device=device))
    # jcall >=4
    jcallp4 = (jcall >= 4).reshape((-1, 1))
    a5 = torch.where(jcallp4, a1 + 4.0 * a4 / x, torch.tensor(0.0, dtype=dtype, device=device))

    return torch.cat((a1, a2, a3, a4, a5), dim=1)


def bintgs(x0, jcall):
    """
    B integrals for diatom_overlap_matrix
    """
    # jcall not used here, but may used later for more element types
    # may ignore jcall, and just compute all the b1, b2, ..., b5 will be used
    # as na>=nb, and in the set in diat2.f
    # setc.sa = s2 =  zeta_b, setc.sb = s1 = zeta_a

    # beta = 0.5*rab*(setc.sb-setc.sa)
    # beta = 0.5*rab*(zeta_a - zeta_b)

    # |x|<=0.5, last = 6, goto 60
    # else goto 40
    # for convenience, may use goto90 for x<1.0e-6
    x = x0.reshape(-1, 1)
    absx = torch.abs(x)

    cond1 = absx > 0.5

    # tx = torch.exp(x)/x    # exp(x)/x  #not working with backward
    # tx = torch.where(cond1, torch.exp(x)/x, torch.tensor(0.0,dtype=dtype)) # not working with backward
    x_cond1 = x[cond1]
    tx_cond1 = torch.exp(x_cond1) / x_cond1

    # tmx = -torch.exp(-x)/x # -exp(-x)/x #not working with backward
    # tmx = torch.where(cond1, -torch.exp(-x)/x,  torch.tensor(0.0,dtype=dtype)) #not working with backward
    tmx_cond1 = -torch.exp(-x_cond1) / x_cond1
    # b1(x=0)=2, b2(x=0)=0,b3(x=0)=2/3,b4(x=0)=0,b5(x=0)=2/5

    # do some test to choose which one is faster
    # b1 = torch.where(cond1,  tx+tmx         , torch.tensor(2.0))
    # b2 = torch.where(cond1, -tx+tmx+b1/x    , torch.tensor(0.0))
    # b3 = torch.where(cond1,  tx+tmx+2.0*b2/x, torch.tensor(2.0/3.0))
    # b4 = torch.where(cond1, -tx+tmx+3.0*b3/x, torch.tensor(0.0))
    # b5 = torch.where(cond1,  tx+tmx+4.0*b4/x, torch.tensor(2.0/5.0))
    # b4 = torch.where(cond1 & (jcall>=3), -tx+tmx+3.0*b3/x, torch.tensor(0.0))
    # b5 = torch.where(cond1 & (jcall>=4),  tx+tmx+4.0*b4/x, torch.tensor(2.0/5.0))

    # do some test to choose which one is faster

    # can't use this way torch.where to do backpropagating
    b1 = torch.ones_like(x) * 2.0
    b2 = torch.zeros_like(x)
    b3 = torch.ones_like(x) * (2.0 / 3.0)
    b4 = torch.zeros_like(x)
    b5 = torch.ones_like(x) * (2.0 / 5.0)
    b1_cond1 = tx_cond1 + tmx_cond1
    b1[cond1] = b1_cond1
    b2_cond1 = -tx_cond1 + tmx_cond1 + b1_cond1 / x_cond1
    b2[cond1] = b2_cond1
    b3_cond1 = tx_cond1 + tmx_cond1 + 2.0 * b2_cond1 / x_cond1
    b3[cond1] = b3_cond1
    b4_cond1 = -tx_cond1 + tmx_cond1 + 3.0 * b3_cond1 / x_cond1
    b4[cond1] = b4_cond1
    b5_cond1 = tx_cond1 + tmx_cond1 + 4.0 * b4_cond1 / x_cond1
    b5[cond1] = b5_cond1

    # b1 = torch.where(cond1,  tx + tmx           , torch.tensor(2.0, dtype=dtype))
    # b2 = torch.where(cond1, -tx + tmx +     b1/x, torch.tensor(0.0, dtype=dtype))
    # b3 = torch.where(cond1,  tx + tmx + 2.0*b2/x, torch.tensor(2.0/3.0, dtype=dtype))
    # b4 = torch.where(cond1, -tx + tmx + 3.0*b3/x, torch.tensor(0.0, dtype=dtype))
    # b5 = torch.where(cond1,  tx + tmx + 4.0*b4/x, torch.tensor(2.0/5.0, dtype=dtype))

    # |x|<=0.5
    cond2 = (absx <= 0.5) & (absx > 1.0e-6)
    # b_{i+1}(x) = \sum_m (-x)^m * (2.0 * (m+i+1)%2 ) / m! / (m+i+1)
    # i is even, i=0,2,4, m = 0,2,4,6
    # b_{i+1} (x) = \sum_{m=0,2,4,6} x^m * 2.0/(m! * (m+i+1))
    # factors
    #      m =   0    2      4       6
    # i=0        2  1/3   1/60  1/2520     2/(m!*(m+1))
    # i=2      2/3  1/5   1/84  1/3240     2/(m!*(m+3))
    # i=4      2/5  1/7  1/108  1/3960     2/(m!*(m+5))
    b1[cond2] = 2.0 + x[cond2]**2 / 3.0 + x[cond2]**4 / 60.0 + x[cond2]**6 / 2520.0
    b3[cond2] = 2.0 / 3.0 + x[cond2]**2 / 5.0 + x[cond2]**4 / 84.0 + x[cond2]**6 / 3240.0
    b5[cond2] = 2.0 / 5.0 + x[cond2]**2 / 7.0 + x[cond2]**4 / 108.0 + x[cond2]**6 / 3960.0
    # b5[cond2 & (jcall>=4)] = -x[cond2]*2.0/7.0 - x[cond2]**3/27.0 - x[cond2]**5/660.0

    # b_{i+1}(x) = \sum_m (-x)^m * (2.0 * (m+i+1)%2 ) / m! / (m+i+1)
    # i is odd, i = 1,3, m= 1,3,5
    # b_{i+1} (x) = \sum_{m=1,3,5} -x^m * 2.0/(m! * (m+i+1))
    # factors
    #      m =    1     3      5
    # i=1       2/3  1/15  1/420  2/(m!*(m+2))
    # i=3       2/5  1/21  1/540  2/(m!*(m+4))

    b2[cond2] = -2.0 / 3.0 * x[cond2] - x[cond2]**3 / 15.0 - x[cond2]**5 / 420.0
    b4[cond2] = -2.0 / 5.0 * x[cond2] - x[cond2]**3 / 21.0 - x[cond2]**5 / 540.0
    # b4[cond2 & (jcall>=3)] = -2.0/5.0*x[cond2] - x[cond2]**3/21.0 - x[cond2]**5/540.0

    return torch.cat((b1, b2, b3, b4, b5), dim=1)
