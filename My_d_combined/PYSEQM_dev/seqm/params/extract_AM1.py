from elements import ELEMENTS
from collections import defaultdict
import re

parameterslist=["ussam1",
                "uppam1",
                "zsam1",
                "zpam1",
                "zdam1",
                "betasa",
                "betapa",
                "gssam1",
                "gspam1",
                "gppam1",
                "gp2am1",
                "hspam1",
                "alpam1",
                "guesa1",
                "guesa2",
                "guesa3"]
parameters=dict()

for pname in parameterslist:
    if 'guesa' in pname:
        parameters[pname] = [[0.0, 0.0, 0.0, 0.0] for _ in range(107)]
    else:
        parameters[pname] = [0.0 for _ in range(107)]

def floatconvert(str):
    a,b=str.lower().split('d')
    a=float(a)
    b=int(b)
    return a*10**b

with open("parameters_for_AM1_C.f90") as f:
    for l in f:
        if 'data' in l:
            sign=1.0
            if '-' in l:
                sign=-1.0
            t=re.sub('[(,)/-]',' ',l.strip()).split()
            pname=t[1].lower()
            atomn=int(t[2])
            v=sign*floatconvert(t[-1])
            #print(l.strip())
            #print(t)
            #print(v)
            if 'guesa' in pname:
                indx = int(t[3])
                parameters[pname][atomn-1][indx-1] = v
            else:
                parameters[pname][atomn-1] = v

"""
parameterslist=["ussam1",
                "uppam1",
                "zsam1",
                "zpam1",
                "zdam1",
                "betasa",
                "betapa",
                "gssam1",
                "gspam1",
                "gppam1",
                "gp2am1",
                "hspam1",
                "alpam1",
                "guesa1",
                "guesa2",
                "guesa3"]
"""

with open('parameters_AM1_MOPAC.csv', 'w') as f:
    f.write("   N, sym,         U_ss,         U_pp,       zeta_s,       zeta_p,       zeta_d,       beta_s,       beta_p,         g_ss,         g_sp,         g_pp,         g_p2,         h_sp,        alpha, ")
    f.write(" Gaussian1_K,  Gaussian1_L,  Gaussian1_M, ")
    f.write(" Gaussian2_K,  Gaussian2_L,  Gaussian2_M, ")
    f.write(" Gaussian3_K,  Gaussian3_L,  Gaussian3_M, ")
    f.write(" Gaussian4_K,  Gaussian4_L,  Gaussian4_M\n")
    for atomn in range(107):
        f.write("% 4d, % 3s, " % (atomn+1, ELEMENTS[atomn+1].symbol))
        for pname in parameterslist:
            if 'guesa' not in pname:
                f.write("% 12.7f, " % parameters[pname][atomn])
        for i in range(3):
            f.write("% 12.7f, % 12.7f, % 12.7f, " % (parameters["guesa1"][atomn][i], parameters["guesa2"][atomn][i], parameters["guesa3"][atomn][i]))
        i=3
        f.write("% 12.7f, % 12.7f, % 12.7f\n" % (parameters["guesa1"][atomn][i], parameters["guesa2"][atomn][i], parameters["guesa3"][atomn][i]))
