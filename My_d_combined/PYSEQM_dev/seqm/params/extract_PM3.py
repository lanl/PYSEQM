from elements import ELEMENTS
from collections import defaultdict
import re

parameterslist=["usspm3",
                "upppm3",
                "uddpm3",
                "zspm3",
                "zppm3",
                "zdpm3",
                "betasp",
                "betapp",
                "gsspm3",
                "gsppm3",
                "gpppm3",
                "gp2pm3",
                "hsppm3",
                "alppm3",
                "guesp1",
                "guesp2",
                "guesp3"]
parameters=dict()

for pname in parameterslist:
    if 'guesp' in pname:
        parameters[pname] = [[0.0, 0.0, 0.0, 0.0] for _ in range(107)]
    else:
        parameters[pname] = [0.0 for _ in range(107)]
#print(parameters.keys())

def floatconvert(str):
    a,b=str.lower().split('d')
    a=float(a)
    b=int(b)
    return a*10**b

with open("parameters_for_pm3_C.f90") as f:
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
            if 'guesp' in pname:
                indx = int(t[3])
                parameters[pname][atomn-1][indx-1] = v
            else:
                parameters[pname][atomn-1] = v

"""
parameterslist=["usspm3",
                "upppm3",
                "uddpm3",
                "zspm3",
                "zppm3",
                "zdpm3",
                "betasp",
                "betapp",
                "gsspm3",
                "gsppm3",
                "gpppm3",
                "gp2pm3",
                "hsppm3",
                "alppm3",
                "guesp1",
                "guesp2",
                "guesp3"]
"""

with open('parameters_PM3_MOPAC.csv', 'w') as f:
    f.write("   N, sym,         U_ss,         U_pp,         U_dd,       zeta_s,       zeta_p,       zeta_d,       beta_s,       beta_p,         g_ss,         g_sp,         g_pp,         g_p2,         h_sp,        alpha, ")
    f.write(" Gaussian1_K,  Gaussian1_L,  Gaussian1_M, ")
    f.write(" Gaussian2_K,  Gaussian2_L,  Gaussian2_M, ")
    f.write(" Gaussian3_K,  Gaussian3_L,  Gaussian3_M, ")
    f.write(" Gaussian4_K,  Gaussian4_L,  Gaussian4_M\n")
    for atomn in range(107):
        f.write("% 4d, % 3s, " % (atomn+1, ELEMENTS[atomn+1].symbol))
        for pname in parameterslist:
            if 'guesp' not in pname:
                f.write("% 12.7f, " % parameters[pname][atomn])
        for i in range(3):
            f.write("% 12.7f, % 12.7f, % 12.7f, " % (parameters["guesp1"][atomn][i], parameters["guesp2"][atomn][i], parameters["guesp3"][atomn][i]))
        i=3
        f.write("% 12.7f, % 12.7f, % 12.7f\n" % (parameters["guesp1"][atomn][i], parameters["guesp2"][atomn][i], parameters["guesp3"][atomn][i]))
