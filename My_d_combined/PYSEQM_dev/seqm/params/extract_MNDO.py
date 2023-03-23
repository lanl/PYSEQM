from elements import ELEMENTS
from collections import defaultdict
import re

parameterslist=["ussm",
                "uppm",
                "uddm",
                "zsm",
                "zpm",
                "zdm",
                "betasm",
                "betapm",
                "gssm",
                "gspm",
                "gppm",
                "gp2m",
                "hspm",
                "alpm",
                "polvom"]
parameters=dict()

for pname in parameterslist:
    parameters[pname] = [0.0 for _ in range(107)]

def floatconvert(str):
    a,b=str.lower().split('d')
    a=float(a)
    b=int(b)
    return a*10**b

with open("parameters_for_mndo_C.f90") as f:
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
            parameters[pname][atomn-1] = v

"""
parameterslist=["ussm",
                "uppm",
                "uddm",
                "zsm",
                "zpm",
                "zdm",
                "betasm",
                "betapm",
                "gssm",
                "gspm",
                "gppm",
                "gp2m",
                "hspm",
                "alpm",
                "polvom"]
"""

with open('parameters_MNDO_MOPAC.csv', 'w') as f:
    f.write("   N, sym,         U_ss,         U_pp,         U_dd,       zeta_s,       zeta_p,       zeta_d,       beta_s,       beta_p,         g_ss,         g_sp,         g_pp,         g_p2,         h_sp,        alpha,       polvom\n")
    for atomn in range(107):
        f.write("% 4d, % 3s, " % (atomn+1, ELEMENTS[atomn+1].symbol))
        for pname in parameterslist:
            if 'polvom' not in pname:
                f.write("% 12.7f, " % parameters[pname][atomn])
            else:
                f.write("% 12.7f\n" % parameters[pname][atomn])
