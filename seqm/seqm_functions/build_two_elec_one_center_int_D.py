import torch
from .constants import Constants
from .two_elec_two_center_int import GetSlaterCondonParameter
from .cal_par import *
from .constants import ev
import sys
import numpy
# import scipy.special
from .parameters import  PWCCT
import math

#two electron two center integrals



def calc_integral(zetas, zetap, zetad, Z, size, maskd, P0, F0SD, G2SD):
        const = Constants()
        
        #### populations elements with d-oribitals
        isY = (zetad != 0)
        j = 0
        device = P0.device

        ####
        psindex = Z.clone()
        dindex  = Z.clone()
        psindex = const.qn_int.to(device)[Z]
        dindex = const.qnD_int.to(device)[Z]
        j = 0
        R016 = torch.zeros(zetap.shape[0],dtype = torch.double, device = device)
        R066 = torch.zeros(zetap.shape[0],dtype = torch.double, device = device)
        R244 = torch.zeros(zetap.shape[0],dtype = torch.double, device = device)
        R246 = torch.zeros(zetap.shape[0],dtype = torch.double, device = device)
        R466 = torch.zeros(zetap.shape[0],dtype = torch.double, device = device)
        R266 = torch.zeros(zetap.shape[0],dtype = torch.double, device = device)
        R036 = torch.zeros(zetap.shape[0],dtype = torch.double, device = device)
        R155 = torch.zeros(zetap.shape[0],dtype = torch.double, device = device)
        R125 = torch.zeros(zetap.shape[0],dtype = torch.double, device = device)
        R236 = torch.zeros(zetap.shape[0],dtype = torch.double, device = device)
        R234 = torch.zeros(zetap.shape[0],dtype = torch.double, device = device)
        R355 = torch.zeros(zetap.shape[0],dtype = torch.double, device = device)
        for k in zetad:
            if(dindex[j] > 0 ):
                R016[j]   = GetSlaterCondonParameter(0,int(psindex[j]),zetas[j],int(psindex[j]),zetas[j],int(dindex[j]),zetad[j],int(dindex[j]),zetad[j])
                R066[j]   = GetSlaterCondonParameter(0,int(dindex[j]),zetad[j],int(dindex[j]),zetad[j],int(dindex[j]),zetad[j],int(dindex[j]),zetad[j])
                R244[j]   = GetSlaterCondonParameter(2,int(psindex[j]),zetas[j],int(dindex[j]),zetad[j],int(psindex[j]),zetas[j],int(dindex[j]),zetad[j])
                R246[j]   = GetSlaterCondonParameter(2,int(psindex[j]),zetas[j],int(dindex[j]),zetad[j],int(dindex[j]),zetad[j],int(dindex[j]),zetad[j])
                R466[j]   = GetSlaterCondonParameter(4,int(dindex[j]),zetad[j],int(dindex[j]),zetad[j],int(dindex[j]),zetad[j],int(dindex[j]),zetad[j])
                R266[j]   = GetSlaterCondonParameter(2,int(dindex[j]),zetad[j],int(dindex[j]),zetad[j],int(dindex[j]),zetad[j],int(dindex[j]),zetad[j])
                if ( abs(F0SD[j]) > 10**(-9)):
                    R016[j] = F0SD[j]
                if ( abs(G2SD[j]) > 10**(-9)):
                    R244[j] = G2SD[j]

            if(psindex[j] > 0 and dindex[j] > 0 ):
                R036[j]   = GetSlaterCondonParameter(0,int(psindex[j]),zetap[j],int(psindex[j]),zetap[j],int(dindex[j]),zetad[j],int(dindex[j]),zetad[j])
                R155[j]   = GetSlaterCondonParameter(1,int(psindex[j]),zetap[j],int(dindex[j]),zetad[j],int(psindex[j]),zetap[j],int(dindex[j]),zetad[j])
                R125[j]   = GetSlaterCondonParameter(1,int(psindex[j]),zetas[j],int(psindex[j]),zetap[j],int(psindex[j]),zetap[j],int(dindex[j]),zetad[j])
                R236[j]   = GetSlaterCondonParameter(2,int(psindex[j]),zetap[j],int(psindex[j]),zetap[j],int(dindex[j]),zetad[j],int(dindex[j]),zetad[j])
                R234[j]   = GetSlaterCondonParameter(2,int(psindex[j]),zetap[j],int(psindex[j]),zetap[j],int(psindex[j]),zetas[j],int(dindex[j]),zetad[j])
                R355[j]   = GetSlaterCondonParameter(3,int(psindex[j]),zetap[j],int(dindex[j]),zetad[j],int(psindex[j]),zetap[j],int(dindex[j]),zetad[j])
            j = j + 1

        integral = torch.zeros(zetap.shape[0],53,dtype=torch.double, device = device)
        #print("HERE")
        #print(R016,R066,R244,R246,R466,R266)
        #print(R036,R155,R125,R236,R234,R355)


        half = 0.5
        ONE = 1.0
        TWO = 2.0
        THREE = 3.0
        FOUR = 4.0
        S3 = 1.7320508075689
        S5 = 2.2360679774998
        S15 = 3.8729833462074
        integral[isY,1-1] = R016[isY]
        integral[isY, 2-1] = (TWO/(THREE*S5))*R125[isY]
        integral[isY, 3-1] = (ONE/S15)*R125[isY]
        integral[isY, 4-1] = (TWO/(5.0*S5))*R234[isY]
        integral[isY, 5-1] = R036[isY] + (FOUR/35.0)*R236[isY]
        integral[isY, 6-1] = R036[isY] + (TWO /35.0)*R236[isY]
        integral[isY, 7-1] = R036[isY] - (FOUR/35.0)*R236[isY]
        integral[isY, 8-1] = -(ONE/(THREE*S5))*R125[isY]
        integral[isY, 9-1] = math.sqrt(THREE/125.0)*R234[isY]
        integral[isY,10-1] = (S3   /35.0)*R236[isY]
        integral[isY,11-1] = (THREE/35.0)*R236[isY]
        integral[isY,12-1] = -(0.2/S5)*R234[isY]
        integral[isY,13-1] = R036[isY] - (TWO/35.0)*R236[isY]
        integral[isY,14-1] = -(TWO*S3/35.0)*R236[isY]
        integral[isY,15-1] = -integral[isY,3-1]
        integral[isY,16-1] = -integral[isY,11-1]
        integral[isY,17-1] = -integral[isY, 9-1]
        integral[isY,18-1] = -integral[isY,14-1]
        integral[isY,19-1] = 0.2*R244[isY]
        integral[isY,20-1] = (TWO/(7.0*S5))*R246[isY]
        integral[isY,21-1] =  integral[isY,20-1]*half
        integral[isY,22-1] = -integral[isY,20-1]
        integral[isY,23-1] = (FOUR  /15.0)*R155[isY] + (27.0  /245.0)*R355[isY]
        integral[isY,24-1] = (TWO*S3/15.0)*R155[isY] - (9.0*S3/245.0)*R355[isY]
        integral[isY,25-1] = (ONE/15.0)*R155[isY] + (18.0   /245.0)*R355[isY]
        integral[isY,26-1] = -(S3/15.0)*R155[isY] + (12.0*S3/245.0)*R355[isY]
        integral[isY,27-1] = -(S3/15.0)*R155[isY] - (THREE*S3/245.0)*R355[isY]
        integral[isY,28-1] = -integral[isY,27-1]
        integral[isY,29-1] = R066[isY] + (FOUR/49.0)*R266[isY] + (FOUR / 49.0)*R466[isY]
        integral[isY,30-1] = R066[isY] + (TWO /49.0)*R266[isY] - (24.0/441.0)*R466[isY]
        integral[isY,31-1] = R066[isY] - (FOUR/49.0)*R266[isY] + ( 6.0/441.0)*R466[isY]
        integral[isY,32-1] = math.sqrt(THREE/245.0)*R246[isY]
        integral[isY,33-1] = 0.2*R155[isY] + (24.0/245.0)*R355[isY]
        integral[isY,34-1] = 0.2*R155[isY] - ( 6.0/245.0)*R355[isY]
        integral[isY,35-1] = (THREE/49.0)*R355[isY]
        integral[isY,36-1] = (ONE/49.0)*R266[isY] + (30.0  /441.0)*R466[isY]
        integral[isY,37-1] = (S3 /49.0)*R266[isY] - (5.0*S3/441.0)*R466[isY]
        integral[isY,38-1] = R066[isY] - (TWO/49.0)*R266[isY] -  (FOUR/441.0)*R466[isY]
        integral[isY,39-1] = -(TWO*S3/49.0)*R266[isY] + (10.0*S3/441.0)*R466[isY]
        integral[isY,40-1] = -integral[isY,32-1]
        integral[isY,41-1] = -integral[isY,34-1]
        integral[isY,42-1] = -integral[isY,35-1]
        integral[isY,43-1] = -integral[isY,37-1]
        integral[isY,44-1] = (THREE/49.0)*R266[isY] + (20.0/441.0)*R466[isY]
        integral[isY,45-1] = -integral[isY,39-1]
        integral[isY,46-1] = 0.20*R155[isY]-(THREE/35.0)*R355[isY]
        integral[isY,47-1] = -integral[isY,46-1]
        integral[isY,48-1] = (FOUR /49.0)*R266[isY] + (15.0/441.0)*R466[isY]
        integral[isY,49-1] = (THREE/49.0)*R266[isY] - ( 5.0/147.0)*R466[isY]
        integral[isY,50-1] = -integral[isY,49-1]
        integral[isY,51-1] = R066[isY] + (FOUR/49.0)*R266[isY] - (34.0/441.0)*R466[isY]
        integral[isY,52-1] = (35.0/441.0)*R466[isY]

        
        
        W = torch.zeros(size,243, device = device)
        IntRf1 = [ \
        19,19,19,19,19, 3, 3, 8, 3, 3,33,33, 8,27,25,35,33,15, 8, 3, \
         3,34, 3,27,15,33,35, 8,28,25,33,33, 3, 2, 3, 3,34,24,35, 3, \
        41,26,35,35,33, 2,23,33,35, 3,15, 1,32,22,40, 3, 6,11,14, 0, \
        15, 6,18,16, 0, 7,11,16,19,33,33,35,29,44,22,48,44,52, 3, 1, \
        32,21,32, 3,11, 6,10,11, 7,11,11, 3,11, 6,10,11,34,32,38,37, \
        50,19,33,35,33,32,44,29,21,37,36,44,44, 8, 8, 2,22,21, 1,20, \
        21,22, 8,14,10,13,14, 8,18,13,10,14, 2,10, 5,10,27,28,22,37, \
        31,43,24,21,37,30,39,19,25,25,23,48,36,20,29,36,48, 3, 1,40, \
        21,32,11, 7,11, 3,16,11,10, 6, 3,16,10, 6,11,41,40,38,45,49, \
        34,38,32,37,26,21,45,30,37,19,35,33,33,40,44,44,21,43,36,29, \
        44, 3,32, 1,22, 3, 0,14,11, 6, 3, 0,11,14, 6,11,11, 7,51,35, \
        32,49,37,38,27,37,22,31,35,32,50,39,38,19,33,33,35,52,44,22, \
        48,44,29]

        IntRf2 = [ \
        19,19,19,19,19, 9, 9,12, 9, 3,33,33, 8,27,25,35,33,17,12, 9, \
         9,35, 3,27,15,33,35, 8,28,25,33,33, 9, 4, 9, 3,35,26,34, 3, \
        42,24,34,35,33, 2,23,33,35, 3,15,19,32,22,40, 9,33,35,27,47, \
        17,33,28,42,46,35,34,41,19,33,33,35,29,44,22,48,44,52, 3,19, \
        32,21,32, 9,34,33,26,35,35,34,34, 9,35,33,24,35,35,32,44,39, \
         0,19,33,35,33,32,44,29,21,37,36,44,44, 8, 8, 2,22,21,19,20, \
        21,22,12,27,24,25,27,12,28,25,24,27, 4,26,23,26,27,28,22,37, \
        48,43,26,21,39,36,37,19,25,25,23,48,36,20,29,36,48, 3,19,40, \
        21,32,34,35,34, 9,41,35,26,33, 9,42,24,33,35,42,40,44,43, 0, \
        35,44,32,37,24,21,43,36,39,19,35,33,33,40,44,44,21,43,36,29, \
        44, 3,32,19,22, 9,46,27,35,33, 9,47,35,27,33,34,34,35,52,34, \
        32, 0,39,44,27,37,22,48,34,32, 0,37,44,19,33,33,35,52,44,22, \
        48,44,29]


        IntRep = [ \
         1, 1, 1, 1, 1, 3, 3, 8, 3, 9, 6, 6,12,14,13, 7, 6,15, 8, 3, \
         3,11, 9,14,17, 6, 7,12,18,13, 6, 6, 3, 2, 3, 9,11,10,11, 9, \
        16,10,11, 7, 6, 4, 5, 6, 7, 9,17,19,32,22,40, 3,33,34,27,46, \
        15,33,28,41,47,35,35,42, 1, 6, 6, 7,29,38,22,31,38,51, 9,19, \
        32,21,32, 3,35,33,24,34,35,35,35, 3,34,33,26,34,11,32,44,37, \
        49, 1, 6, 7, 6,32,38,29,21,39,30,38,38,12,12, 4,22,21,19,20, \
        21,22, 8,27,26,25,27, 8,28,25,26,27, 2,24,23,24,14,18,22,39, \
        48,45,10,21,37,36,37, 1,13,13, 5,31,30,20,29,30,31, 9,19,40, \
        21,32,35,35,35, 3,42,34,24,33, 3,41,26,33,34,16,40,44,43,50, \
        11,44,32,39,10,21,43,36,37, 1, 7, 6, 6,40,38,38,21,45,30,29, \
        38, 9,32,19,22, 3,47,27,34,33, 3,46,34,27,33,35,35,35,52,11, \
        32,50,37,44,14,39,22,48,11,32,49,37,44, 1, 6, 6, 7,51,38,22, \
        31,38,29]


        j = 0
        
        while (j < 243):
            i1 = IntRf1[j]
            i2 = IntRf2[j]
            W[maskd,j] = integral[...,IntRep[j]-1]
#            if(j == 203):
#                print(W[maskd,j],i1,i2,IntRep[j]-1)
            if(i1>0):
                    W[maskd,j] = W[maskd,j]-0.25*integral[...,i1-1]
#                    if(j == 203):
#                        print(W[maskd,j],i1)
            if(i2>0):
                    W[maskd,j] = W[maskd,j]-0.25*integral[...,i2-1]
#                    if(j == 203):
#                        print(W[maskd,j],i2)

            j = j +1
       

        return W


