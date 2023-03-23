import torch
import sys
#as there are 0 padding in many matrixes
#to save space as well as to make diag, matmul faster
#pack and unpack remove this padding and put the padding back


def packoned(x, nhoho, nho, nHydro, norb):
        x0 = torch.zeros((norb,norb), dtype=x.dtype, device=x.device)
        x0[:nhoho,:nhoho]=x[:nhoho,:nhoho]

        ##DP
        a = x[:nhoho,nhoho:(nhoho+nho*9//4)]
        x0[:nhoho,nhoho:(nhoho+nho):4] = a[:,::9]
        x0[:nhoho,nhoho+1:(nhoho+nho):4] = a[:,1::9]
        x0[:nhoho,nhoho+2:(nhoho+nho):4] = a[:,2::9]
        x0[:nhoho,nhoho+3:(nhoho+nho):4] = a[:,3::9]

        ##PD
        a = x[nhoho:(nhoho+nho*9//4),:nhoho]
        x0[nhoho:(nhoho+nho):4,:nhoho] = a[::9,:]
        x0[nhoho+1:(nhoho+nho):4,:nhoho] = a[1::9,:]
        x0[nhoho+2:(nhoho+nho):4,:nhoho] = a[2::9,:]
        x0[nhoho+3:(nhoho+nho):4,:nhoho] = a[3::9,:]

        ##PP 
        ##DIAGONAL
        a = x[nhoho:(nhoho+nho*9//4),nhoho:(nhoho+nho*9//4)]
        x0[nhoho:(nhoho+nho):4,nhoho:(nhoho+nho):4] = a[::9,::9]
        x0[nhoho+1:(nhoho+nho):4,nhoho+1:(nhoho+nho):4] = a[1::9,1::9]
        x0[nhoho+2:(nhoho+nho):4,nhoho+2:(nhoho+nho):4] = a[2::9,2::9]
        x0[nhoho+3:(nhoho+nho):4,nhoho+3:(nhoho+nho):4] = a[3::9,3::9]

        ##0*
        x0[nhoho:(nhoho+nho):4,nhoho+1:(nhoho+nho):4] = a[::9,1::9]
        x0[nhoho:(nhoho+nho):4,nhoho+2:(nhoho+nho):4] = a[::9,2::9]
        x0[nhoho:(nhoho+nho):4,nhoho+3:(nhoho+nho):4] = a[::9,3::9]

        ##*0
        x0[nhoho+1:(nhoho+nho):4,nhoho:(nhoho+nho):4] = a[1::9,::9]
        x0[nhoho+2:(nhoho+nho):4,nhoho:(nhoho+nho):4] = a[2::9,::9]
        x0[nhoho+3:(nhoho+nho):4,nhoho:(nhoho+nho):4] = a[3::9,::9]

        ##1*
        x0[nhoho+1:(nhoho+nho):4,nhoho+2:(nhoho+nho):4] = a[1::9,2::9]
        x0[nhoho+1:(nhoho+nho):4,nhoho+3:(nhoho+nho):4] = a[1::9,3::9]

        ##*1
        x0[nhoho+2:(nhoho+nho):4,nhoho+1:(nhoho+nho):4] = a[2::9,1::9]
        x0[nhoho+3:(nhoho+nho):4,nhoho+1:(nhoho+nho):4] = a[3::9,1::9]

        ##23
        x0[nhoho+2:(nhoho+nho):4,nhoho+3:(nhoho+nho):4] = a[2::9,3::9]

        ##32
        x0[nhoho+3:(nhoho+nho):4,nhoho+2:(nhoho+nho):4] = a[3::9,2::9]

        ##DS/SD
        x0[:nhoho,(nhoho+nho):(nhoho+nho+nHydro)] = x[:nhoho,(nhoho+nho*9//4):(nhoho+nho*9//4+nHydro*9):9]
        x0[(nhoho+nho):(nhoho+nho+nHydro),:nhoho] = x[(nhoho+nho*9//4):(nhoho+nho*9//4+nHydro*9):9,:nhoho]


        a = x[nhoho:(nhoho+nho*9//4),nhoho+nho*9//4:(nhoho+nho*9//4+nHydro*9)]
        x0[nhoho:(nhoho+nho):4,(nhoho+nho):(nhoho+nho+nHydro)] =  a[::9,::9]
        x0[nhoho+1:(nhoho+nho):4,(nhoho+nho):(nhoho+nho+nHydro)] =  a[1::9,::9]
        x0[nhoho+2:(nhoho+nho):4,(nhoho+nho):(nhoho+nho+nHydro)] =  a[2::9,::9]
        x0[nhoho+3:(nhoho+nho):4,(nhoho+nho):(nhoho+nho+nHydro)] =  a[3::9,::9]

        a = x[nhoho+nho*9//4:(nhoho+nho*9//4+nHydro*9),nhoho:(nhoho+nho*9//4)]
        x0[(nhoho+nho):(nhoho+nho+nHydro),nhoho:(nhoho+nho):4] =  a[::9,::9]
        x0[(nhoho+nho):(nhoho+nho+nHydro),nhoho+1:(nhoho+nho):4] =  a[::9,1::9]
        x0[(nhoho+nho):(nhoho+nho+nHydro),nhoho+2:(nhoho+nho):4] =  a[::9,2::9]
        x0[(nhoho+nho):(nhoho+nho+nHydro),nhoho+3:(nhoho+nho):4] =  a[::9,3::9]



        ##SS
        x0[(nhoho+nho):(nhoho+nho+nHydro),(nhoho+nho):(nhoho+nho+nHydro)] =x[(nhoho+nho*9//4):(nhoho+nho*9//4+nHydro*9):9,(nhoho+nho*9//4):(nhoho+nho*9//4+nHydro*9):9]


        return x0

def unpackoned(x0, nhoho, nho, nHydro, size):
    x = torch.zeros((size, size), dtype=x0.dtype, device=x0.device)
    x[:nhoho,:nhoho]=x0[:nhoho,:nhoho]
    
    ##DP
    a = x0[:nhoho,nhoho:(nhoho+nho)]
    # print(a)
    # print(a.size())
    # print(x)
    # print(x.size())
    x[:nhoho,nhoho:(nhoho+nho*9//4):9] = a[:,::4]
    x[:nhoho,nhoho+1:(nhoho+nho*9//4):9] = a[:,1::4]
    x[:nhoho,nhoho+2:(nhoho+nho*9//4):9] = a[:,2::4]
    x[:nhoho,nhoho+3:(nhoho+nho*9//4):9] = a[:,3::4]

    ##PD
    a = x0[nhoho:(nhoho+nho),:nhoho]
    x[nhoho:(nhoho+nho*9//4):9,:nhoho] = a[::4,:]
    x[nhoho+1:(nhoho+nho*9//4):9,:nhoho] = a[1::4,:]
    x[nhoho+2:(nhoho+nho*9//4):9,:nhoho] = a[2::4,:]
    x[nhoho+3:(nhoho+nho*9//4):9,:nhoho] = a[3::4,:] 

    ##PP
    ##DIAGONAL
    a = x0[nhoho:(nhoho+nho),nhoho:(nhoho+nho)]
    x[nhoho:(nhoho+nho*9//4):9,nhoho:(nhoho+nho*9//4):9] = a[::4,::4]
    x[nhoho+1:(nhoho+nho*9//4):9,nhoho+1:(nhoho+nho*9//4):9] = a[1::4,1::4]
    x[nhoho+2:(nhoho+nho*9//4):9,nhoho+2:(nhoho+nho*9//4):9] = a[2::4,2::4]
    x[nhoho+3:(nhoho+nho*9//4):9,nhoho+3:(nhoho+nho*9//4):9] = a[3::4,3::4]

    ###0*
    x[nhoho:(nhoho+nho*9//4):9,nhoho+1:(nhoho+nho*9//4):9] = a[::4,1::4]
    x[nhoho:(nhoho+nho*9//4):9,nhoho+2:(nhoho+nho*9//4):9] = a[::4,2::4]
    x[nhoho:(nhoho+nho*9//4):9,nhoho+3:(nhoho+nho*9//4):9] = a[::4,3::4]

    ##*0
    x[nhoho+1:(nhoho+nho*9//4):9,nhoho:(nhoho+nho*9//4):9] = a[1::4,::4]
    x[nhoho+2:(nhoho+nho*9//4):9,nhoho:(nhoho+nho*9//4):9] = a[2::4,::4]
    x[nhoho+3:(nhoho+nho*9//4):9,nhoho:(nhoho+nho*9//4):9] = a[3::4,::4]

    ##1*
    x[nhoho+1:(nhoho+nho*9//4):9,nhoho+2:(nhoho+nho*9//4):9] = a[1::4,2::4]
    x[nhoho+1:(nhoho+nho*9//4):9,nhoho+3:(nhoho+nho*9//4):9] = a[1::4,3::4]

    ##*1
    x[nhoho+2:(nhoho+nho*9//4):9,nhoho+1:(nhoho+nho*9//4):9] = a[2::4,1::4]
    x[nhoho+3:(nhoho+nho*9//4):9,nhoho+1:(nhoho+nho*9//4):9] = a[3::4,1::4]

    ##23
    x[nhoho+2:(nhoho+nho*9//4):9,nhoho+3:(nhoho+nho*9//4):9] = a[2::4,3::4]

    ##32
    x[nhoho+3:(nhoho+nho*9//4):9,nhoho+2:(nhoho+nho*9//4):9] = a[3::4,2::4]
 
    ##SD
    x[nhoho+nho*9//4:(nhoho+nho*9//4+nHydro*9):9,:nhoho] =  x0[nhoho+nho:(nhoho+nho+nHydro),:nhoho]
    ##DS
    x[:nhoho,nhoho+nho*9//4:(nhoho+nho*9//4+nHydro*9):9] =  x0[:nhoho,nhoho+nho:(nhoho+nho+nHydro)]

    ##PS
    a = x0[nhoho:nhoho+nho,nhoho+nho:(nhoho+nho+nHydro)]
    x[nhoho:(nhoho+nho*9//4):9,(nhoho+nho*9//4):(nhoho+nho*9//4+nHydro*9):9] = a[::4,:]
    x[nhoho+1:(nhoho+nho*9//4):9,(nhoho+nho*9//4):(nhoho+nho*9//4+nHydro*9):9] = a[1::4,:]
    x[nhoho+2:(nhoho+nho*9//4):9,(nhoho+nho*9//4):(nhoho+nho*9//4+nHydro*9):9] = a[2::4,:]
    x[nhoho+3:(nhoho+nho*9//4):9,(nhoho+nho*9//4):(nhoho+nho*9//4+nHydro*9):9] = a[3::4,:]

    ##SP
    a = x0[nhoho+nho:(nhoho+nho+nHydro),nhoho:nhoho+nho]
    x[(nhoho+nho*9//4):(nhoho+nho*9//4+nHydro*9):9,nhoho:(nhoho+nho*9//4):9] = a[:,::4]
    x[(nhoho+nho*9//4):(nhoho+nho*9//4+nHydro*9):9,nhoho+1:(nhoho+nho*9//4):9] = a[:,1::4]
    x[(nhoho+nho*9//4):(nhoho+nho*9//4+nHydro*9):9,nhoho+2:(nhoho+nho*9//4):9] = a[:,2::4]
    x[(nhoho+nho*9//4):(nhoho+nho*9//4+nHydro*9):9,nhoho+3:(nhoho+nho*9//4):9] = a[:,3::4]


    
    ##SS
    x[(nhoho+nho*9//4):(nhoho+nho*9//4+nHydro*9):9,(nhoho+nho*9//4):(nhoho+nho*9//4+nHydro*9):9] = x0[(nhoho+nho):(nhoho+nho+nHydro),(nhoho+nho):(nhoho+nho+nHydro)]  


 

    return x

def packd(x, nSuperHeavy, nHeavy, nHydro):
    nho = 4*nHeavy
    nhoho = 9*nSuperHeavy
    if x.dim()==2:
        x0 = packoned(x, nSuperHeavy*9, nHeavy*4, nHydro, nhoho+nho+nHydro)
    elif x.dim()==4:
        norb = torch.max(nho+nHydro+nhoho)
        x = x.flatten(start_dim=0, end_dim=1)
        x0 = torch.stack(list(map(lambda a, b, c, d : packoned(a, b, c, d,  norb), x, nhoho, nho, nHydro)))
    else:
        norb = torch.max(nho+nHydro+nhoho)
        x0 = torch.stack(list(map(lambda a, b, c, d : packoned(a, b, c, d,  norb), x, nhoho, nho, nHydro)))

    return x0


def unpackd(x0, nSuperHeavy, nHeavy, nHydro, size):

    if x0.dim()==2:
        x = unpackoned(x0, nSuperHeavy*9, nHeavy*4, nHydro, size)
    else:
        nho = 4*nHeavy
        nhoho = 9*nSuperHeavy
        x = torch.stack(list(map(lambda a, b, c, d : unpackoned(a, b, c, d,  size), x0, nhoho, nho, nHydro)))
    return x
