import torch

# SP2 convergence thresholds for different precisions
SP2_EPS_FLOAT32 = 1.0e-2  # float32 is harder to converge, use relaxed threshold
SP2_EPS_FLOAT64_MAX = 1.0e-3
SP2_EPS_FLOAT64_MIN = 1.0e-7

def SP2(a, nocc, eps=1.0e-4, factor=2.0):
    #print(a.shape)
    # a: batch of fock matrixes, don't need to be truncated
    # noccd: number of occupied MO
    #return a0: denisty matrixes wich commute with a
    # factor = 1.0 or 2.0, return a0, tr(a0)= factor*nocc
    device = a.device
    dtype = a.dtype
    flag = dtype==torch.float32
    if flag:
        #float32, harder to converge to a smaller eps, for this one set eps=1.0e-2, and break when no more improvement
        #use critiria to check there is no more improvement
        if eps < SP2_EPS_FLOAT32:
            eps = SP2_EPS_FLOAT32
    else:
        #float64, if use above critiria, the error will keep going down and take lots of iteration to reach no more improvement
        #so put eps as a small one like 1.0e-4, to recude the number of iterations
        #use critiria to check the err of current and last iterations both <= eps
        if eps > SP2_EPS_FLOAT64_MAX:
            eps = SP2_EPS_FLOAT64_MAX
        elif eps < SP2_EPS_FLOAT64_MIN:
            eps = SP2_EPS_FLOAT64_MIN
    noccd = nocc.type(dtype)

    N, D, _ = a.shape
    #Gershgorin circle theorem estimate
    ###maximal and minimal eigenvalues
    aii = a.diagonal(dim1=1,dim2=2)
    ri = torch.sum(torch.abs(a),dim=2)-torch.abs(aii)
    h1 = torch.min(aii-ri,dim=1)[0]
    hN = torch.max(aii+ri,dim=1)[0]
    #scale a
    a0 = (torch.eye(D,dtype=dtype,device=device).unsqueeze(0).expand(N,D,D)*hN.reshape(-1,1,1)-a)/(hN-h1).reshape(-1,1,1)

    #error from current iteration
    errm0=torch.abs(torch.sum(a0.diagonal(dim1=1,dim2=2),dim=1)-noccd)
    errm1=errm0.clone() #error from last iteration
    errm2=errm1.clone() #error from last to second iteration

    notconverged = torch.ones(N,dtype=torch.bool,device=device)
    a2 = torch.zeros_like(a)
    cond = torch.zeros_like(notconverged)
    k=0
    while notconverged.any():
        a2[notconverged] = a0[notconverged].matmul(a0[notconverged]) #batch supported

        tr_a2 = torch.sum(a2[notconverged].diagonal(dim1=1,dim2=2),dim=1)
        cond[notconverged] = torch.abs(tr_a2-noccd[notconverged]) < \
                             torch.abs(2.0*torch.sum(a0[notconverged].diagonal(dim1=1,dim2=2),dim=1) - tr_a2 - noccd[notconverged])
        cond1 = notconverged * cond
        cond2 = notconverged * (~cond)
        a0[cond1] = a2[cond1]
        a0[cond2] = 2.0*a0[cond2]-a2[cond2]
        errm2[notconverged] = errm1[notconverged]
        errm1[notconverged] = errm0[notconverged]
        errm0[notconverged] = torch.abs(torch.sum(a0[notconverged].diagonal(dim1=1,dim2=2),dim=1)-noccd[notconverged])
        k+=1
        #"""
        #print('SP2', k,' '.join([str(x) for x in errm0.tolist()]))
        #print(' '.join([str(x) for x in torch.symeig(a0)[0][0].tolist()]))
        #"""
        if flag:
            #float32, harder to converge to a smaller eps, for this one set eps=1.0e-2, and break when no more improvement
            notconverged[notconverged.clone()] = ~((errm0[notconverged] < eps) * (errm0[notconverged] >= errm2[notconverged]))
        else:
            #float64, if use above critiria, the error will keep going down and take lots of iteration to reach no more improvement
            #so put eps as a small one like 1.0e-4, to recude the number of iterations
            notconverged[notconverged.clone()] = ~((errm0[notconverged] < eps) * (errm1[notconverged] < eps))

    return factor*a0
