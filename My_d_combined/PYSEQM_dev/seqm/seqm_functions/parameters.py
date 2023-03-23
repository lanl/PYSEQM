import torch


def params(method='MNDO', elements=[1,6,7,8],
           parameters=['U_ss', 'U_pp', 'zeta_s', 'zeta_p','beta_s', 'beta_p',
                       'g_ss', 'g_sp', 'g_pp', 'g_p2', 'h_sp', 'alpha'],
           root_dir='./params/MOPAC/'):
    """
    load parameters from AM1 PM3 MNDO
    """
    # method=MNDO, AM1, PM3
    # load the parameters taken from MOPAC
    # elements: elements needed, not checking on the type, but > 0 and <= 107
    # parameters: parameter lists
    # root_dir : directory for these parameter files
    fn=root_dir+"parameters_"+method+"_MOPAC.csv"
    #will directly use atomic number as array index
    #elements.sort()
    m=max(elements)
    n=len(parameters)
    p=torch.zeros((m+1,n))
    f=open(fn)
    header=f.readline().strip().replace(' ', '').split(',')
    idx = [header.index(item) for item in parameters]
    for l in f:
        t=l.strip().replace(' ', '').split(',')
        id=int(t[0])
        if id in elements:
            p[id,:] = torch.tensor([float(t[x]) for x in idx])
    f.close()
    return torch.nn.Parameter(p, requires_grad=False)

def PWCCT(method='MNDO', elements=[1,6,7,8],
           parameters=['U_ss', 'U_pp', 'zeta_s', 'zeta_p','beta_s', 'beta_p',
                       'g_ss', 'g_sp', 'g_pp', 'g_p2', 'h_sp', 'alpha'],
           root_dir='./params/MOPAC/'):

    """
    loads the diatomic core-core paramters for PM6
    returns them in q
    """
    #will directly use atomic number as array index just as in the params method
    m=max(elements)
    q=torch.zeros((m+1,m+1))
    p=torch.zeros((m+1,m+1))

    if method == 'PM6':
        fo=root_dir+"PWCCT_"+method+"_MOPAC.csv"
        f=open(fo)
        for l in f:
            t=l.strip().replace(' ', '').split(',')
            id=int(t[0])
            id2=int(t[1])
            if id in elements and id2 in elements:
                q[id,id2] = float(t[2])
                p[id,id2] = float(t[3])
        f.close()
    return  torch.nn.Parameter(q, requires_grad=False),torch.nn.Parameter(p, requires_grad=False)


