import numpy as np
for m in ['MNDO', 'AM1', 'PM3']:
    fn='parameters_'+m+'_MOPAC.csv'
    with open(fn) as f:
        header=f.readline().strip().replace(" ","").split(',')
        nbetas=header.index('beta_s')
        nbetap=header.index('beta_p')
        nzetas=header.index('zeta_s')
        nzetap=header.index('zeta_p')
        for i in range(1,18):
            t=f.readline().strip().replace(" ","").split(',')
            alist = [float(t[x]) for x in [nbetas, nbetap, nzetas, nzetap]]
            n=int(t[0])
            if n!=i:
                raise ValueError('Something is wrong')
            #beta_s, beta_p, zeta_s, zeta_p
            #parameters['MNDO'][1]=[-6.9890640, 0.0, 1.3319670, 0.0]
            if np.sum(np.abs(np.array(alist)))>0.0:
                print("parameters['%s'][%d]=%s" % (m,n,str(alist)))
