# %%
'''
simple script for running prototype Davidson algorithm

'''

import torch
import argparse
from seqm.seqm_functions.excited.davidson_algorithm import davidson
from seqm.seqm_functions.excited.aux_routines import run_seqm_1mol

# %%

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to parse parameters for the given functions.")
    
    parser.add_argument("--xyz", type=str, required=True, help="Input XYZ file (e.g., 'c6h6.xyz')")
    parser.add_argument("--nexc", type=int, required=True, help="Number of excitations")
    parser.add_argument("--keepn", type=int, required=True, help="Number of states to keep")
    parser.add_argument("--vmax", type=int, required=True, help="Maximum number of vectors")
    parser.add_argument("--maxiter", type=int, default=500, help="Maximum iterations (default: 100)")
    parser.add_argument("--tol", type=float, default=6, help="Tolerance (default: 1e-6)")
    
    args = parser.parse_args()
    dtype = torch.float64
    device = torch.device('cpu') # cpu run, shoud be tested on GPU as well 
    mol = run_seqm_1mol(args.xyz, device, dtype) # get a PYSEQM molecule object 
   

    eval, evec = davidson(
        device   = device,
        mol      = mol,
        N_exc    = args.nexc,
        keep_n   = args.keepn,
        n_V_max  = args.vmax,
        max_iter = args.maxiter,
        tol       = 10 ** (-1*(args.tol)))
    

# %%

