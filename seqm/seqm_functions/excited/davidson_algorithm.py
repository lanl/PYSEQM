import torch
import seqm
import sys
from ase.io import read as ase_read

# import general seqm functions from PYSEQM

from seqm.seqm_functions.fock import fock
from seqm.seqm_functions.pack import unpack
import seqm.seqm_functions.pack as pack
from seqm.seqm_functions.constants import Constants
from seqm.Molecule import Molecule
from seqm.ElectronicStructure import Electronic_Structure
from termcolor import colored

# import Davidson-specific functions from excited module

from seqm.seqm_functions.excited import ortho as orthogonalize
from seqm.seqm_functions.excited.hamiltonian import gen_V
from seqm.seqm_functions.excited.hamiltonian import form_cis
import seqm.seqm_functions.excited
from seqm.seqm_functions.excited.orb_transform import mo2ao
from seqm.seqm_functions.excited.ortho import orthogonalize_matrix

#=== TORCH OPTIONS ===

device = torch.device('cpu')
torch.set_default_dtype(torch.float64)
dtype = torch.float64
torch.set_printoptions(precision=5, linewidth=200, sci_mode=False, profile = 'short')



def davidson(device, 
             mol, 
             N_exc, 
             keep_n, 
             n_V_max,  
             max_iter, 
             tol):
    """
    Davidson algorithm for solving eigenvalue problem of large sparse diagonally dominant matrices
    Hamiltonian is not generated or stored explicitly, only matrix-vector products are used on-the fly:
    guess space V should be orthogonalized at each iteration
    M (projection of smaller size) is V.T @ H @ V 
    #! RPA (TDHF) is not implemented yet, non-Hermitian (non-symmetric), requires also left eigenvectors 
    note that notation differes between implementations: V.T x A x V is bAb
    # TODO: 1) check if convergence of e_vals is needed
    # TODO: 2) vectorize and optimize orthogonalization
    # TODO: 3) check if some vectors should be dropped 
    # TODO: 4) eliminate loops 
    # TODO: 5) check if whole M should be regenerated, or only sub-blocks corresponding to new guess vectors
    # TODO: 6) add parameter checker like Krylov dims << N_cis

    Args:
        mol (PYSEQM object): object to hold all qm data from PYSEQM
        N_exc (int)        : number of excited states to calculate
        keep_n (int)       : number of e_vals, e_vecs to keep at each iteration
        n_V_max (int)      : maximum size of Krylov subspace, 
                             projected matrix will be no more than M(n_V_max x n_V_max)
        max_iter (int)     : maximum number of iterations in Davidson
        tol (float)        : treshold for residual
        
    Returns:
        tuple of tensors: eigenvalues (excitation energies in default units, eV) and eigenvectors 
    """    
    
    n_V_start = N_exc * 2 # dimension of Krylov subspace, analogue of nd1  
    N_cis = mol.nocc * mol.nvirt # size of CIS space
    
    assert n_V_max >= 4 * N_exc, 'n_V_max should be at least 4x N_exc or 2x n_V_start | n_V_start = ' + str(n_V_start) + ' '
    
    N_rpa = 2 * N_cis # size of RPA problem; RPA is not implemented yet but should be very easy
    term = False  # terminate algorithm, will be set to true after convergence
    iter = 0
    L_xi = torch.zeros((N_cis, n_V_start), device=device) 
    
    #======== Krylov iterations ========
    V = gen_V(device, mol, N_cis, n_V_start) # generate initial guess, V here #! should be renamed
    diag = None # create diagonal of M only once
    
    while iter < max_iter and not term: # Davidson loop
        
        if iter > 0: # skip first step, as initial V is orthonormal
            V = orthogonalize_matrix(V)
            
        print('---------------------------------', flush=True)
        print(colored(f' ITERATION : {iter} ', 'red', 'on_white', attrs=['bold']), flush=True)
        print(f'subspace size V = {V.shape[0]} x {V.shape[1]}', flush=True)
        print('---------------------------------', flush=True)
       
        # ---------- form A x b product --------------------
        L_xi = torch.zeros((N_cis, V.shape[1]), device=device) #! preallocate

        for i in range(V.shape[1]):   # column x guess_vector opearations 
            L_xi[:,i] = form_cis(device, V[:,i], mol, N_cis, N_rpa) # TODO: should be vectorized  

    
        right_V = L_xi # (A)b in common notations ; nexmd duplicates size for RPA problem 
        
        # ---------- form b.T x Ab product --------------------
        
        M =  V.T @ right_V
        if iter == 0:
            diag = torch.diag(M) # create diagonal only once

        iter += 1
        
        # ---------- diagonalize projection M --------------------
        r_eval, r_evec = torch.linalg.eigh(M) # find eigenvalues and eigenvectors
        r_eval = r_eval.real
        r_evec = r_evec.real
        r_eval, r_idx = torch.sort(r_eval, descending=False) # sort eigenvalues in ascending order
        r_evec = r_evec[:, r_idx] # sort eigenvectors accordingly
        e_val_n = r_eval[:keep_n] # keep only the lowest keep_n eigenvalues; full are still stored as e_val
        e_vec_n = r_evec[:, :keep_n]
        
        print(colored('Eigenvalues: ', 'green', attrs=['bold']) + ', '.join(['{:.4}'.format(x) for x in e_val_n]), flush=True)
    
        resids = torch.zeros(V.shape[0], len(e_val_n)) # account for right e_vecs

        # ---------- calculate residual vectors --------------------
        #TODO: vectorize residuals calculationsl PROBABLY HIGHLY INEFFICIENT with loops
        for j in range(len(e_val_n)): # calc residuals 
            resids[:,j] = right_V @ e_vec_n[:,j] - e_val_n[j] * (V @ e_vec_n[:,j])
            
        resids_norms_r = torch.tensor([resids[:,x].norm() for x in range(resids.shape[1])]) # get norms of all resids

        # ---------- expand guess space V by non-converged resids --------------------
        # TODO: vectorize 
        if torch.any(resids_norms_r > tol):
            mask_r = resids_norms_r >= tol
            large_res_r = resids[:,mask_r] # residuals larger than tol argument      
            large_res_r.to(device)
            cor_e_val_r = e_val_n[mask_r] # corresponding eigenvalues
            print('# of large resids: ', large_res_r.shape[1], flush=True)
            print('norms: ', ' '.join(["{:.2e}".format(x) for x in resids_norms_r[mask_r]]), flush=True)
            # ------keep adding new resids till max V size--------------------
            if V.shape[1] <= n_V_max:     

                    for j in range(large_res_r.shape[1]):
                        if V.shape[1] <= n_V_max:
                            s = large_res_r[:,j] # conditioned residuals > tol

                            if s.norm() >= tol:
                                denom = (diag[j] - cor_e_val_r[j])
                                denom.to(device) 
                                s = s/denom # conditioned residuals
                                s.to(device)
                                V = torch.column_stack((V, s/s.norm()))
                            else:
                                pass
            # ------ collapse (restart) if space V is too large; mix eigenvectors with V------------
            else:
                print('*** guess space V expansion reached limit, restarting with n_V_start = ', n_V_start, flush=True)
                V =  V @ r_evec[:, :n_V_start]
                continue

        else:
            term = True
            print('\n========================', flush=True)
            print('*** all residuals are below tolerance ***')
            print(colored('Davidson algorithm CONVERGED successfully', 'green', attrs=['bold']), flush=True)
            print(colored('Final eigenvalues (excitation energies) in eV:\n', 'green', attrs=['bold']), flush=True)     
            print(*["{:.4}".format(x) for x in r_eval[:keep_n]], sep='\n', flush=True)         
            print('\n==========================', flush=True)

            return r_eval, r_evec

    # runs after big loop if did not converge
    print('============================')
    print(colored('Davidson algorithm DID NOT converge', 'red', attrs=['bold']), flush=True)
    print(colored('Final eigenvalues (excitation energies) in eV:\n', 'red', attrs=['bold']), flush=True)
    print(*["{:.4}".format(x) for x in r_eval[:keep_n]], sep='\n', flush=True)
    print('\ntry:')
    print(' 1. increasing number of steps (max_iter) or tolerance (tol)')
    print(' 2. increasing initial guess space size (n_V_start) or max subspace size (n_V_max)')
    print(' if # of residuals and their norms are going down, you are on track\n')
    print('*** If nothing helps, contact developers as it is module under development ***')
        

    return r_eval, r_evec