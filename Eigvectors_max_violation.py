#!/usr/bin/env python
# coding: utf-8

# In[11]:


## Code for calculating the eigenvectors corresponding to maximal violation
## from the results obtained using Eig_second_stage.py and stored in file Eig_second_stage.pi

import numpy as np
from numpy.linalg.linalg import eigvalsh
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os
from _pickle import load, dump
from util import LHS_of_BCCB


with open('Eig_second_stage.pi','rb') as f: 
    results = load(f)

res_list = [(k, v) for k, v in results.items()] 
op = [(i[0],i[1].x) for i in res_list] # Store results in (n,Delta(n)) form in l

######### Recover eigenvectors corresponding to maximal violation #########

def genL(n,x): # Generates Cholesky decomposition and returns matrix L
    corr = 3e0
    G = np.exp(np.array([[-(abs(i-j)**2 + i*j.conj() - j*i.conj())/2 for j in x] for i in x]))
    try:
        L = np.linalg.cholesky(G)
    except:
        mi=np.abs(np.amin(np.linalg.eigvalsh(G)))
        G+=corr*mi*np.eye(n)
        L = np.linalg.cholesky(G)
    return L
    


def eigvec_BCCB(n,b):
    """
    Function to recover the eigenvectors for maximal eigenvalue 
    of CHSH matrix from the optimized parameters stored for various n
    from Eig_second_stage.py. Parameters are n and x.
    """
    
    gamma = np.zeros(n)
    beta = np.zeros(n)
    if n<10:
        gamma[1:] = b[:n-1]
        beta[1:] = b[n-1:]
    else:
        gamma = b[:n]
        beta = b[n:]
    BCCB = LHS_of_BCCB(beta,gamma)
    
    
    #######################################################
    
    evall, evec = np.linalg.eig(BCCB) # Eigenvalues and eigenvectors of CHSH matrix
    evect = evec.T 
    eval_max = np.amax(evall) # Max. eigenvalue
    evec_max = evect[np.argmax(evall)] # Eigenvector corresponding to max. eigenvalue
    L_op = np.kron(np.linalg.inv(genL(n,gamma).conj().T),np.linalg.inv(genL(n,beta).conj().T))
    return n, eval_max, evec_max, L_op @ (evec_max), b # Stores n, max eigenvalue and corresponding eigenvector in orthonormal and $\ket{\beta_i}\otimes\ket{\gamma_i}$ basis


######### Obtaining and storing the eigenvectors ###########

opti_evec = [eigvec_BCCB(op[i][0],op[i][1]) for i in range(len(op))] # Stores the eigvec_CHSH(n,b) for all n calculated from n_MZI.py

with open('test_max_viol_states.pi','wb') as f: # Stores opti_evec in max_chsh_states.pi file
    dump(opti_evec,f)


