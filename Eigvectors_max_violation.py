#!/usr/bin/env python
# coding: utf-8

## Code for calculating the eigenvectors corresponding to maximal violation
## from the results obtained using Eig_second_stage.py and stored in file Eig_second_stage.pi

import numpy as np
from numpy.linalg.linalg import eigvalsh
from _pickle import load, dump
from util import LHS_of_BCCB, genL

######### Recover eigenvectors corresponding to maximal violation #########

def eigvec_BCCB(n,b):
	"""
    Function to recover the eigenvectors for maximal eigenvalue 
    of CHSH matrix from the optimized parameters stored for various n
    from Eig_second_stage.py. Parameters are n and x.
"""
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

if __name__ == "__main__":

	with open('Eig_second_stage.pi','rb') as f: 
		results = load(f)

	op = [(k, v.x) for k, v in results.items()] 
	opti_evec = [eigvec_BCCB(*i) for i in op] # Stores the eigvec_CHSH(n,b) for all n calculated from n_MZI.py

	with open('test_max_viol_states.pi','wb') as f: # Stores opti_evec in max_chsh_states.pi file
		dump(opti_evec,f)


