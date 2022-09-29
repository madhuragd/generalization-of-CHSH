#!/usr/bin/env python
# coding: utf-8

# In[11]:


## Code for calculating the eigenvectors corresponding to maximal violation
## from the results obtained using n_MZI.py and stored in file max_chsh_eig.pi


import numpy as np
from numpy.linalg.linalg import eigvalsh
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os
from _pickle import load, dump

#########  Load previously calculated results for Delta  ##########

path = "C:\\Users\\user\\OneDrive\\Documents\\Confocal_acq\\Test\\"
with open(path+'max_chsh_eig.pi','rb') as f: 
    results = load(f)

res_list = [(k, v) for k, v in results.items()] 
op = [(res_list[i][0],res_list[i][1]['x'].tolist()) for i in range(len(res_list))] # Store results in (n,Delta(n)) form in list

######### Recover eigenvectors corresponding to maximal violation #########
def eigvec_CHSH(n,b):
    """
    Function to recover the eigenvectors for maximal eigenvalue 
    of CHSH matrix from the optimized Deltas stored for various n
    from n_MZI.py. Parameters are n and Delta.
    """
    corr = 3e1
    D = b # Calculated Delta from n_MZI.py for given n
    
    #### Similar to max_viol(n) function in n_MZI.py ####
    gamma = np.array([D*i for i in range(n)])
    beta = np.array([D*i for i in range(n)])
    Ga = np.exp(np.array([[-(abs(i-j)**2 + i*j.conj() - j*i.conj())/2 for j in gamma] for i in gamma]))
    Gb = np.exp(np.array([[-(abs(i-j)**2 + i*j.conj() - j*i.conj())/2 for j in beta] for i in beta]))
    try:
      La = np.linalg.cholesky(Ga)
    except:
      mi=np.abs(np.amin(np.linalg.eigvalsh(Ga)))
      Ga+=corr*mi*np.eye(n)
      La = np.linalg.cholesky(Ga)
    try:
      Lb = np.linalg.cholesky(Gb)
    except:
      mi=np.abs(np.amin(np.linalg.eigvalsh(Gb)))
      Gb+=corr*mi*np.eye(n)
      Lb = np.linalg.cholesky(Gb)
    As = [np.eye(n) - 2*row.reshape(1,n).T.conj() @ row.reshape(1,n) for row in La]
    Bs = [np.eye(n) - 2*row.reshape(1,n).T.conj() @ row.reshape(1,n) for row in Lb]
    CHSH = sum([np.kron(As[i],Bs[i]) + np.kron(As[(i+1)%n],Bs[i]) for i in range(n)])-2*np.kron(As[0],Bs[n-1])
    #######################################################
    
    evall, evec = np.linalg.eig(CHSH) # Eigenvalues and eigenvectors of CHSH matrix
    evect = evec.T 
    eval_max = np.amax(evall) # Max. eigenvalue
    evec_max = evect[np.argmax(evall)] # Eigenvector corresponding to max. eigenvalue
    L_op = np.kron(np.linalg.inv(La.conj().T),np.linalg.inv(Lb.conj().T))
    return n, eval_max, evec_max, L_op @ (evec_max), b # Stores n, max eigenvalue and corresponding eigenvector in orthonormal and $\ket{\beta_i}\otimes\ket{\gamma_i}$ basis


######### Obtaining and storing the eigenvectors ###########

opti_evec = [eigvec_CHSH(op[i][0],op[i][1]) for i in range(len(op))] # Stores the eigvec_CHSH(n,b) for all n calculated from n_MZI.py

with open(path+'max_chsh_states.pi','wb') as f: # Stores opti_evec in max_chsh_states.pi file
    dump(opti_evec,f)

