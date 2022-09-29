#!/usr/bin/env python
# coding: utf-8

# In[5]:


## Code to calculate and plot the Von Neumann entropies vs. number of settings

import matplotlib.pyplot as plt
import numpy as np
from _pickle import load
import math

###### Retrieving calculated values from relevant file #########

path = "C:\\Users\\user\\OneDrive\\Documents\\Confocal_acq\\Test\\"
with open(path+'max_chsh_states.pi','rb') as f:
     opti_evec= load(f)

##### Calculating (nxn) matrices of eigenvectors in orthonormal and
##### $\ket{\beta_i}\otimes\ket{\gamma_i}$ bases.

mat_delta = [d[3].reshape(d[0],d[0]) for d in opti_evec] # In $\ket{\beta_i}\otimes\ket{\gamma_i}$ basis
mat_gamma = [d[2].reshape(d[0],d[0]) for d in opti_evec] # In orthonormal basis
mat_svd = [np.linalg.svd(d) for d in mat_gamma] # Singular value decomposition of eigenvectors in orthonormal basis

def VN_entro(x):
    """
    Function for calculating Von Neumann entropies 
    from the Schmidt Coefficients calculated via
    Singular Value Decomposition (above). 
    """
    scoeff = x
    scoeff_sq = [l*l.conjugate() for l in scoeff] # Taking absolute squared values of Schmidt Coefficients
    S_vn = -sum(l*math.log2(l) for l in scoeff_sq) # Von Neumann Entropy
    return S_vn

VN = [(m+2,VN_entro(mat_svd[m][1])) for m in range(len(mat_svd))] # Stores Von Neumann entropies in a list
NormVN = [(m+2,math.log2(m+2)) for m in range(len(mat_svd))] # Normalization factor: log_2(n) stored in a list

#####################     Plot Details    ##########################

plt.plot(*zip(*VN[:18]),'r.', *zip(*NormVN[:18]),'b*')

plt.legend(["$\mathrm{S}_{n}(\lambda)$","$log_2$(n)"], loc = "upper left", fontsize=12)#bbox_to_anchor=(0.8, 0.5))
plt.xlabel(r"$n$",fontsize=14)
plt.ylabel(r"$\mathrm{S}_{n}(\lambda)$",fontsize=14)
plt.title(r'$\mathrm{S}_{n}(\lambda)$ vs. $n$',fontsize=15)
plt.xticks(np.arange(2,21, step=2))
plt.minorticks_on()

plt.savefig(path+'VN_entropies.pdf', format='pdf', bbox_inches="tight")

