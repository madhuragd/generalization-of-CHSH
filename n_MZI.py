#!/usr/bin/env python
# coding: utf-8

# In[6]:


## Code to calculate the maximal CHSH bound for n-MZI settings.

import numpy as np
from numpy.linalg.linalg import eigvalsh
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os
from _pickle import dump
from operator import itemgetter
from time import time

##########   Calculating CHSH bound   ############

def max_viol(n):
    """
    Function for calculating the bound of CHSH 
    inequality for general n-MZI settings.
    Parameters: n = number of settings per party
    Inner function is used to provide a second 
    parameter: x = number; takes the initial
    random guess for Delta (difference between
    numbers in arithmetic sequence formed by 
    each beta_i and gamma_i).
    Optimization is performed over x.
    """
    def inner_fn(x):
        D = x[0]  # Delta = beta_{i+1} - beta_i = gamma_{i+1} - gamma_i
        gamma = np.array([D*i for i in range(n)]) # gamma_1 = 0
        beta = np.array([D*i for i in range(n)])  # beta_1 = 0
        Ga = np.exp(np.array([[-(abs(i-j)**2 + i*j.conj() - j*i.conj())/2 for j in gamma] for i in gamma])) # Gram matrix for gammas
        Gb = np.exp(np.array([[-(abs(i-j)**2 + i*j.conj() - j*i.conj())/2 for j in beta] for i in beta]))   # Gram matrix for betas
        # Cholesky Decomposition for Gram matrices of betas and gammas:
        try:
            try:
                La = np.linalg.cholesky(Ga) 
            except:
                mi=np.abs(np.amin(np.linalg.eigvalsh(Ga)))
                Ga+=corr*mi*np.eye(n) # Correction factor added to La. Required only when matrix is not semi-positive definite due to numerical inaccuracy 
                                      # In some cases very-small nonzero entries are present in place of zero. Correction factor rectifies this problem. 
                La = np.linalg.cholesky(Ga)
            try:
                Lb = np.linalg.cholesky(Gb)
            except:
                mi=np.abs(np.amin(np.linalg.eigvalsh(Gb)))
                Gb+=corr*mi*np.eye(n) # Same argument as for correction factor for La.
                Lb = np.linalg.cholesky(Gb)
                
            # Observables (As and Bs) for each party in orthonormal basis given by columns of La and Lb:
            As = [np.eye(n) - 2*row.reshape(1,n).T.conj() @ row.reshape(1,n) for row in La]
            Bs = [np.eye(n) - 2*row.reshape(1,n).T.conj() @ row.reshape(1,n) for row in Lb]
            
            # CHSH matrix
            CHSH = sum([np.kron(As[i],Bs[i]) + np.kron(As[(i+1)%n],Bs[i]) for i in range(n)])-2*np.kron(As[0],Bs[n-1])
            return np.amin(np.linalg.eigvalsh(CHSH)) # Min. eigenvalue for CHSH matrix
        except:
            return 0
    return inner_fn

def optimizeCHSH(n):  
    """
    Optimization protocol performed on a loop of m times
    for a given n. Stores the minimum value of function 
    obtained from the m optimized values. Parameters stored:
    i. Minimized value of max_viol(n)
    ii. Optimal x
    """
    t=time()
    m = 5 # No. of times to optimize function.
    min_res = min((minimize(max_viol(n),.09*np.random.random(1),method='Powell',tol=1e-15) for _ in range(m)), key=itemgetter('fun'))
    print(n, time()-t, min_res['fun'])
    return min_res

###########   Obtaining and storing results   ############

corr = 3e0 # Correction factor
Min = 2 # Min. n for doing above calculations
Max = 3 # Max. n for doing above calculations

if __name__=="__main__":
    for n in range(Min,Max):
        o = optimizeCHSH(n)
        if 'max_chsh_eig.pi' in os.listdir(): # Stores in filename "max_chsh_eig.pi"
            with open('max_chsh_eig.pi','rb') as f:
                d = load(f)
                if n in d and o['fun'] < d[n]['fun'] or n not in d:
                # Stores in file if max_viol(n) was not optimized previously 
                # or optimization result is better than previous iteration.
                    d[n] = o
                    with open('max_chsh_eig.pi','wb') as f:
                        dump(d,f)
        else:
        # In the first run, generates the file max_chsh_eig.pi with
        # a initial run of optimization.
            d = {n:o}
            with open('max_chsh_eig.pi','wb') as f:
                dump(d,f)

