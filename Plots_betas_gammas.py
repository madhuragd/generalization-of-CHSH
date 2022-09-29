#!/usr/bin/env python
# coding: utf-8

# In[3]:


## Initial code to calculate max. bound of CHSH for n-MZI settings.
## Can be used to check how betas and gammas behave in the complex plane,
## for max. bound of n-MZI settings.

import numpy as np
from numpy.linalg.linalg import eigvalsh
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os
from time import time 

def max_viol(n):
  """
  Initial version of function to calculate max. 
  bound of CHSH inequality for n-MZI settings. 
  Parameters: n (number of settings).
  Inner function: inner_fn(x) takes 'x' as parameter
  for optimization. x: array_like; having initial 
  guesses for real and imaginary parts of gamma_i,
  beta_i (i = 1 to n). Dim of x = 4n.
  
  """
  def inner_fn(x):
    gamma = x[:n] + 1j * x[n:2*n]
    beta = x[2*n:3*n] + 1j * x[3*n:]
    Ga = np.exp(np.array([[-(abs(i-j)**2 + i*j.conj() - j*i.conj())/2 for j in gamma] for i in gamma]))
    Gb = np.exp(np.array([[-(abs(i-j)**2 + i*j.conj() - j*i.conj())/2 for j in beta] for i in beta]))
    try:
      La = np.linalg.cholesky(Ga)
      Lb = np.linalg.cholesky(Gb)
      As = [np.eye(n) - 2*row.reshape(1,n).T.conj() @ row.reshape(1,n) for row in La]
      Bs = [np.eye(n) - 2*row.reshape(1,n).T.conj() @ row.reshape(1,n) for row in Lb]
      CHSH = sum([np.kron(As[i],Bs[i]) + np.kron(As[(i+1)%n],Bs[i]) for i in range(n)])-2*np.kron(As[0],Bs[n-1])
      return np.amin(np.linalg.eigvalsh(CHSH))
    except:
      return 0
  return inner_fn

def optimizeCHSH(n):
    """
    Optimization protocol for max_viol(n).
    Repeats minimization for m times and
    selects the minimum of the function 
    from the m results. Returns n, max.
    function (negative of min.) and array x
    for which optimal value of function
    is achieved.
    """
    m = 200
    t = time()  
    min_res = [minimize(max_viol(n),np.random.randn(4*n)) for _ in range(m)]
    print(n, np.abs(t-time()))
    min_fun = [m['fun'] for m in min_res]
    return n, -min(min_fun), min_res[min_fun.index(min(min_fun))].x 



#####################     Plot Details    ##########################   
fig = plt.figure(figsize=(16,4)) 
axs = [fig.add_subplot(1,3,i+1) for i in range(3)]
fig.subplots_adjust(wspace=0.45)

d = [optimizeCHSH(n) for n in range(3,6)] # Calculates max. bounds for n in [3,6)

for i in range(len(d)):
  n = d[i][0]
  x = d[i][2]
    
## Retrieving betas and gammas from optimized x
  gamma = x[:n] + 1j * x[n:2*n]
  beta = x[2*n:3*n] + 1j * x[3*n:]

## Separating real and imaginary parts for betas and gammas
  x1 = [i.real for i in gamma]
  y1 = [i.imag for i in gamma]
  x2 = [i.real for i in beta]
  y2 = [i.imag for i in beta]

  axs[i].scatter(x1,y1,label=r'$\beta_i$')
  axs[i].scatter(x2,y2,label=r'$\gamma_i$')
  axs[i].set_title(r'$n$ = %s' %n, fontsize = 15)
  axs[i].set_xlabel("Real Axis", fontsize=14)
  axs[i].set_ylabel("Imaginary Axis", fontsize=14)
  plt.legend()
  axs[len(d)-1].legend(bbox_to_anchor=(1.5, 1.02))

# plt.text(-5.15, 1.05, '(a)', fontsize=14)
# plt.text(-3.21, 1.05, '(b)', fontsize=14)
# plt.text(-1.21, 1.05, '(c)', fontsize=14)

path = "C:\\Users\\user\\OneDrive\\Documents\\Confocal_acq\\Test\\"
plt.savefig(path+'complex_plane_beta_gamma.pdf', format='pdf', bbox_inches="tight") # Saves figure

