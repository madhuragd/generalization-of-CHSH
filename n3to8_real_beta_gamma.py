#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## Initial code to calculate max. bound of CHSH for n-MZI settings.
## Can be used to check how betas and gammas behave in the complex plane,
## for max. bound of n-MZI settings.

import numpy as np
from numpy.linalg.linalg import eigvalsh
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os
from time import time 
from operator import itemgetter
from _pickle import load, dump

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
    gamma = np.zeros(n,dtype=np.clongdouble)
    beta = np.zeros(n,dtype=np.clongdouble)
    gamma[1:] = x[:n-1] #+ 1j * x[n:2*n]
    beta[1:] = x[n-1:] #+ 1j * x[3*n:]
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
    t = time()
    min_res = [minimize(max_viol(n),x0) for _ in range(100)] #np.random.randn(4*n)
    print(n,t-time(),min(i['fun'] for i in min_res))
    # min_fun = [m['fun'] for m in min_res]
    return min(min_res, key = itemgetter('fun'))


# d = [optimizeCHSH(n) for n in range(3,8)]
## Comment above line and then run to plot from file which gathers previously compiled data

#####################     Plot Details    ########################## 
with open('n3to8_real.pi','rb') as f:
    results = load(f)
    res_list = [(k, v) for k, v in results.items()]
    
res_list[6]
fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(16, 8))
fig.subplots_adjust(hspace=0.3,wspace=0.3)
for i in range(1,len(res_list)):
  n = res_list[i][0]
  x = res_list[i][1].x

    ## Retrieving betas and gammas from optimized x
  gamma = np.zeros(n,dtype=np.clongdouble)
  beta = np.zeros(n,dtype=np.clongdouble)
  gamma[1:] = x[:n-1] #+ 1j * x[n:2*n]
  beta[1:] = x[n-1:] #+ 1j * x[3*n:]

## Separating real and imaginary parts for betas and gammas
  x1 = [i.real for i in gamma]
  y1 = [i.imag for i in gamma]
  x2 = [i.real for i in beta]
  y2 = [i.imag for i in beta]

  if i < 4:
      axs[0][i-1].scatter(x1,y1,label=r'$\beta_i$')
      axs[0][i-1].scatter(x2,y2,label=r'$\gamma_i$')
      axs[0][i-1].set_title(r'$n$ = %s' %n, fontsize = 15)
      axs[0][i-1].set_xlabel("Real Axis", fontsize=14)
      axs[0][i-1].set_ylabel("Imaginary Axis", fontsize=14)
  else:
      axs[1][i-4].scatter(x1,y1,label=r'$\beta_i$')
      axs[1][i-4].scatter(x2,y2,label=r'$\gamma_i$')
      axs[1][i-4].set_title(r'$n$ = %s' %n, fontsize = 15)
      axs[1][i-4].set_xlabel("Real Axis", fontsize=14)
      axs[1][i-4].set_ylabel("Imaginary Axis", fontsize=14)
  plt.legend()
  axs[0][2].legend(bbox_to_anchor=(1, 1.02))
    

# plt.savefig(path+'complex_plane_beta_gamma.pdf', format='pdf', bbox_inches="tight") # Saves figure

