#!/usr/bin/env python
# coding: utf-8

# In[14]:


## Code to calculate the maximal CHSH bound for n-MZI+TMSV setting.

import numpy as np
from scipy.optimize import minimize
from _pickle import dump
import matplotlib.pyplot as plt
from operator import itemgetter
from time import time
import math
import os

####### Internal functions in the CHSH inequality #######

f1 = lambda r,x: np.exp(-abs(x)**2)*np.exp(np.tanh(r)**2)**(np.abs(x)**2)
f2 = lambda r,t,x,y: np.exp(-abs(x)**2-abs(y)**2)*np.exp(-2*(np.exp(1j*t)*x*y).real*np.tanh(r))
# Function $g_{i,j}=\bra{\Psi_{TMSV}}A(x_i) \otimes A(y_i)\ket{\Psi_{TMSV}}$ in manuscript
f3 = lambda r,t,x,y: 1+(-2*f1(r,x)-2*f1(r,y)+4*f2(r,t,x,y))/np.cosh(r)**2

##########   Calculating CHSH bound   ############

def CHSH_sq(n):
    """
    Function for calculating the bound of the CHSH 
    inequality for TMSV: <\Psi_{TMSV}|S|\Psi_{TMSV}>.
    Parameters: n = number of settings per party
    Inner function is used to provide a second 
    parameter: x = array_like; it takes the random
    initial guess for: 
    i. {beta_i} = beta
    ii. {gamma_j} = gamma
    iii. r, theta = r,t
    
    Optimization is performed over x.
    """
    def inn_fn(x):
        beta = np.zeros(n,dtype=np.clongdouble) #beta_1 = 0
        gamma = np.zeros(n,dtype=np.clongdouble) #gamma_1 = 0
        r = x[0]
        t = 0
        x1 = x[1:]#+1j*x[2*n:]
        beta[1:] = x1[:n-1]
        gamma[1:] = x1[n-1:]
        
        # Expectation value of S (L.H.S. of CHSH inequality) for |Psi_{TMSV}>
        chsh_val = (sum([f3(r,t,beta[i],gamma[i]) for i in range(n)]) +                     sum([f3(r,t,beta[i],gamma[(i+1)%n]) for i in range(n)]) -                     2*f3(r,t,beta[0],gamma[n-1]))
        return -np.abs(chsh_val) # Negative of expected value
    return inn_fn

def max_violation(n):
    """
    Optimization protocol performed on a loop of m times
    for a given n. Stores the minimum value of function 
    obtained from the m optimized values. Parameters stored:
    i. Minimized value of CHSH_sq(n)
    ii. Optimal x
    """
    t=time()
    m = 50 # No. of times to optimize function.
    min_res = [minimize(CHSH_sq(n),np.random.rand(2*n-1).astype(np.longdouble)) for _ in range(m)] # Can change m=50 to 100,200,etc. to inc. no. 
                                                                                                  # of times initial parameters are tossed
    print(n,t-time(),min(i['fun'] for i in min_res)) # Prints n, time of computation, min CHSH_sq(n)
    return min(min_res, key = itemgetter('fun'))


###########   Obtaining and storing results   ############

min_n = 2 # Min. n for doing above calculations
max_n = 4 # Max. n for doing above calculations


for n in range(min_n,max_n):
    o = max_violation(n)
    if 'max_chsh_tmsv.pi' in os.listdir():  # Stores in filename "max_chsh_tmsv.pi"
        with open('max_chsh_tmsv.pi','rb') as f:
            x = load(f)
            if n in x and o['fun'] < x[n]['fun'] or n not in x:
            # Stores in file if CHSH_sq(n) was not optimized previously 
            # or optimization result is better than previous iteration.
                x[n] = o
                with open('max_chsh_tmsv.pi','wb') as f:
                    dump(x,f)
    else:
    # In the first run, generates the file max_chsh_tmsv.pi with
    # a initial run of optimization.
        x = {n:o}
        with open('max_chsh_tmsv.pi','wb') as f:
            dump(x,f)
