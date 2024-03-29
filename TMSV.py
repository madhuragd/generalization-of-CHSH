#!/usr/bin/env python
# coding: utf-8

## Code to calculate the maximal CHSH bound for n-MZI+TMSV setting.

import numpy as np
from scipy.optimize import minimize
from _pickle import dump
import matplotlib.pyplot as plt
from operator import itemgetter
from time import time
import math
import os
from util import pickle_results

####### Internal functions in the CHSH inequality #######

f1 = lambda r,x: np.exp(-abs(x)**2/np.cosh(r)**2)
f2 = lambda r,x,y: np.exp(-abs(x)**2-abs(y)**2-2*(x*y).real*np.tanh(r))
# Function $g_{r,i,j}=\bra{\Psi_{TMSV(r)}}A(\beta_i) \otimes A(\gamma_i)\ket{\Psi_{TMSV}(r)}$ in manuscript
f3 = lambda r,x,y: 1+(-2*f1(r,x)-2*f1(r,y)+4*f2(r,x,y))/np.cosh(r)**2

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
        x1 = x[1:]#+1j*x[2*n:]
        beta[1:] = x1[:n-1]
        gamma[1:] = x1[n-1:]
        
        # Expectation value of S (L.H.S. of CHSH inequality) for |Psi_{TMSV}>
        chsh_val = (sum([f3(r,beta[i],gamma[i]) for i in range(n)]) + sum([f3(r,beta[i],gamma[(i+1)%n]) for i in range(n)]) - 2*f3(r,beta[0],gamma[n-1]))
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
    m = 100 # No. of times to optimize function.
    min_res = [minimize(CHSH_sq(n),np.random.rand(2*n-1).astype(np.longdouble)) for _ in range(m)] # Can change m=50 to 100,200,etc. to inc. no. 
                                                                                                  # of times initial parameters are tossed
    print(n,t-time(),min(i['fun'] for i in min_res)) # Prints n, time of computation, min CHSH_sq(n)
    return min(min_res, key = itemgetter('fun'))


###########   Obtaining and storing results   ############

min_n = 2 # Min. n for doing above calculations
max_n = 20 # Max. n for doing above calculations

if __name__ == "__main__":

	pickle_results('max_viol_tmsv.pi', max_violation, range(min_n,max_n))
    
    
