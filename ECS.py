#!/usr/bin/env python
# coding: utf-8

# In[27]:


## Code to calculate the maximal CHSH bound for n-MZI+ECS setting.

import numpy as np
from scipy.optimize import minimize
from _pickle import dump, load
import matplotlib.pyplot as plt
from operator import itemgetter
from time import time
import math
import os

####### Internal functions in the CHSH inequality #######

def coh_state_prod(x,y):
	"""
	For displacements x,y returns the inner product of related coderent states: \bra{x}\ket{y}
	""" 
	return np.exp(-np.abs(x-y)**2/2+1j*(x.conjugate()*y).imag)
	
def one_party_expv(x,y,b):
	"""
	For displacements x,y,b returns: \bra{x}(I - 2\ket{x}\bra{b})\ket{y}
	"""  
	return coh_state_prod(x,y) - 2*coh_state_prod(x,b)*coh_state_prod(b,y)

# Function $f_{i,j}=\bra{\Psi_E}A(x_i) \otimes A(y_i)\ket{\Psi_E}$ in manuscript
def two_party_expv(a1,a,b,c,d,b1,b2):
	"""
	For displacements a,b,c,d,b1,b2 and complex amplitude a1, 
	returns an expected value of (I - 2\ket{b1}\bra{b1})\otimes(I - 2\ket{b2}\bra{b2}) 
	in the (unnormalised) pure state (a1 * \ket{a}\otimes\ket{b} + \ket{c}\otimes\ket{d})
	""" 
	return (np.abs(a1)**2 * one_party_expv(a,a,b1)*one_party_expv(c,c,b2) + \
			2*a1*one_party_expv(b,a,b1)*one_party_expv(d,c,b2) + 
			one_party_expv(b,b,b1)*one_party_expv(d,d,b2)).real

def norm2(a1,a,b,c,d):
	"""
	Returns the squared norm of the state vector:
	(a1 * \ket{a}\otimes\ket{b} + \ket{c}\otimes\ket{d})
	""" 
	return np.abs(a1)**2 + 2*(a1*coh_state_prod(b,a)*coh_state_prod(d,c)).real + 1

##########   Calculating CHSH bound   ############

def chsh(n):
	"""
	Function for calculating the bound of the 
	CHSH inequality for ECS: <\Psi_E|S|\Psi_E>.
	Parameters: n = number of settings per party
	Inner function is used to provide a second 
	parameter: x = array_like; it takes the random
	initial guess for: 
	i. {beta_i} = beta_A
	ii. {gamma_j} = gamma_B
	iii. alpha, epsilon, eta = a,b,c

	Optimization is performed over x.
	"""
	def inn(x):
		#beta_A = np.zeros(n,dtype=np.clongdouble) #beta_1 = 0
		#gamma_B = np.zeros(n,dtype=np.clongdouble) #gamma_1 = 0
		x = x[:2*n+2]+1j*x[2*n+2:] # Modifying x to have 2n complex numbers
		a1,a = x[:2]
		b = 0j
		c = 0j
		d = a
		beta_A = x[2:n+2]
		gamma_B = x[n+2:]
		
		# Expectation value of S (L.H.S. of CHSH inequality) for |Psi_E>
		chsh_val = (sum([two_party_expv(a1,a,b,c,d,beta_A[i],gamma_B[i]) + two_party_expv(a1,a,b,c,d,beta_A[i],gamma_B[(i+1)%n]) for i in range(n)]) \
		-2*two_party_expv(a1,a,b,c,d,beta_A[0],gamma_B[(n-1)]))/norm2(a1,a,b,c,d)
		return -chsh_val # Negative of expected value
	return inn

def max_violation(n):
    """
    Optimization protocol performed on a loop of m times
    for a given n. Stores the minimum value of function 
    obtained from the m optimized values. Parameters stored:
    i. Minimized value of chsh(n)
    ii. Optimal x
    """
    t = time()
    m = 300
    min_res = [minimize(chsh(n),np.random.rand(4*n+4).astype(np.longdouble)) for _ in range(m)] # Can change m=50 to 100,200,etc. to inc. no. 
                                                                                                # of times initial parameters are tossed
    min_res = min(min_res, key = itemgetter('fun'))
    print(n,time()-t,min_res['fun']) # Prints n, time of computation, min chsh(n)
    return min_res


############   Obtaining and storing results   ############

min_n = 2 # Min. n for doing above calculations
max_n = 20 # Max. n for doing above calculations

for n in range(min_n,max_n):
    o = max_violation(n)
    if 'max_chsh_ecs.pi' in os.listdir():  # Stores in filename "max_chsh_ecs.pi"
        with open('max_chsh_ecs.pi','rb') as f:
            x = load(f)
            if n in x and o['fun'] < x[n]['fun'] or n not in x:
            # Stores in file if chsh(n) was not optimized previously 
            # or optimization result is better than previous iteration.
                x[n] = o
                with open('max_chsh_ecs.pi','wb') as f:
                    dump(x,f)
    else:
    # In the first run, generates the file max_chsh_ecs.pi with
    # a initial run of optimization.
        x = {n:o}
        with open('max_chsh_ecs.pi','wb') as f:
            dump(x,f)
