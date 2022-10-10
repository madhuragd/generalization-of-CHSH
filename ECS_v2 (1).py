#!/usr/bin/env python
# coding: utf-8

# In[16]:


## Code to calculate the maximal CHSH bound for n-MZI+ECS setting.

import numpy as np
from scipy.optimize import minimize
from operator import itemgetter
from time import time
import os
from _pickle import dump
from _pickle import load

####### Internal functions in the CHSH inequality #######

coh_state_prod = lambda x,y: np.exp(-np.abs(x-y)**2/2+1j*(x.conjugate()*y).imag)  # Inner product of two coherent states <x|y>
term1 = lambda x,b,g: (1-2*np.exp(-np.abs(x-b)**2))*(1-2*np.exp(-np.abs(g)**2))
term2 = lambda x,b: np.exp(-np.abs(x)**2/2)-2*np.exp(-np.abs(b)**2/2)*coh_state_prod(b,x)

tot_term = lambda a1,a,b,g: np.abs(a1)**2*term1(a,b,g)+2*(a1*term2(a,b)*term2(a,g)).real+term1(a,g,b)
normf = lambda a1,a: 1+np.abs(a1)**2+2*a1.real*np.exp(-np.abs(a)**2)

##########   Calculating CHSH bound   ############
def chsh(n):
#     Function for calculating the bound of the 
#     CHSH inequality for ECS: <\Psi_{ECS}|S|\Psi_{ECS}>.
#     Parameters: n = number of settings per party
#     Inner function is used to provide a second 
#     parameter: x = array_like; it takes the random
#     initial guess for: 
#     i. {beta_i} = beta_A
#     ii. {gamma_j} = gamma_B
#     iii. alpha, a1 = a, a1
    
#     Optimization is performed over x.
    def inn(x):
        beta_A = np.zeros(n,dtype=np.clongdouble) #beta_1 = 0
        beta_B = np.zeros(n,dtype=np.clongdouble) #gamma_1 = 0
        x = x[:2*n]+1j*x[2*n:] # Modifying x to have 2n+2 complex numbers
        a1,a = x[:2]
        beta_A[1:] = x[2:n+1]
        beta_B[1:] = x[n+1:]
        # Expectation value of S (L.H.S. of CHSH inequality) for |Psi_{ECS}>
        chsh_val = (sum([tot_term(a1,a,beta_A[i],beta_B[i]) for i in range(n)]) +             sum([tot_term(a1,a,beta_A[i],beta_B[(i+1)%n]) for i in range(n)]) -             2*tot_term(a1,a,beta_A[0],beta_B[(n-1)]))/normf(a1,a)
        return -chsh_val  #Negative of expected value
    return inn

def max_violation(n):
    # Optimization protocol performed on a loop of m times
    # for a given n. Stores the minimum value of function 
    # obtained from the m optimized values. Parameters stored:
    # i. Minimized value of chsh(n)
    # ii. Optimal x
    t = time()
    m = 200
    min_res = [minimize(chsh(n),np.random.rand(4*n).astype(np.longdouble)) for _ in range(m)] # Can change m=50 to 100,200,etc. to inc. no. 
                                                                                                # of times initial parameters are tossed 
    print(time()-t,n,-min(min_res, key = itemgetter('fun'))['fun'])  # Prints n, time of computation, min chsh(n)
    return min(min_res, key = itemgetter('fun'))


############   Obtaining and storing results   ############

min_n = 8 # Min. n for doing above calculations
max_n = 9 # Max. n for doing above calculations

for n in range(min_n,max_n):
    o = max_violation(n)
    if 'max_chsh_ecs_v2.pi' in os.listdir():  # Stores in filename "max_chsh_ecs.pi"
        with open('max_chsh_ecs_v2.pi','rb') as f:
            x = load(f)
            if n in x and o['fun'] < x[n]['fun'] or n not in x:
            # Stores in file if chsh(n) was not optimized previously 
            # or optimization result is better than previous iteration.
                x[n] = o
                with open('max_chsh_ecs_v2.pi','wb') as f:
                    dump(x,f)
    else:
    # In the first run, generates the file max_chsh_ecs.pi with
    # a initial run of optimization.
        x = {n:o}
        with open('max_chsh_ecs_v2.pi','wb') as f:
            dump(x,f)

