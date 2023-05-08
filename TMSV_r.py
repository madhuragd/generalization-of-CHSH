#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[14]:


## Code to calculate the maximal CHSH bound for n-MZI+TMSV setting.

import numpy as np
from scipy.optimize import minimize
from _pickle import dump,load
import matplotlib.pyplot as plt
from operator import itemgetter
from time import time
import math
import os

####### Internal functions in the CHSH inequality #######

def f1(r,x):
    return np.exp(-abs(x)**2)*np.exp(np.tanh(r)**2)**(np.abs(x)**2)
# f1 = lambda r,x: np.exp(-abs(x)**2)*np.exp(np.tanh(r)**2)**(np.abs(x)**2)

def f2(r,t,x,y):
    return np.exp(-abs(x)**2-abs(y)**2)*np.exp(-2*(np.exp(1j*t)*x*y).real*np.tanh(r))
# f2 = lambda r,t,x,y: np.exp(-abs(x)**2-abs(y)**2)*np.exp(-2*(np.exp(1j*t)*x*y).real*np.tanh(r))
# Function $g_{i,j}=\bra{\Psi_{TMSV}}A(x_i) \otimes A(y_i)\ket{\Psi_{TMSV}}$ in manuscript

def f3(r,t,x,y):
    return 1+(-2*f1(r,x)-2*f1(r,y)+4*f2(r,t,x,y))/np.cosh(r)**2

# f3 = lambda r,t,x,y: 1+(-2*f1(r,x)-2*f1(r,y)+4*f2(r,t,x,y))/np.cosh(r)**2

def TMSV_r(n: int,r,params: callable):
    def inn_fn(x):
        t,beta,gamma = params(n,x)
        bccb_val = (sum([f3(r,t,beta[i],gamma[i]) for i in range(n)]) +  sum([f3(r,t,beta[i],gamma[(i+1)%n]) for i in range(n)]) - 2*f3(r,t,beta[0],gamma[n-1]))
        return -np.abs(bccb_val) # Negative of expected value
    return inn_fn
 
def params(n,x):
    beta = np.zeros(n,dtype=np.longdouble)
    gamma = np.zeros(n,dtype=np.longdouble)
    t = 0
    beta[1:] = x[:n-1] 
    gamma[1:] = x[n-1:]
    return t,beta,gamma
    
params.size = lambda n: 2*n-2

def max_viol(n: int,r):
    m = 100
    t = time()
    min_res = [minimize(TMSV_r(n,r,params),np.random.rand(params.size(n)).astype(np.longdouble)) for _ in range(m)] # Can change m=50 to 100,200,etc. to inc. no. 
    min_res = min(min_res, key = itemgetter('fun'))
    # print("time: ",time()-t, "min func: ",min_res['fun'])
    return min_res

# r_lin

def pickle_results_tmsv(file_name, range_of_r, range_of_n):
    for n in range_of_n:
        o = [max_viol(n,r) for r in range_of_r]
        if file_name in os.listdir(): # Stores in file: file_name
            with open(file_name,'rb') as f:
                d = load(f)
                if n not in d:
                # Stores in file if max_viol(n) was not optimized previously 
                # or optimization result is better than previous iteration.
                    d[n] = o
                    with open(file_name,'wb') as f:
                        dump(d,f)
        else:
        # In the first run, generates the file: file_name with
        # a initial run of optimization.
            d = {n:o}
            with open(file_name,'wb') as f:
                dump(d,f)

# r = 1
n = 2
# st = max_viol(2)
path = "C:\\Users\\user\\OneDrive\\Documents\\Confocal_acq\\Test\\"

r_lin = np.linspace(0,3,31)
# store = [max_viol(n,r) for r in r_lin]
pickle_results_tmsv(path+"tmsv_r_test.pi", r_lin, range(2,8))

