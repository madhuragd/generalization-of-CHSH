
## Code to calculate the maximal BCCB bound for n-MZI+TMSV setting as a function of squeezing parameter (r).

import numpy as np
from scipy.optimize import minimize
from _pickle import dump,load
import matplotlib.pyplot as plt
from operator import itemgetter
from time import time
import math
import os

####### Internal functions in the BCCB inequality #######

def f1(r,x):
    return np.exp(-abs(x)**2)*np.exp(np.tanh(r)**2)**(np.abs(x)**2)


def f2(r,t,x,y):
    return np.exp(-abs(x)**2-abs(y)**2)*np.exp(-2*(np.exp(1j*t)*x*y).real*np.tanh(r))


def f3(r,t,x,y):
    return 1+(-2*f1(r,x)-2*f1(r,y)+4*f2(r,t,x,y))/np.cosh(r)**2



def TMSV_r(n,r,params: callable):
    # Function for calculating BCCBi LHS for n-MZI+TMSV with r as a free parameter
    def inn_fn(x):
        t,beta,gamma = params(n,x)
        bccb_val = (sum([f3(r,t,beta[i],gamma[i]) for i in range(n)]) +  sum([f3(r,t,beta[i],gamma[(i+1)%n]) for i in range(n)]) - 2*f3(r,t,beta[0],gamma[n-1]))
        return -np.abs(bccb_val) # Negative of the summed value
    return inn_fn
 
def params(n,x):
    # Function for calculating the displacements or measurement settings
    beta = np.zeros(n,dtype=np.longdouble) 
    gamma = np.zeros(n,dtype=np.longdouble)
    t = 0 # phase factor of squeezing parameter
    beta[1:] = x[:n-1] 
    gamma[1:] = x[n-1:]
    return t,beta,gamma
    
params.size = lambda n: 2*n-2 

def max_viol(n,r):
    # Function for optimization of BCCBi LHS value
    m = 100 # No. of runs for optimization
    t = time()
    min_res = [minimize(TMSV_r(n,r,params),np.random.rand(params.size(n)).astype(np.longdouble)) for _ in range(m)] # Minimization protocol
    min_res = min(min_res, key = itemgetter('fun'))
    # print("time: ",time()-t, "min func: ",min_res['fun'])
    return min_res

if __name__ == "__main__":

	r_lin = np.linspace(0,3,31)

	st = {i:[max_viol(i,r) for r in r_lin] for i in range(2,8)}

	with open(path+"tmsv_r.pi",'wb') as f:
		dump(st,f)

