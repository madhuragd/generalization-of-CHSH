#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.optimize import minimize
from operator import itemgetter
from time import time

## Code to calculate the maximal BCCB ineq. violation for n-MZI+ECS setting.

####### Internal functions in the BCCB inequality #######

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

def two_party_expv(a,alpha,b,c,d,b1,b2):
	"""
	For displacements alpha,b,c,d,b1,b2 and complex amplitude a, 
	returns an expected value of (I - 2\ket{b1}\bra{b1})\otimes(I - 2\ket{b2}\bra{b2}) 
	in the (unnormalised) pure state (a * \ket{alpha}\otimes\ket{b} + \ket{c}\otimes\ket{d})
	""" 
	return (np.abs(a)**2 * one_party_expv(alpha,alpha,b1)*one_party_expv(c,c,b2) +\
		2*a*one_party_expv(b,alpha,b1)*one_party_expv(d,c,b2) +\
		one_party_expv(b,b,b1)*one_party_expv(d,d,b2)).real

def norm2(a,alpha,b,c,d):
	"""
	Returns the squared norm of the state vector:
	(a * \ket{alpha}\otimes\ket{b} + \ket{c}\otimes\ket{d})
	""" 
	return np.abs(a)**2 + 2*(a*coh_state_prod(b,alpha)*coh_state_prod(d,c)).real + 1

def bccb_expv(a,alpha,b,c,d,beta,gamma):
    # bccb_val = 1+2
    # return 0
    bccb_val = (sum([two_party_expv(a,alpha,b,c,d,beta[i],gamma[i]) for i in range(n)]) + \
            sum([two_party_expv(a,alpha,b,c,d,beta[i],gamma[(i+1)%n]) for i in range(n)]) - \
            2*two_party_expv(a,alpha,b,c,d,beta[0],gamma[(n-1)])) / norm2(a,alpha,b,c,d)
    return -bccb_val

##########   Calculating the BCCB ineq. expected value   ############

def bccb(param: callable, n: int):
    """
    For a given n, returns a function inn calculating for the given array x 
    the expected value of BCCB inequality in entangled coherent state described by parameters a,alpha,b,c,d
    and for sequences of MZI observables defined in arrays: beta and gamma for the first and the second party respectively.
    The first parameter param is a function translating a real array x into a,alpha,b,c,d,beta,gamma    
    """
    def inn(x):
        a,alpha,b,c,d,beta,gamma = param(n,x)
        bccb_val = (sum([two_party_expv(a,alpha,b,c,d,beta[i],gamma[i]) for i in range(n)]) + \
            sum([two_party_expv(a,alpha,b,c,d,beta[i],gamma[(i+1)%n]) for i in range(n)]) - \
            2*two_party_expv(a,alpha,b,c,d,beta[0],gamma[(n-1)])) / norm2(a,alpha,b,c,d)
        return -bccb_val
    return inn

###########  Different parametrisations ##############

def param1(n,x):	# alpha, a: real, displacements: real, beta_1 = ... = beta_{n-2}, beta_{n-1} = beta_n, gamma_1 = gamma_{n-1}, gamma_2 = ... = gamma_{n-2} 
	alpha,a,beta_1,beta_n,gamma_1,gamma_2,gamma_n=x
	b = 0;	c = 0;	d = -alpha
	beta = np.zeros(n,dtype=np.clongdouble)
	gamma = np.zeros(n,dtype=np.clongdouble)
	beta[:n-2] = beta_1
	beta[n-2:] = beta_n
	gamma[1] = gamma_1
	gamma[1:n-2] = gamma_2
	gamma[n-2] = gamma_1
	gamma[n-1] = gamma_n
	return a,alpha,b,c,d,beta,gamma
	
param1.size = lambda n: 7

def param2(n,x):	# alpha, a: real, displacements: real, beta_1 = ... = beta_{n-2}, beta_{n-1} = beta_n, gamma_1 = gamma_{n-1}
	alpha,a,beta_1,beta_n=x[:4]
	b = 0;	c = 0;	d = -alpha
	beta = np.zeros(n,dtype=np.clongdouble)
	beta[:n-2] = beta_1
	beta[n-2:] = beta_n
	gamma = np.zeros(n,dtype=np.clongdouble)
	gamma[1:] = x[4:]
	gamma[0] = gamma[n-2]
	return a,alpha,b,c,d,beta,gamma

param2.size = lambda n: n+3

def param3(n,x):	# alpha, a: real, displacements: real
	alpha,a = x[:2]
	x = x[2:]
	beta = x[:n]
	gamma = x[n:]
	b = 0;	c = 0;	d = -alpha
	return a,alpha,b,c,d,beta,gamma

param3.size = lambda n: 2*n+2

def param4(n,x):	# alpha, a: real, displacements: complex
	alpha,a = x[:2]
	x = x[2:2*n+2]+1j*x[2*n+2:]
	beta = x[:n]
	gamma = x[n:]
	b = 0;	c = 0;	d = -alpha
	return a,alpha,b,c,d,beta,gamma

param4.size = lambda n: 4*n+2

def param5(n,x):	# alpha, a: complex, displacements: complex
	x = x[:2*n+2]+1j*x[2*n+2:]
	alpha,a = x[:2]
	x=x[2:]
	beta = x[:n]
	gamma = x[n:]
	b = 0;	c = 0;	d = -alpha
	return a,alpha,b,c,d,beta,gamma

param5.size = lambda n: 4*n+4

############################

def max_violation(n: int, m=300):
	"""
    Two-stage optimization protocol for given n - the number of MZI settings for each party.
    On the first stage the optimisation over a reduced set of parameters is performed m times.
    The best result is a starting point for the second stage optimisation which is performed 
    once over the full set of parameters. The function returns the result of the second stage 
    optimisation.
    """
	if n==2:
		param = param3
	elif n==6:
		param = param2
	else:
		param = param1
	t = time()
	min_res = [minimize(bccb(param,n),np.random.rand(param.size(n)).astype(np.longdouble)) for _ in range(m)]
	min_res = min(min_res,key = itemgetter('fun'))
	print(f"n={n}, 1st stage violation: {-min_res['fun']-2*(n-1)}, time: {time()-t},", end=" ")
	t=time()
	x = min_res['x']
	a,alpha,b,c,d,beta,gamma = param(n,x)
	x = np.concatenate([beta,gamma])
	x = x.real									# if param3 in the 2nd stage
	#x = np.concatenate([x.real,x.imag])		# if param4 in the 2nd stage
	x = np.concatenate([np.array([alpha,a]),x])
	min_res = minimize(bccb(param3,n),x)
	print(f"2nd stage violation {-min_res['fun']-2*(n-1)}, time: {time()-t}")
	return min_res

if __name__ == "__main__":

	from util import pickle_results 

	pickle_results('test_max_viol_ecs.pi', max_violation, [4,17])
