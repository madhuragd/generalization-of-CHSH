import numpy as np
from numpy.linalg.linalg import eigvalsh
from scipy.optimize import minimize
from _pickle import dump,load
from operator import itemgetter
from util import LHS_of_BCCB
import os

min_n = 2
max_n = 8
num_opti = 100 # No. of times to optimize

def max_viol(n):
	def inner_fn(x):
		beta = x[:n] + 1j * x[n:2*n]
		gamma = x[2*n:3*n] + 1j * x[3*n:]
		return np.amin(np.linalg.eigvalsh(LHS_of_BCCB(beta,gamma)))
	return inner_fn

def optimize(n,num_opti):
	min_res = min((minimize(max_viol(n),np.random.rand(4*n),method='Powell',tol=1e-15) for _ in range(num_opti)), key=itemgetter('fun'))
	print(n, min_res['fun'])
	return -min_res['fun'], min_res.x

if __name__=="__main__":
	pickle_results('n3to8.pi',optimize,range(min_n,max_n))
