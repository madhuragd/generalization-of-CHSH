import numpy as np
from numpy.linalg.linalg import eigvalsh
from scipy.optimize import minimize
from _pickle import dump,load
from operator import itemgetter
from util import LHS_of_BCCB, pickle_results, optimize
import os

min_n = 2
max_n = 8
num_opti = 100 # No. of times to optimize

def max_viol(n):
	def inner_fn(x):
		D = x[0]  # Delta = beta_{i+1} - beta_i = gamma_{i+1} - gamma_i
		gamma = np.array([D*i for i in range(n)]) # gamma_1 = 0
		beta = np.array([D*i for i in range(n)])  # beta_1 = 0
		return np.amin(np.linalg.eigvalsh(LHS_of_BCCB(beta,gamma)))
	return inner_fn

def init_point(n):
	return lambda : .09*np.random.random(1) # an argumentless function returning .09*np.random.random(1)


if __name__=="__main__":
	pickle_results('max_chsh_eig.pi',optimize(max_viol, init_point, num_opti),range(min_n,max_n))
