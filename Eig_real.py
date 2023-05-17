import numpy as np
from numpy.linalg.linalg import eigvalsh
from util import LHS_of_BCCB, pickle_results, optimize

min_n = 2
max_n = 8
num_opti = 100 # No. of times to optimize

def max_viol(n):
	def inner_fn(x):
		beta = np.zeros(n) # gamma_1 = 0
		gamma = np.zeros(n) # beta_1 = 0
		beta[1:] = x[:n-1] #+ 1j * x[n:2*n]
		gamma[1:] = x[n-1:] #+ 1j * x[3*n:]
		return np.amin(np.linalg.eigvalsh(LHS_of_BCCB(beta,gamma)))
	return inner_fn

def init_point(n):
	return lambda : np.random.rand(2*n-2) # an argumentless function returning np.random.rand(2*n-2)


if __name__=="__main__":
	pickle_results('n3to8_real.pi',optimize(max_viol, init_point, num_opti),range(min_n,max_n))
