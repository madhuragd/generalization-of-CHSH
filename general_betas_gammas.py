import numpy as np
from numpy.linalg.linalg import eigvalsh
from scipy.optimize import minimize
from operator import itemgetter
from util import LHS_of_BCCB, pickle_results, optimize

min_n = 2
max_n = 8
num_opti = 5 # No. of times to optimize

def max_viol(n):
	def inner_fn(x):
		beta = x[:n] + 1j * x[n:2*n]
		gamma = x[2*n:3*n] + 1j * x[3*n:]
		try:
			return np.amin(np.linalg.eigvalsh(LHS_of_BCCB(beta,gamma)))
		except:
			print(n)
			return 0
	return inner_fn

def init_point(n):
	return lambda : np.random.rand(4*n) # an argumentless function returning np.random.rand(4*n)

if __name__=="__main__":
	pickle_results('n3to8.pi',optimize(max_viol, init_point, num_opti),range(min_n,max_n))
