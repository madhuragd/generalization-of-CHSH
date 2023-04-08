###########   Obtaining and storing results   ############

from util2 import optimize1
import os
from _pickle import dump,load

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

def optimize(n,x0,num_opti):
	min_res = min((minimize(max_viol(n),.09*np.random.random(1),method='Powell',tol=1e-15) for _ in range(num_opti)), key=itemgetter('fun'))
	print(n, min_res['fun'])
	return -min_res['fun'], min_res.x

if __name__=="__main__":
	pickle_results('max_chsh_eig.pi',optimize,range(min_n,max_n))
