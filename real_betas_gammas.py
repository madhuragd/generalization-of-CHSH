import numpy as np
from numpy.linalg.linalg import eigvalsh
from scipy.optimize import minimize
from _pickle import dump,load
from operator import itemgetter
from util import LHS_of_BCCB
import os

Min = 2 # Min. n for doing above calculations
Max = 9 # Max. n for doing above calculations
num_opti = 100 # No. of times to optimize

def max_viol(n):
	def inner_fn(x):
		beta = np.zeros(n) # gamma_1 = 0
		gamma = np.zeros(n) # beta_1 = 0
		beta[1:] = x[:n-1] #+ 1j * x[n:2*n]
		gamma[1:] = x[n-1:] #+ 1j * x[3*n:]
		return np.amin(np.linalg.eigvalsh(LHS_of_BCCB(beta,gamma)))
	return inner_fn

def optimize(n,num_opti):
	min_res = min((minimize(max_viol(n),np.random.rand(2*n-2),method='Powell',tol=1e-15) for _ in range(num_opti)), key=itemgetter('fun'))
	print(n, min_res['fun'])
	return -min_res['fun'], min_res.x

if __name__=="__main__":
	for n in range(Min,Max):
		o = optimize(n,num_opti)
		if 'n3to8_real.pi' in os.listdir(): # Stores in filename "n3to8_real.pi"
			with open('n3to8_real.pi','rb') as f:
				d = load(f)
				print(d)
				if n in d and o['fun'] < d[n]['fun'] or n not in d:
				# Stores in file if max_viol(n) was not optimized previously 
				# or optimization result is better than previous iteration.
					d[n] = o
					with open('n3to8_real.pi','wb') as f:
						dump(d,f)
		else:
		# In the first run, generates the file n3to8_real.pi with
		# a initial run of optimization.
			d = {n:o}
			with open('n3to8_real.pi','wb') as f:
				dump(d,f)
