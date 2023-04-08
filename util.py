import numpy as np
from _pickle import dump,load
import os

def one_party_local_observables(setting, corr = 3e0):
	"""Returns a list of local observables for one party using MZI setup, where the 1d numpy.array of displacements are given in the argument: setting
"""
	n, = setting.shape
	# Gram matrix
	G = np.exp(np.array([[-(abs(i-j)**2 + i*j.conj() - j*i.conj())/2 for j in setting] for i in setting]))
	try:
		# Cholesky decomposition
		L = np.linalg.cholesky(G)
	except:
		# Correction factor added to La. Required only when matrix is not semi-positive definite due to numerical inaccuracy 
		# In some cases very-small nonzero entries are present in place of zero. Correction factor rectifies this problem.
		mi=np.abs(np.amin(np.linalg.eigvalsh(G)))
		G+=corr*mi*np.eye(n)
		L = np.linalg.cholesky(G)
	X = [np.eye(n) - 2*row.reshape(1,n).T.conj() @ row.reshape(1,n) for row in L] # Observables in orthonormal basis given by columns of L
	return X

def LHS_of_BCCB(betas,gammas):
	"""Returns the observable in the expected value on the LHS of BCCB inequality, 
where observables are defined by displacements betas, gammas for respective parties.
"""
	n = len(betas)
	obsA = one_party_local_observables(betas)
	obsB = one_party_local_observables(gammas)
	return sum([np.kron(obsA[i],obsB[i]) + np.kron(obsA[(i+1)%n],obsB[i]) for i in range(n)])-2*np.kron(obsA[0],obsB[n-1])

def n_successfull(f: callable, init_point: callable, num_opti: int):
	"""returns n_opti successfull optimisations of f with initial point produced each time by init_point
"""
	i = 0
	while i < num_opti:
		yield minimize(f,init_point(),method='Powell',tol=1e-15)
		try:
			yield minimize(f,init_point(),method='Powell',tol=1e-15)
			i+=1
		except:
			continue

def optimize(f: callable, init_point: callable, num_opti: int):
	"""returns the function prescribing to n the minimum over num_opti successfull optimisations. 
f returns for n an optimised function, 
init_point returns for n an argumentless function returning a random init point
"""
	def inner_fn(n):
		min_res = min(n_successfull(f(n),init_point(n),num_opti), key=itemgetter('fun'))
		print(n, min_res['fun'])
		return -min_res['fun'], min_res.x
	return inner_fn

def pickle_results(file_name, optimize_function, range_of_n):
	for n in range_of_n:
		o = optimize_function(n)
		if file_name in os.listdir(): # Stores in file: file_name
			with open(file_name,'rb') as f:
				d = load(f)
				if n in d and o['fun'] < d[n]['fun'] or n not in d:
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

# unit tests of funtions

if __name__ == "__main__":
	setting = (np.array([0+0j,1+1j]))
	for i in one_party_local_observables(setting):
		print(i)
	print(LHS_of_BCCB(setting,setting))

