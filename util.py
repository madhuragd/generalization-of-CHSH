import numpy as np
from numpy.linalg.linalg import eigvalsh
import time as time

def one_party_local_observables(setting):
	"""Returns a list of local observables for one party using MZI setup, where the 1d numpy.array of displacements are given in the argument: setting"""
	n, = setting.shape
	# Gram matrix
	G = np.exp(np.array([[-(abs(i-j)**2 + i*j.conj() - j*i.conj())/2 for j in setting] for i in setting]))
	try:
		# Cholesky decomposition
		L = np.linalg.cholesky(G) 
	except:
		# Correction factor added to La. Required only when matrix is not semi-positive definite due to numerical inaccuracy 
		# In some cases very-small nonzero entries are present in place of zero. Correction factor rectifies this problem.
		corr = 3e0
		mi=np.abs(np.amin(np.linalg.eigvalsh(G)))
		G+=corr*mi*np.eye(n)
	X = [np.eye(n) - 2*row.reshape(1,n).T.conj() @ row.reshape(1,n) for row in L] # Observables in orthonormal basis given by columns of L
	return X

def LHS_of_BCCB(beta,gamma):
	n = len(beta)
	obsA = one_party_local_observables(beta)
	obsB = one_party_local_observables(gamma)
	return sum([np.kron(obsA[i],obsB[i]) + np.kron(obsA[(i+1)%n],obsB[i]) for i in range(n)])-2*np.kron(obsA[0],obsB[n-1])


if __name__ == "__main__":
	setting = (np.array([0+0j,1+1j]))
	for i in one_party_local_observables(setting):
		print(i)
	print(LHS_of_BCCB(setting,setting))

