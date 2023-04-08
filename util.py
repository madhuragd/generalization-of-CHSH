import numpy as np
from numpy.linalg.linalg import eigvalsh
import time as time

def f(n,setting):
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


def max_viol(n):
  """
  Initial version of function to calculate max. 
  bound of CHSH inequality for n-MZI settings. 
  Parameters: n (number of settings).
  Inner function: inner_fn(x) takes 'x' as parameter
  for optimization. x: array_like; having initial 
  guesses for real and imaginary parts of gamma_i,
  beta_i (i = 1 to n). Dim of x = 4n.
  
  """
  def inner_fn(x):
    if len(x) == 4*n: # In case of randomly chosen complex betas and gammas without correlation
        gamma = x[:n] + 1j * x[n:2*n]
        beta = x[2*n:3*n] + 1j * x[3*n:]
    elif len(x) == 1: # In case of randomly chosen betas and gammas in equal arithmetic sequences
        D = x[0]  # Delta = beta_{i+1} - beta_i = gamma_{i+1} - gamma_i
        gamma = np.array([D*i for i in range(n)]) # gamma_1 = 0
        beta = np.array([D*i for i in range(n)])  # beta_1 = 0
    elif len(x) == 2*n-2:  # In case of randomly chosen real betas and gammas without correlation
        gamma = np.zeros(n,dtype=np.clongdouble) # gamma_1 = 0
        beta = np.zeros(n,dtype=np.clongdouble) # beta_1 = 0
        gamma[1:] = x[:n-1] #+ 1j * x[n:2*n]
        beta[1:] = x[n-1:] #+ 1j * x[3*n:]
    try:
    # Observables (As and Bs) for each party:
      As = f(n,gamma)
      Bs = f(n,beta)
    # CHSH matrix
      CHSH = sum([np.kron(As[i],Bs[i]) + np.kron(As[(i+1)%n],Bs[i]) for i in range(n)])-2*np.kron(As[0],Bs[n-1])
      return np.amin(np.linalg.eigvalsh(CHSH)) # Min. eigenvalue for CHSH matrix
    except:
      return 0
  return inner_fn



