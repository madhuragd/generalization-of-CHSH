## Initial code to calculate max. bound of CHSH for n-MZI settings.
## Can be used to check how betas and gammas behave in the complex plane,
## for max. bound of n-MZI settings.

import numpy as np
from numpy.linalg.linalg import eigvalsh
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os
from time import time 
from operator import itemgetter
from _pickle import load, dump

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
    gamma = np.zeros(n,dtype=np.clongdouble)
    beta = np.zeros(n,dtype=np.clongdouble)
    gamma[1:] = x[1:n] #+ 1j * x[n:2*n]
    beta[1:] = x[n+1:] #+ 1j * x[3*n:]
    Ga = np.exp(np.array([[-(abs(i-j)**2 + i*j.conj() - j*i.conj())/2 for j in gamma] for i in gamma]))
    Gb = np.exp(np.array([[-(abs(i-j)**2 + i*j.conj() - j*i.conj())/2 for j in beta] for i in beta]))
    # print(np.linalg.eigvalsh(corr*Ga),np.linalg.eigvalsh(corr*Gb)) #np.linalg.eigvalsh(Ga),np.linalg.eigvalsh(Gb),
    try:
        try:
            La = np.linalg.cholesky(Ga) 
        except:
            mi=np.abs(np.amin(np.linalg.eigvalsh(Ga)))
            Ga+=corr*mi*np.eye(n) # Correction factor added to La. Required only when matrix is not semi-positive definite due to numerical inaccuracy 
                                  # In some cases very-small nonzero entries are present in place of zero. Correction factor rectifies this problem. 
            La = np.linalg.cholesky(Ga)
        try:
            Lb = np.linalg.cholesky(Gb)
        except:
            mi=np.abs(np.amin(np.linalg.eigvalsh(Gb)))
            Gb+=corr*mi*np.eye(n) # Same argument as for correction factor for La.
            Lb = np.linalg.cholesky(Gb)
#     try:
#         try:
#             La = np.linalg.cholesky(Ga) 
#         except:
#             mi=np.abs(np.amin(np.linalg.eigvalsh(Ga)))
#             Ga=corr*Ga # Correction factor added to La. Required only when matrix is not semi-positive definite due to numerical inaccuracy 
#                                   # In some cases very-small nonzero entries are present in place of zero. Correction factor rectifies this problem. 
#             La = np.linalg.cholesky(Ga)
#             La = La/corr
            
#         try:
#             Lb = np.linalg.cholesky(Gb)
#         except:
#             mi=np.abs(np.amin(np.linalg.eigvalsh(Gb)))
#             Gb=corr*Gb # Same argument as for correction factor for La.
#             Lb = np.linalg.cholesky(Gb)
#             Lb = Lb/corr
#             # print(np.linalg.eigvalsh(Gb))
            
            

        # Observables (As and Bs) for each party in orthonormal basis given by columns of La and Lb:
        As = [np.eye(n) - 2*row.reshape(1,n).T.conj() @ row.reshape(1,n) for row in La]
        Bs = [np.eye(n) - 2*row.reshape(1,n).T.conj() @ row.reshape(1,n) for row in Lb]

        # CHSH matrix
        CHSH = sum([np.kron(As[i],Bs[i]) + np.kron(As[(i+1)%n],Bs[i]) for i in range(n)])-2*np.kron(As[0],Bs[n-1])
        return np.amin(np.linalg.eigvalsh(CHSH)) # Min. eigenvalue for CHSH matrix
    except:
        return 0
  return inner_fn


path = "C:\\Users\\user\\OneDrive\\Documents\\Confocal_acq\\ECS\\"
def gen_guess(n,x):
    # x = np.random.rand(2)
    D1,D2 = x[:]
    guess_beta = [0.0]+[D1+D2*i for i in range(n-1)]
    guess_gamma = [D2*i for i in range(n-1)]+[D2*(n-2)+D1]
    return guess_beta+guess_gamma


with open(path+'max_chsh_eig.pi','rb') as f:
    results_guess = load(f)
    res_list_guess = [(k, v) for k, v in results_guess.items()]
    
guess_all = [gen_guess(i[0],i[1].x) for i in res_list_guess]  


def optimizeCHSH(n):
    """
    Optimization protocol for max_viol(n).
    Repeats minimization for m times and
    selects the minimum of the function 
    from the m results. Returns n, max.
    function (negative of min.) and array x
    for which optimal value of function
    is achieved.
    """
    x0 = guess_all[n-2]
    t = time()
    # print()
    print(max_viol(n)(x0))
    min_res = [minimize(max_viol(n),x0) for _ in range(1)] #np.random.randn(4*n)
    print(n,t-time(),min(i['fun'] for i in min_res))
    # min_fun = [m['fun'] for m in min_res]
    return min(min_res, key = itemgetter('fun'))


# d = [optimizeCHSH(n) for n in range(9,11)]
corr = 1e0 # Correction factor
Min = 11
Max = 12
if __name__=="__main__":
    for n in range(Min,Max):
        o = optimizeCHSH(n)
        if 'n3to8_real.pi' in os.listdir(): # Stores in filename "max_chsh_eig.pi"
            with open('n3to8_real.pi','rb') as f:
                d = load(f)
                if n in d and o['fun'] < d[n]['fun'] or n not in d:
                # Stores in file if max_viol(n) was not optimized previously 
                # or optimization result is better than previous iteration.
                    d[n] = o
                    with open('n3to8_real.pi','wb') as f:
                        dump(d,f)
        else:
        # In the first run, generates the file max_chsh_eig.pi with
        # a initial run of optimization.
            d = {n:o}
            with open('n3to8_real.pi','wb') as f:
                dump(d,f)

