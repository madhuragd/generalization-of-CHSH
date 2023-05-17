import numpy as np
from numpy.linalg.linalg import eigvalsh
from util import LHS_of_BCCB, pickle_results, optimize


min_n = 2
max_n = 20
num_opti = 100 # No. of times to optimize

def max_viol(n):
    def inner_fn(x):
        D1,D2 = x[:]  # D1 = beta_n - beta_{n-1} = gamma_1 - gamma_0; D2 = beta_1 - beta_0 = gamma_n - gamma_{n-1}
        gamma = np.array([0]+[D1+D2*i for i in range(n-1)]) # gamma_1 = 0
        beta = np.array([D2*i for i in range(n-1)]+[D2*(n-2)+D1])  # beta_1 = 0
        
        try:
            return np.amin(np.linalg.eigvalsh(LHS_of_BCCB(beta,gamma)))
        except np.linalg.LinAlgError:
            print(n)
            return 0
    return inner_fn

def init_point(n):
    return lambda : .09*np.random.rand(2) # an argumentless function returning .09*np.random.random(2)


if __name__=="__main__":
    
    pickle_results('Eig_first_stage.pi',optimize(max_viol, init_point, num_opti),range(min_n,max_n))
