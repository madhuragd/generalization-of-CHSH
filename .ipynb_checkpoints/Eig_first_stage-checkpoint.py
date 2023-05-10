import numpy as np
from numpy.linalg.linalg import eigvalsh
from scipy.optimize import minimize
from _pickle import dump,load
from operator import itemgetter
from util import LHS_of_BCCB, pickle_results, optimize
import os

min_n = 2
max_n = 4
num_opti = 10 # No. of times to optimize

def max_viol(n):
    def inner_fn(x):
        D1,D2 = x[:]  # Delta = beta_{i+1} - beta_i = gamma_{i+1} - gamma_i
        gamma = np.array([0]+[D1+D2*i for i in range(n-1)]) # gamma_1 = 0
        beta = np.array([D2*i for i in range(n-1)]+[D2*(n-2)+D1])  # beta_1 = 0
        return np.amin(np.linalg.eigvalsh(LHS_of_BCCB(beta,gamma)))
    return inner_fn

def init_point(n):
    return lambda : .09*np.random.random(2) # an argumentless function returning .09*np.random.random(1)


if __name__=="__main__":
    pickle_results('test_first_stage.pi',optimize(max_viol, init_point, num_opti),range(min_n,max_n))