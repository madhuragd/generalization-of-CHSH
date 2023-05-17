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
from util import LHS_of_BCCB, pickle_results, optimize
from _pickle import load, dump

def max_viol(n):
    def inner_fn(x):
        gamma = x[:n] 
        beta = x[n:] 
        
        try:
            return np.amin(np.linalg.eigvalsh(LHS_of_BCCB(beta,gamma)))
        except np.linalg.LinAlgError:
            print(n)
            return 0
    return inner_fn

def gen_guess(n,x): # Generates the sets of betas and gammas from Eig_first_stage.pi
    D1,D2 = x[:] # Two optimized parameters from first stage of optimization
    guess_beta = [0.0]+[D1+D2*i for i in range(n-1)] # Initial guess for betas
    guess_gamma = [D2*i for i in range(n-1)]+[D2*(n-2)+D1] # Initial guess for gammas
    return guess_beta+guess_gamma # Combined list of betas and gammas


def init_point(n): # Generates initial point dictionary with n values as keys
    with open('Eig_first_stage.pi','rb') as f: 
        results_guess = load(f)
        res_list_guess = [(k, v) for k, v in results_guess.items()]
    guess_all = {i[0]:gen_guess(i[0],i[1].x) for i in res_list_guess}  # All betas and gammas guess values stored in dictionary

    return guess_all[n]

def optimize_second_stage(n): # Function for second stage of optimization. Optimizes over 2n-2 parameters deterministically.
    x0 = init_point(n) # Initial point taken from first_stage file
    t = time()
    min_res = minimize(max_viol(n),x0) # Minimization
    print(n,time()-t,min_res['fun'])
    return min_res

if __name__=="__main__":
    pickle_results('Eig_second_stage.pi',optimize_second_stage,range(2,20)) # Stores data after optimization 
                    # Filename
