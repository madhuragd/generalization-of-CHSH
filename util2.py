import numpy as np
from numpy.linalg.linalg import eigvalsh
# import time as time
from scipy.optimize import minimize
from operator import itemgetter
from util import max_viol

def optimize1(n,num_opti,typ):
    """
    Optimization protocol for max_viol(n).
    Repeats minimization for m times and
    selects the minimum of the function 
    from the m results. Returns n, max.
    function (negative of min.) and array x
    for which optimal value of function
    is achieved.
    """
    # t=time()
    if typ == 'arithmetic':
        x0 = .09*np.random.random(1)
    elif typ == 'real':
        x0 = np.random.rand(2*n-2)
    elif typ == 'general':
        x0 = np.random.rand(4*n)
        min_res = min((minimize(max_viol(n),x0,method='Powell',tol=1e-15) for _ in range(num_opti)), key=itemgetter('fun'))
        print(n, min_res['fun'])
        return n, -min_res['fun'], min_res.x
    min_res = min((minimize(max_viol(n),x0,method='Powell',tol=1e-15) for _ in range(num_opti)), key=itemgetter('fun'))
    print(n, min_res['fun'])
    return min_res