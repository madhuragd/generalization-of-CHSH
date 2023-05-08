
## Code to calculate the maximal CHSH bound for n-MZI+TMSV setting.

import numpy as np
from scipy.optimize import minimize
from _pickle import dump,load
import matplotlib.pyplot as plt
from operator import itemgetter
from time import time
import math
import os

####### Internal functions in the BCCB inequality #######

def f1(r,x):
    return np.exp(-abs(x)**2)*np.exp(np.tanh(r)**2)**(np.abs(x)**2)


def f2(r,t,x,y):
    return np.exp(-abs(x)**2-abs(y)**2)*np.exp(-2*(np.exp(1j*t)*x*y).real*np.tanh(r))


def f3(r,t,x,y):
    return 1+(-2*f1(r,x)-2*f1(r,y)+4*f2(r,t,x,y))/np.cosh(r)**2



def TMSV_r(n,r,params: callable):
    def inn_fn(x):
        t,beta,gamma = params(n,x)
        bccb_val = (sum([f3(r,t,beta[i],gamma[i]) for i in range(n)]) +  sum([f3(r,t,beta[i],gamma[(i+1)%n]) for i in range(n)]) - 2*f3(r,t,beta[0],gamma[n-1]))
        return -np.abs(bccb_val) # Negative of expected value
    return inn_fn
 
def params(n,x):
    beta = np.zeros(n,dtype=np.longdouble)
    gamma = np.zeros(n,dtype=np.longdouble)
    t = 0
    beta[1:] = x[:n-1] 
    gamma[1:] = x[n-1:]
    return t,beta,gamma
    
params.size = lambda n: 2*n-2

def max_viol(n,r):
    m = 100
    t = time()
    min_res = [minimize(TMSV_r(n,r,params),np.random.rand(params.size(n)).astype(np.longdouble)) for _ in range(m)] # Can change m=50 to 100,200,etc. to inc. no. 
    min_res = min(min_res, key = itemgetter('fun'))
    # print("time: ",time()-t, "min func: ",min_res['fun'])
    return min_res


r_lin = np.linspace(0,3,31)

st = {i:[max_viol(i,r) for r in r_lin] for i in range(2,8)}

with open(path+"tmsv_r_1.pi",'wb') as f:
    dump(st,f)

    
with open(path+"tmsv_r_1.pi",'rb') as f:
    d = load(f)
    
r_lin = np.linspace(0,3,31)
n = range(2,8)
funs = [[-d[i][j]['fun']-2*i+2 for j in range(len(r_lin))] for i in n]


for i in range(len(n)):
    plt.plot(r_lin,funs[i],'.')
    
#      plt.legend(bbox_to_anchor=(1.2,1))
    # plt.yticks(np.arange(0,n_max))
plt.ylabel(r'$D(n)$',fontsize=14)
plt.xlabel(r'$r$',fontsize=14)
    # plt.text(gamma[0]-0.05,n-0.5,r'$r=$%s'%m,fontsize=11)
plt.title(r'$D(n)$ vs. $r$',fontsize=15)
# path = "C:\\Users\\user\\OneDrive\\Documents\\Confocal_acq\\Test\\"
plt.savefig('tmsv_viol_r.pdf', format='pdf',bbox_inches="tight") # Saves figure
plt.show()
