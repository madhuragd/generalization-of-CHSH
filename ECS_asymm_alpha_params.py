#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from scipy.optimize import minimize
from operator import itemgetter
# import os

coh_state_prod = lambda x,y: np.exp(-np.abs(x-y)**2/2+1j*(x.conjugate()*y).imag)
one_party_expv = lambda x,y,b: coh_state_prod(x,y) - 2*coh_state_prod(x,b)*coh_state_prod(b,y)
two_party_expv = lambda a1,a,b,c,d,b1,b2: (np.abs(a1)**2 * one_party_expv(a,a,b1)*one_party_expv(c,c,b2) + 	2*a1*one_party_expv(b,a,b1)*one_party_expv(d,c,b2) + 	one_party_expv(b,b,b1)*one_party_expv(d,d,b2)).real
norm = lambda a1,a,b,c,d: np.abs(a1)**2 + 2*(a1*coh_state_prod(b,a)*coh_state_prod(d,c)).real + 1

def chsh(n):
    def inn(x):
        #n = (x.shape[0] // 2 - 1 ) // 2
        beta_A = np.zeros(n,dtype=np.clongdouble)
        beta_B = np.zeros(n,dtype=np.clongdouble)
        x = x[:2*n]+1j*x[2*n:]
        a1,a = x[:2]
        # a1 = 1+1j*0
        # b = 0
        # c = 0
        b = 0+1j*0
        # d = -a
        c = 0+1j*0
        d = a + b - c
        beta_A[1:] = x[2:n+1]
        beta_B[1:] = x[n+1:]
        # print(beta_A,beta_B)
        chsh_val = (sum([two_party_expv(a1,a,b,c,d,beta_A[i],beta_B[i]) for i in range(n)]) +             sum([two_party_expv(a1,a,b,c,d,beta_A[i],beta_B[(i+1)%n]) for i in range(n)]) -             2*two_party_expv(a1,a,b,c,d,beta_A[0],beta_B[(n-1)]))/norm(a1,a,b,c,d)
        return -chsh_val
    return inn



# for i in range(2,5):
# 	mv = max_violation(i)
# 	print(i,-mv['fun'],2*n*np.cos(np.pi/(2*n)))

def max_violation(n):
    min_res = [minimize(chsh(n),np.random.rand(4*n).astype(np.longdouble)) for _ in range(100)]  #*np.ones(4*n+4,dtype=np.longdouble)
    print(n,min(min_res,key = itemgetter('fun'))['fun'],2*n*np.cos(np.pi/(2*n)))
    return min(min_res, key = itemgetter('fun'))

from _pickle import load, dump
import os 

min_n = 2
max_n = 20
# path = "C:\\Users\\user\\OneDrive\\Documents\\Confocal_acq\\Test\\"

for n in np.arange(3,4):
    o = max_violation(n)
    if 'max_chsh_ecs_1.pi' in os.listdir():
        with open('max_chsh_ecs_1.pi','rb') as f:
            x = load(f)
            if n in x and o['fun'] < x[n]['fun'] or n not in x:
                x[n] = o
                with open('max_chsh_ecs_1.pi','wb') as f:
                    dump(x,f)
    else:
        x = {n:o}
        with open('max_chsh_ecs_1.pi','wb') as f:
            dump(x,f)


# In[11]:




