#!/usr/bin/env python
# coding: utf-8

# In[20]:


## Code for plotting the maximal violations obtained for various cases.
## Also plots the diff. between arithmetic sequences {\beta_i} and {\gamma_i}
## as a function of n.

from _pickle import load, dump
import matplotlib.pyplot as plt
import numpy as np

###### Retrieving calculated values from all relevant files #########

with open('max_viol_ecs.pi','rb') as f:
    results_ECS = load(f)
    res_list_ECS = [(k, v) for k, v in results_ECS.items()]

with open('max_viol_tmsv.pi','rb') as f:
    res_list_TMSV = load(f)
    
with open('Eig_second_stage.pi','rb') as f:
    results_MZI = load(f)
    res_list_MZI = [(k, v) for k, v in results_MZI.items()]
	

    
    

# Maximal bounds of CHSH inequality for:    
# evalls = [(n[0],-n[1]['fun'].tolist()) for n in res_list_MZI[:18]]  # n-MZI settings
expecs = [(n[0],2*n[0]*np.cos(np.pi/(2*n[0]))) for n in res_list_MZI[:18]] # Theoretically calculated
encohs = [(n[0],-n[1]['fun']) for n in res_list_ECS] # MZI + ECS
squeez = [(k[0],-k[1]['fun']) for k in res_list_TMSV[:18]] # MZI + TMSV
real = [(n[0],-n[1]['fun']) for n in res_list_MZI] # n-MZI settings

# Maximal violations of CHSH inequality for: 
# viol_evalls = [(n[0],(n[1]-2*n[0]+2)) for n in evalls] # n-MZI settings
viol_expecs = [(n[0],(n[1]-2*n[0]+2)) for n in expecs] # Theoretically calculated
viol_encohs = [(n[0],(n[1]-2*n[0]+2)) for n in encohs] # MZI + ECS
viol_squeez = [(n[0],(n[1]-2*n[0]+2)) for n in squeez] # MZI + TMSV
viol_real = [(n[0],(n[1]-2*n[0]+2)) for n in real] # n-MZI settings


#####################     Plot Details    ##########################

### Plots CHSH violations all cases: theoretical, n-MZI settings, n-MZI + ECS, n-MZI + TMSV. 

plt.plot(*zip(*viol_expecs),'r.', *zip(*viol_evalls),'b^', *zip(*viol_encohs), 'k*',*zip(*viol_squeez), 'y*')

plt.legend(["Quantum bound for BCCB inequality",
			"Bound for BCCBi with $n$-MZI settings",
			"Bound for BCCBi with MZI + EC state",
			"Bound for BCCBi with MZI + TMSV state"], 
			bbox_to_anchor=(1, 0.37))
plt.xlabel(r"$n$",fontsize=14)
plt.xticks(np.arange(2, 21, step=2))
plt.minorticks_on()
plt.ylabel(r"$\mathrm{D}(n)$",fontsize=14)
plt.title(r'$\mathrm{D}(n)$ vs. $n$',fontsize=15)
plt.savefig('max_viol_all2.pdf', format='pdf', bbox_inches="tight") # Saves figure
plt.show()

### Plots CHSH violations for theoretical and n-MZI settings only 

plt.plot(*zip(*viol_expecs),'r.',*zip(*viol_evalls),'b^')
plt.legend(["Generalized CHSH","CHSH for $n$-MZI settings"], bbox_to_anchor=(0.4, 0.5))
plt.xlabel(r"$n$",fontsize=14)
plt.xticks(np.arange(2, 21, step=2))
plt.minorticks_on()
plt.ylabel(r"$\mathrm{D}(n)$",fontsize=14)
plt.title(r'$\mathrm{D}(n)$ vs. $n$',fontsize=15)
plt.savefig('max_viol_all.pdf', format='pdf', bbox_inches="tight") # Saves figure
plt.show()







