#!/usr/bin/env python
# coding: utf-8

## Code for plotting the maximal violations obtained for various cases.
## Also plots the diff. between arithmetic sequences {\beta_i} and {\gamma_i}
## as a function of n.

from _pickle import load, dump
import matplotlib.pyplot as plt
import numpy as np

###### Retrieving violations from all relevant files #########

with open('max_viol_ecs.pi','rb') as f:
    results_ECS = load(f)
    viol_ecs = [(n,-v['fun']-2*n+2) for n, v in results_ECS.items()]

with open('max_viol_tmsv.pi','rb') as f:
    res_list_TMSV = load(f)
    viol_tmsv = [(n,-v['fun']-2*n+2) for n,v in res_list_TMSV.items()] # MZI + TMSV
    
with open('Eig_second_stage.pi','rb') as f:
    results_MZI = load(f)
    viol_eig = [(n,-v['fun']-2*n+2) for n,v in results_MZI.items()] # n-MZI settings
    viol_theor = [(n,2*n*np.cos(np.pi/(2*n))-2*n+2) for n in results_MZI.keys()]
        
#####################     Plot Details    ##########################

### Plots CHSH violations all cases: theoretical, n-MZI settings, n-MZI + ECS, n-MZI + TMSV. 

plt.plot(*zip(*viol_theor),'r.', *zip(*viol_eig),'b^', *zip(*viol_ecs), 'k*',*zip(*viol_tmsv), 'y*')

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

plt.plot(*zip(*viol_all),'r.',*zip(*viol_theor),'b^')
plt.legend(["Generalized CHSH","CHSH for $n$-MZI settings"], bbox_to_anchor=(0.4, 0.5))
plt.xlabel(r"$n$",fontsize=14)
plt.xticks(np.arange(2, 21, step=2))
plt.minorticks_on()
plt.ylabel(r"$\mathrm{D}(n)$",fontsize=14)
plt.title(r'$\mathrm{D}(n)$ vs. $n$',fontsize=15)
plt.savefig('max_viol_all.pdf', format='pdf', bbox_inches="tight") # Saves figure
plt.show()







