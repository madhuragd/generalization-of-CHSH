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

with open('max_chsh_tmsv.pi','rb') as f:
    results_TMSV = load(f)
    res_list_TMSV = results_TMSV
    # res_list_TMSV = [(k, v) for k, v in results_TMSV.items()]
    
with open('max_chsh_eig.pi','rb') as f:
    results_MZI = load(f)
    res_list_MZI = [(k, v) for k, v in results_MZI.items()]
	
with open(path+'n3to8_real.pi','rb') as f:
    results_real = load(f)
    res_list_real = [(k, v) for k, v in results_real.items()]
    
    

# Maximal bounds of CHSH inequality for:    
evalls = [(n[0],-n[1]['fun'].tolist()) for n in res_list_MZI[:18]]  # n-MZI settings
expecs = [(n[0],2*n[0]*np.cos(np.pi/(2*n[0]))) for n in res_list_MZI[:18]] # Theoretically calculated
encohs = [(n[0],-n[1]['fun']) for n in res_list_ECS] # MZI + ECS
squeez = [(k[0],-k[1]['fun']) for k in res_list_TMSV[:18]] # MZI + TMSV
real = [(n[0],-n[1]['fun']) for n in res_list_real] # Theoretically calculated

# Maximal violations of CHSH inequality for: 
viol_evalls = [(n[0],(n[1]-2*n[0]+2)) for n in evalls] # n-MZI settings
viol_expecs = [(n[0],(n[1]-2*n[0]+2)) for n in expecs] # Theoretically calculated
viol_encohs = [(n[0],(n[1]-2*n[0]+2)) for n in encohs] # MZI + ECS
viol_squeez = [(n[0],(n[1]-2*n[0]+2)) for n in squeez] # MZI + TMSV
viol_real = [(n[0],(n[1]-2*n[0]+2)) for n in real] # Theoretically calculated


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


### Plots \Delta (diff. between arithmentic sequences of {\beta_i} and {\gamma_i}) and 1/|\Delta| vs. n:

fig = plt.figure(figsize=(10,4)) 
axs = [fig.add_subplot(1,2,i+1) for i in range(2)]
fig.subplots_adjust(wspace=0.3, hspace=0.2)

###### Plotting |\Delta| vs. n ######

Delta = [(n[0],np.abs(n[1].x.tolist())) for n in res_list_MZI[:18]] # Stores |\Delta| in a list
axs[0].plot(*zip(*Delta[:18]),'k.',markersize = 10)
axs[0].set_title('$|\Delta|$ vs. $n$',fontsize=14)
axs[0].set_xlabel("$n$", fontsize=14)
axs[0].set_ylabel("$|\Delta|$", fontsize=14)
axs[0].set_xticks(np.arange(2,21,step=2))
axs[0].minorticks_on()
plt.text(-27, 17, '(a)', fontsize=14)

###### Plotting 1/|\Delta| vs. n #######

absDelta_inv = [(n[0],1/np.abs(n[1])) for n in Delta] # Stores 1/|\Delta| in a list
axs[1].plot(*zip(*absDelta_inv),'b.',markersize=8)
axs[1].set_title('$1/|\Delta|$ vs. $n$',fontsize=15)
axs[1].set_xlabel("$n$", fontsize=14)
axs[1].set_ylabel("$1/|\Delta|$", fontsize=14)
axs[1].set_xticks(np.arange(2,21,step=2))
plt.minorticks_on()
plt.text(-2.7, 17, '(b)', fontsize=14)

plt.savefig('absD_vs_n.pdf', format='pdf') # Saves figure
plt.show()


##### Plots BCCBi violation for n-MZI with fitting

from scipy.optimize import curve_fit

### Function to fit
def func(x,c,b):
    return 2**1.5-2 + c*(np.exp(-b*x)-np.exp(-2*b))

xdata = [n[0] for n in real[:9]] # n values for which the fitting is to be done
ydata = [n[1]-2*n[0]+2 for n in real[:9]] # D(n) values corresponding to n values
popt, pcov = curve_fit(func, xdata, ydata) # Fitting
# popt_tmsv =  optimal fitted values for a,b,c;  pcov_tmsv = covariance matrix of fit

xdata1 = [n[0] for n in real] # All n values to be plotted
ydata1 = [n[1]-2*n[0]+2 for n in real] # D(n) values corresponding to n values for plot
fitted_x = [func(x,popt[0],popt[1]) for x in np.linspace(2,20,100)] # Fitted function for n in [2,20)
plt.plot(xdata1,ydata1,'.',np.linspace(2,20,100),fitted_x,'r-') # Plot for violation and fitting
plt.legend(["Bound for 2-parameter optimisation","Bound for (2n-2)-parameter optimisation","Fitting for (2n-2)-parameter optimisation"], bbox_to_anchor=(0.2, 0.5))
plt.xlabel(r"$n$",fontsize=14)
plt.xticks(np.arange(2, 21, step=2))
plt.minorticks_on()
plt.ylabel(r"$\mathrm{D}(n)$",fontsize=14)
plt.title(r'$\mathrm{D}(n)$ vs. $n$',fontsize=15)
plt.savefig('max_viol_with_fitting_tri.pdf', format='pdf', bbox_inches="tight") # Saves figure

plt.show()
# popt



##### Plots TMSV violation with fitting

### Function to fit
def func_sq(x,a,c,b):
    return a + c*(np.exp(-b*x)) #-np.exp(-2*b)

xdata_tmsv = [n[0] for n in viol_squeez] # n values
ydata_tmsv = [n[1] for n in viol_squeez] # D(n) values
popt_tmsv, pcov_tmsv = curve_fit(func_sq, xdata_tmsv, ydata_tmsv) # Fitting

# popt_tmsv =  optimal fitted values for a,b,c;  pcov_tmsv = covariance matrix of fit

fitted_tmsvx = [func_sq(x,popt_tmsv[0],popt_tmsv[1],popt_tmsv[2]) for x in np.linspace(2,20,100)] # Fitted function for n in [2,20) 
plt.plot(np.linspace(2,20,100),fitted_tmsvx,'r-',xdata_tmsv,ydata_tmsv,'y*') # Plots violation and fitting
plt.legend(["Fitting for TMSV optimisation","Bound for TMSV optimisation"], bbox_to_anchor=(0.2, 0.5))
plt.xlabel(r"$n$",fontsize=14)
plt.xticks(np.arange(2, 21, step=2))
plt.minorticks_on()
plt.ylabel(r"$\mathrm{D}(n)$",fontsize=14)
plt.title(r'$\mathrm{D}(n)$ vs. $n$',fontsize=15)
plt.savefig('TMSV_with_fitting.pdf', format='pdf', bbox_inches="tight") # Saves figure
