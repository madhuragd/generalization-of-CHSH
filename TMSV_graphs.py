from _pickle import dump,load
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


# To plot MZI+TMSV max violation as a function of squeezing param (r) for different n, n \in [2,8)

with open("tmsv_r.pi",'rb') as f: # Loads data
    d = load(f)
    
r_lin = np.linspace(0,3,31) # Defines r
funs = {n:[-d[n][j]['fun']-2*n+2 for j in range(len(r_lin))] for n in range(2,8)} # Violations D(n)

# Plot details
for i in range(2,8):
    plt.plot(r_lin,funs[i],'.')
    
plt.xlabel(r'$r$',fontsize=14)
plt.title(r'$D(n)$ vs. $r$',fontsize=15)
plt.savefig('tmsv_viol_r.pdf', format='pdf',bbox_inches="tight") # Saves figure
plt.show()


##### Plots TMSV violation with fitting

with open('test_max_viol_tmsv.pi','rb') as f: # Loads data
	res_list_TMSV = load(f)

viol_squeez = [(n,-v['fun']-2*n+2) for n,v in res_list_TMSV.items()] # Violation D(n)
    
### Function to fit
def func_sq(x,a,c,b):
    return a + c*(np.exp(-b*x)) #-np.exp(-2*b)

xdata_tmsv,ydata_tmsv = zip(*viol_squeez)
popt_tmsv, pcov_tmsv = curve_fit(func_sq, xdata_tmsv,ydata_tmsv) # Fitting
print(popt_tmsv, pcov_tmsv)	# popt_tmsv =  optimal fitted values for a,b,c;  pcov_tmsv = covariance matrix of fit

fitted_tmsvx = [func_sq(x,popt_tmsv[0],popt_tmsv[1],popt_tmsv[2]) for x in np.linspace(2,20,100)] # Fitted function for n in [2,20) 
plt.plot(np.linspace(2,20,100),fitted_tmsvx,'r-',xdata_tmsv,ydata_tmsv,'y*') # Plots violation and fitting
plt.legend(["Fitting for TMSV optimisation","Bound for TMSV optimisation"], bbox_to_anchor=(0.2, 0.5))
plt.xlabel(r"$n$",fontsize=14)
plt.xticks(np.arange(2, 21, step=2))
plt.minorticks_on()
plt.ylabel(r"$\mathrm{D}(n)$",fontsize=14)
plt.title(r'$\mathrm{D}(n)$ vs. $n$',fontsize=15)
plt.savefig('TMSV_with_fitting.pdf', format='pdf', bbox_inches="tight") # Saves figure
plt.show()


##### Plots sequences of displacements for both parties

n=19
x = res_list_TMSV[n].x[1:]
beta = x[:n]
gamma = x[n:]
plt.figure().set_figheight(3)
plt.plot(beta,'*',label=r'$\beta_i$')
plt.plot(gamma,'*',label=r'$\gamma_i$')
plt.xlabel(r"$i$",fontsize=14)
plt.ylabel(r"Displacements $\beta_i$, $\gamma_i$",fontsize=14)
plt.xticks(range(n))
plt.minorticks_off()
plt.title(r'Optimal displacements for TMSV state',fontsize=15)
plt.legend(loc='upper left')
plt.savefig('max_viol_displacements_TMSV.pdf', format='pdf', bbox_inches="tight") # Saves figure
plt.show()

