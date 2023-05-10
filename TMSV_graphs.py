from _pickle import dump,load
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


# To plot MZI+TMSV max violation as a funciton of squeezing param (r) for different n, n \in [2,8)

with open("tmsv_r.pi",'rb') as f: # Loads data
    d = load(f)
    
r_lin = np.linspace(0,3,31) # Defines r
n = range(2,8) # Defines n
funs = [[-d[i][j]['fun']-2*i+2 for j in range(len(r_lin))] for i in n] # Violations D(n)


# Plot details
for i in range(len(n)):
    plt.plot(r_lin,funs[i],'.')
    
plt.xlabel(r'$r$',fontsize=14)
plt.title(r'$D(n)$ vs. $r$',fontsize=15)
# plt.savefig('tmsv_viol_r.pdf', format='pdf',bbox_inches="tight") # Saves figure
plt.show()



##### Plots TMSV violation with fitting

with open('max_chsh_tmsv.pi','rb') as f: # Loads data
    res_list_TMSV = load(f)

squeez = [(k[0],-k[1]['fun']) for k in res_list_TMSV[:18]] # Maximal bound
viol_squeez = [(n[0],(n[1]-2*n[0]+2)) for n in squeez] # Violation D(n)
    
    
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
# plt.savefig('TMSV_with_fitting.pdf', format='pdf', bbox_inches="tight") # Saves figure
plt.show()