from util import parse_data, gen_plot
import matplotlib.pyplot as plt
from _pickle import load
from scipy.optimize import curve_fit
import numpy as np


##### Plots BCCBi violation for n-MZI with fitting


with open('Eig_second_stage.pi','rb') as f:
    results_MZI = load(f)
    res_list_MZI = [(k, v) for k, v in results_MZI.items()]
    
real = [(n[0],-n[1]['fun']) for n in res_list_MZI] 
viol_real = [(n[0],(n[1]-2*n[0]+2)) for n in real] 

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
plt.savefig('max_viol_with_fitting.pdf', format='pdf', bbox_inches="tight") # Saves figure

plt.show()



# For general displacements:

betas,gammas = parse_data('Eig_general.pi','general')
gen_plot(betas,gammas)
plt.savefig("betas_gammas_gen.pdf",format='pdf')


    

# popt