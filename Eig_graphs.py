import matplotlib.pyplot as plt
from _pickle import load
from scipy.optimize import curve_fit
import numpy as np

# Function to parse data for plotting displacements    

def parse_data(filename):
	with open(filename,'rb') as f:
		res_list = load(f)
		n,v = zip(*[(i[0],i[2]) for i in res_list])
		gammas = [v[i][:n[i]] + 1j * v[i][n[i]:2*n[i]] for i in range(len(v))] 		# Generates vectors for gammas for each n
		betas = [v[i][2*n[i]:3*n[i]] + 1j * v[i][3*n[i]:] for i in range(len(v))]	# Generates vectors for betas for each n
	return betas,gammas # Returns the displacements for each party


def gen_plot(betas,gammas):
    fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(16, 8))
    fig.subplots_adjust(hspace=0.3,wspace=0.3)
    for i in range(3):
        axs[0][i].plot(betas[i].real,betas[i].imag,'.',label=r"$\beta_i$")
        axs[0][i].plot(gammas[i].real,gammas[i].imag,'.',label=r"$\gamma_i$")
        axs[0][i].set_title(r'$n$ = %s' %(i+3), fontsize = 15)
        axs[0][i].set_xlabel("Real Axis", fontsize=14)
        axs[0][i].set_ylabel("Imaginary Axis", fontsize=14)
        axs[1][i].plot(betas[i+3].real,betas[i+3].imag,'.',gammas[i+3].real,gammas[i+3].imag,'.')
        axs[1][i].set_title(r'$n$ = %s' %(i+6), fontsize = 15)
        axs[1][i].set_xlabel("Real Axis", fontsize=14)
        axs[1][i].set_ylabel("Imaginary Axis", fontsize=14)
    axs[0][2].legend(bbox_to_anchor=(1, 1.02))
    # plt.legend()
    fig
    return fig


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
print(popt, pcov)
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

betas,gammas = parse_data('Eig_general.pi')
gen_plot(betas,gammas)
plt.savefig("betas_gammas_gen.pdf",format='pdf')
plt.show()


# popt
