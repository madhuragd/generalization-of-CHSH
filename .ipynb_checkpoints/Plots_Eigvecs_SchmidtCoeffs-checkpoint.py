#!/usr/bin/env python
# coding: utf-8

# In[6]:


## Code for plotting Eigenvectors and Schmidt Coefficients.

import matplotlib.pyplot as plt
import numpy as np
from _pickle import load
import math

###### Retrieving calculated values from relevant file #########

# path = "C:\\Users\\user\\OneDrive\\Documents\\Confocal_acq\\Test\\"
with open('max_chsh_states.pi','rb') as f:
     opti_evec = load(f)
        
mat_delta = [d[3].reshape(d[0],d[0]) for d in opti_evec] # In $\ket{\beta_i}\otimes\ket{\gamma_i}$ basis
mat_gamma = [d[2].reshape(d[0],d[0]) for d in opti_evec] # In orthonormal basis
mat_svd = [np.linalg.svd(d) for d in mat_gamma] # Singular value decomposition of eigenvectors in orthonormal basis        
        
#####################     Plot Details    ##########################        
        
fig = plt.figure(figsize=(15,7)) 
axs = [fig.add_subplot(2,4,i+1) for i in range(8)] 

### Plots Eigenvectors  in $\ket{\beta_i}\otimes\ket{\gamma_i}$ basis and Schmidt coeffs. for 4 ns, 
### starting from n = 3 at intervals of 3.
n = 3 # Start n
m = n-2 # Index corresponding to n in the "mat_" lists.

fig.subplots_adjust(wspace=0.4, hspace=0.4)
min_cb = [np.amin(mat_delta[m+3*i]) for i in range(4)] # Chooses min. of colorbar
max_cb = [np.amax(mat_delta[m+3*i]) for i in range(4)] # Chooses max. of colorbar
im = axs[0].imshow(mat_delta[m+3*0], vmin = min(min_cb), vmax = max(max_cb)) # Sets colorbar image reference

for i in range(4):
  axs[i].imshow(mat_delta[m+3*i]) # Plots eigenvectors in $\ket{\beta_i}\otimes\ket{\gamma_i}$ basis
  axs[i].set_title(r'$n$ = %s'%(n+3*i),fontsize=15)
  axs[i].set_xlabel("Column $(j)$", fontsize=14)
  axs[i].set_ylabel("Row $(i)$", fontsize=14)
    
  axs[i+4].bar(range(len(mat_svd[m+3*i][1])), mat_svd[m+3*i][1], color ='maroon',width = 0.4) # Plots bar graph of Schmidt Coefficients
  axs[i+4].set_xlim(-0.5,len(mat_svd[m+3*i][1]))
  axs[i+4].set_xticks(np.arange(-0.,len(mat_svd[m+3*i][1]),step=1))
  axs[i+4].set_ylabel(r"$\lambda_i$", fontsize=14)
  axs[i+4].set_xlabel(r"Index $(i)$", fontsize=14)

############ Figure details ##############
plt.text(-57.5, 1.75, '(a)', fontsize=15)
plt.text(-38.5, 1.75, '(b)', fontsize=15)
plt.text(-21, 1.75, '(c)', fontsize=15)
plt.text(-4, 1.75, '(d)', fontsize=15)
plt.text(-57.5, 0.7, '(e)', fontsize=15)
plt.text(-38.5, 0.7, '(f)', fontsize=15)
plt.text(-21, 0.7, '(g)', fontsize=15)
plt.text(-4, 0.7, '(h)', fontsize=15)
cbar_ax = fig.add_axes([0.91, 0.565, 0.015, 0.315])
cbar = fig.colorbar(im, cax=cbar_ax)

plt.savefig('max_eigvec_schmidt_coeff_plots.pdf', format='pdf', bbox_inches="tight") # Saves figure

