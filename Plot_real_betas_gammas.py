import matplotlib.pyplot as plt
from _pickle import load, dump
import numpy as np

with open('n3to8_real.pi','rb') as f:
    results = load(f)
    res_list = [(k, v) for k, v in results.items()]
    
# res_list[6]
fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(16, 8))
fig.subplots_adjust(hspace=0.3,wspace=0.3)
for j in range(1,len(res_list)): #1,len(res_list)
  n = res_list[j][0]
  x = res_list[j][1].x

## Retrieving betas and gammas from optimized x
  gamma = np.zeros(n,dtype=np.clongdouble)
  beta = np.zeros(n,dtype=np.clongdouble)
  gamma[1:] = x[:n-1] #+ 1j * x[n:2*n]
  beta[1:] = x[n-1:] #+ 1j * x[3*n:]

## Separating real and imaginary parts for betas and gammas
  x1 = [i.real for i in gamma]
  y1 = [i.imag for i in gamma]
  x2 = [i.real for i in beta]
  y2 = [i.imag for i in beta]

  if j < 4:
      axs[0][j-1].scatter(x1,y1,label=r'$\beta_i$')
      axs[0][j-1].scatter(x2,y2,label=r'$\gamma_i$')
      axs[0][j-1].set_title(r'$n$ = %s' %n, fontsize = 15)
      axs[0][j-1].set_xlabel("Real Axis", fontsize=14)
      axs[0][j-1].set_ylabel("Imaginary Axis", fontsize=14)
      
  else:
      axs[1][j-4].scatter(x1,y1,label=r'$\beta_i$')
      axs[1][j-4].scatter(x2,y2,label=r'$\gamma_i$')
      axs[1][j-4].set_title(r'$n$ = %s' %n, fontsize = 15)
      axs[1][j-4].set_xlabel("Real Axis", fontsize=14)
      axs[1][j-4].set_ylabel("Imaginary Axis", fontsize=14)
      # plt.show(fig)
  
  plt.legend()
  axs[0][2].legend(bbox_to_anchor=(1, 1.02))
  fig.show()
  
  # plt.savefig('real_betas_gammas.pdf', format='pdf', bbox_inches="tight") # Saves figure