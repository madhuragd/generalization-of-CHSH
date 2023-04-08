
from util2 import optimize1
import os
from _pickle import dump,load


min_n = 2
max_n = 4
num_opti = 10 # No. of times to optimize
typ = 'general' # Type of calculation

if __name__=="__main__":
    for n in range(min_n,max_n):
        o = optimize1(n,num_opti,typ)
        if 'max_chsh_gen.pi' in os.listdir(): # Stores in filename "max_chsh_eig.pi"
            with open('max_chsh_gen.pi','rb') as f:
                d = load(f)
                if n in d and o['fun'] < d[n]['fun'] or n not in d:
                # Stores in file if max_viol(n) was not optimized previously 
                # or optimization result is better than previous iteration.
                    d[n] = o
                    with open('max_chsh_gen.pi','wb') as f:
                        dump(d,f)
        else:
        # In the first run, generates the file max_chsh_eig.pi with
        # a initial run of optimization.
            d = {n:o}
            with open('max_chsh_gen.pi','wb') as f:
                dump(d,f)