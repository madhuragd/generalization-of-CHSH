###########   Obtaining and storing results   ############

from util2 import optimize1
import os
from _pickle import dump,load


Min = 2 # Min. n for doing above calculations
Max = 4 # Max. n for doing above calculations
num_opti = 100 # No. of times to optimize
typ = 'arithmetic' # Type of calculation


if __name__=="__main__":
    for n in range(Min,Max):
        o = optimize1(n,num_opti,typ)
        if 'max_chsh_test.pi' in os.listdir(): # Stores in filename "max_chsh_eig.pi"
            with open('max_chsh_test.pi','rb') as f:
                d = load(f)
                if n in d and o['fun'] < d[n]['fun'] or n not in d:
                # Stores in file if max_viol(n) was not optimized previously 
                # or optimization result is better than previous iteration.
                    d[n] = o
                    with open('max_chsh_test.pi','wb') as f:
                        dump(d,f)
        else:
        # In the first run, generates the file max_chsh_eig.pi with
        # a initial run of optimization.
            d = {n:o}
            with open('max_chsh_test.pi','wb') as f:
                dump(d,f)