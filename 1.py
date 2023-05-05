
from _pickle import load
import matplotlib.pyplot as plt

"""
with open('max_chsh_ecs.pi','rb') as f:
	d = load(f)

for i in d:
	print(i, d[i].fun)
"""

with open('max_chsh_ecs_3.pi','rb') as f:
	d = load(f)

X = sorted([n for n in d.keys()])

Y = [abs(d[n].x[0]) for n in X]
plt.plot(X,Y,marker='*')
plt.plot(X[0],Y[0],marker='*',color='k')
plt.show()

Y = [d[n].x[1] for n in X]
Y = [y if y>1 else 1/y for y in Y]
plt.plot(X,Y,marker='*')
plt.plot(X[0],Y[0],marker='*',color='k')
plt.show()

Y = [-d[n]['fun']-(2*(n-1)) for n in X]
plt.plot(X,Y,marker='*')
plt.plot(X[0],Y[0],marker='*',color='k')
plt.show()

for n in X:
	print(n,-d[n]['fun']-2*(n-1))

"""
n=14
x = d[n].x
x = x[:2*n+2] + 1j*x[2*n+2:]
a,a1 = x[:2]
x = x[2:]
xA = x[:n]
xB = x[n:]
plt.plot(xA.imag,label='Lab X, %s settings'%(n,),marker='*')
plt.plot(xB.imag,label='Lab Y, %s settings'%(n,),marker='*')
plt.plot(xA.imag[0],marker='*',color='k')
plt.plot(xB.imag[0],marker='*',color='k')
#print(n,a,a1,x)

plt.legend()
plt.show()
"""
