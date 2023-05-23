
from _pickle import load
import matplotlib.pyplot as plt

with open('test_max_viol_ecs.pi','rb') as f:
	d = load(f)


X = sorted([n for n in d.keys()])

plt.figure().set_figheight(3)
Y = [abs(d[n].x[0]) for n in X]
plt.plot(X,Y,'*',label=r"$\alpha_n$")
Y = [d[n].x[1] for n in X]
Y = [y if y>1 else 1/y for y in Y]
plt.plot(X,Y,'*',label=r"$a_n$")
plt.xlabel(r"$n$",fontsize=14)
plt.xticks(range(2, 20, 1))
plt.minorticks_off()
plt.ylabel(r"",fontsize=14)
plt.title(r'Optimal EC state parameters',fontsize=15)
plt.legend()
plt.savefig('max_viol_parameters_ECS.pdf', format='pdf', bbox_inches="tight")
plt.show()

plt.figure().set_figheight(3)
Y = [-d[n]['fun']-(2*(n-1)) for n in X]
plt.plot(X,Y,'*')
plt.xlabel(r"$n$",fontsize=14)
plt.xticks(range(2, 20, 1))
plt.minorticks_off()
plt.ylabel(r"$D(n)$",fontsize=14)
plt.title(r'Maximal violation for EC states',fontsize=15)
plt.savefig('max_viol_ECS.pdf', format='pdf', bbox_inches="tight")

plt.show()

for n in X:
	print(n,-d[n]['fun']-2*(n-1),d[n].x.shape)


n=15
x = d[n].x
a,a1 = x[:2]
x = x[2:]
xA = x[:n]
xB = x[n:]
plt.figure().set_figheight(3)
plt.plot(range(1,n+1), xA, '*', label=r'$\beta_i$, %s settings'%(n,))
plt.plot(range(1,n+1), xB, '*', label=r'$\gamma_i$, %s settings'%(n,))
plt.xlabel(r"$i$",fontsize=14)
plt.xticks(range(1, n+1, 1))
plt.minorticks_off()
plt.ylabel(r"Displacements $\beta_i$, $\gamma_i$",fontsize=14)
plt.title(r'Optimal displacements for EC state',fontsize=15)
plt.legend()
plt.savefig('max_viol_displacements_ECS.pdf', format='pdf', bbox_inches="tight")
plt.show()

