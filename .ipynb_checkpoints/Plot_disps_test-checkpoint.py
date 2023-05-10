from util import parse_data, gen_plot
import matplotlib.pyplot as plt


# For general displacements:

betas,gammas = parse_data('Eig_general.pi','general')
gen_plot(betas,gammas)
plt.savefig("betas_gammas_gen.pdf",format='pdf')


# For real displacements:

betas,gammas = parse_data('Eig_real.pi','real')
gen_plot(betas[1:],gammas[1:])
plt.savefig("betas_gammas_real.pdf",format='pdf')
