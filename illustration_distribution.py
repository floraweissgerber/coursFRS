import numpy as np
import matplotlib.pyplot as plt
plt.close('all')


p_rayleigh = np.linspace(0,10,500)
p_gamma = np.linspace(0,20,500)
list_color = [[1,0,0], [1,0.5, 0], [0,0.5,0], [0,0.5,1], [0.5, 0, 1]]
vec_mu = [0.5, 1,2,3,4]

fig_rayleigh, ax_rayleigh = plt.subplots()
rayleigh_max = 1.8
fig_gamma, ax_gamma = plt.subplots()
gamma_max = 4.1
linewidth = 2

for ind_mu in range(len(vec_mu)):
    mu = vec_mu[ind_mu]
    rayleigh = (2*p_rayleigh/mu**2)*np.exp(-p_rayleigh**2/mu**2)
    ax_rayleigh.plot(p_rayleigh, rayleigh, '-', color=list_color[ind_mu], linewidth=linewidth, label = str(mu) )
    ax_rayleigh.plot([mu, mu ], [0, rayleigh_max], '--', color=list_color[ind_mu], linewidth=linewidth )
   
    gamma = (1/mu**2)*np.exp(-p_gamma/mu**2)
    ax_gamma.plot(p_gamma, gamma, '-', color=list_color[ind_mu], linewidth=linewidth, label = str(mu**2) )
    ax_gamma.plot([mu**2, mu**2 ], [0, gamma_max], '--', color=list_color[ind_mu], linewidth=linewidth )
   
ax_rayleigh.set_ylim([0, rayleigh_max])
ax_rayleigh.set_xlim([0, 10])
ax_rayleigh.legend()
ax_gamma.set_ylim([0, gamma_max])
ax_gamma.set_xlim([0, 5])
ax_gamma.legend()

fig_rayleigh.savefig('rayleigh.png', bbox_inches='tight')
fig_rayleigh.savefig('rayleigh.svg', bbox_inches='tight')
fig_gamma.savefig('gamma.png', bbox_inches='tight')
fig_gamma.savefig('gamma.svg', bbox_inches='tight')

plt.show()


