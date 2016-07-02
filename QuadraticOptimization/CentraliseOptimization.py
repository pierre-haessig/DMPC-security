from __future__ import division, print_function
from cvxopt import matrix, solvers
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from tabulate import tabulate


solvers.options['show_progress'] = False
"""

Centralized Optimization  version v1.1

TODO = MPC in version v2.0

"""

## variable definition

# number of users
n = 6
i = np.arange(n)


# max energy in kW
Umax = 6
"""To be change to take more users"""
u_m = np.array([3, 3, 1, 3, 3, 3], dtype=float)
assert len(u_m) == n

# thermal resistance
"""beta can be taken into account to modify the importance of Rth"""

beta = 6 # if higher than 6, the optimization process considers than it isn't necessary to spend enery on room with high Rth
Rth = beta / u_m

# Exterior temperature
Text = 20

# Ideal temperature in degrees
"""To be change to take more users"""
T_id = np.array([21, 25, 21, 25, 21, 21], dtype=float)
assert len(T_id) == n

# Ideal energy
deltaT = (T_id - Text)
u_id = (T_id - Text) /Rth
#print(u_id)


# comfort factor
alpha = 100

#print(alpha)



def optim_central():
    """ Centralized optimization for power allocation
    """
    # Matrix definition
    P = matrix(2 * alpha * np.diag(Rth ** 2), tc='d')
    # print(P)

    q = matrix(1 - 2 * alpha * u_id * (Rth ** 2), tc='d')
    # print(q)

    G = matrix(np.vstack((np.ones(n), -np.identity(n), np.identity(n))), tc='d')
    # print(G)

    h = matrix(np.hstack((Umax, np.zeros(n), u_m)), tc='d')
    # print(h)


    # Resolution
    sol = solvers.qp(P, q, G, h)

    # Solution
    print(sol['x'])
    u_sol = np.asarray(sol['x'])
    print(u_sol)

    return u_sol

def print_sol(u_sol):
    table = tabulate([deltaT, u_m, u_id, u_sol], floatfmt=".6f")
    print(table)
    total = u_sol.sum()
    print(total)

def plot_sol(u_sol):
    """
    AFFICHAGE
    """

    # graph colors (hexadecimal RRGGBB)
    c = {
        'max':'#bbd5f0', # white blue
        'id':'#4582c2', # blue
        'opt':'#f9d600' # golden yellow
    }


    fig, ax1 = plt.subplots(1,1)


    ax1.bar(i, u_m,  label='$u_{max}$', # legend
            align='center',color=c['max'], linewidth=0)
    ax1.bar(i, u_id, label='u_id',
            align='center', color=c['id'], width=0.5, linewidth=0)
    ax1.plot(i, u_sol, 'D', label='u*',
             color=c['opt'], markersize=20)
    ax1.hlines(Umax/n, -1/2, n -1/2, linestyles = '--', label='Umax/n')
    ax1.hlines(u_sol.mean(), -1/2, n-1/2, linestyles = ':', label='<u*>')

    ax1.set(
        title='Power allocation = {:.3f}/{}'.format(u_sol.sum(), Umax),
        xticks=i,
        xlabel='sub system',
        ylabel='heating power (kW)',
    )

    ax1.legend(loc='upper left', markerscale=0.4)

    fig.tight_layout()

    fig.savefig('power_bars.png', dpi=200, bbox_inches='tight')
    fig.savefig('power_bars.pdf', bbox_inches='tight')
    plt.show()

    return fig, ax1



u_sol_c = optim_central()
print_sol(u_sol_c)
plot_sol(u_sol_c)


