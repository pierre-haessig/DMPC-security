from __future__ import division, print_function
from cvxopt import matrix, solvers
import numpy as np
import StaticParam as SPar
import StaticOptimization as SO
import matplotlib.pyplot as plt

def plot_step1(pb, range, pas, e):
    """
        StaticOptimization.plot_sol(param, range, precision, error )
        Parameters : dictionary of parameters (Rth, Text, T_id, Umax, u_m, alpha), range of step, precision, error
        Returns : out graph of the distributed optimization total power allocation and the centralized optimization
        total power allocation.
    """
    fig, ax1 = plt.subplots(1,1)
    step = np.arange(float(range)/float(pas))*pas
    U = []
    u_cent = optim_central(pb).sum()
    for x in np.nditer(step):
        U.append(optim_decen(pb, x, e).sum())
    ax1.plot(step, U, label='$ PA^{*}$ distributed')
    ax1.set(
        ylabel='$ u^{*} $',
        ylim=(u_cent - 5e-2, u_cent + 5e-2)
    )
    plt.show()
    fig.savefig('step_opt.png', dpi=200, bbox_inches='tight')
    fig.savefig('step_opt.pdf', bbox_inches='tight')
    return fig, (ax1)

def plot_step(pb, step_min, step_max, nbr , e):
    """
        StaticOptimization.plot_sol(param, range, precision, error )
        Parameters : dictionary of parameters (Rth, Text, T_id, Umax, u_m, alpha), range of step, precision, error
        Returns : out graph of the distributed optimization total power allocation and the centralized optimization
        total power allocation, the graph of the evolution of the Lagrangian multiplier and the number of iteration
        regarding the step.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3,1)
    fig.tight_layout()
    step = np.logspace(np.log10(step_min), np.log10(step_max), nbr)
    U = []
    k_val = []
    L_val = []

    for x in np.nditer(step):
        u_sol, L, k = optim_decen(pb, x, e)
        U.append(u_sol.sum()-pb['Umax'])
        L_val.append(L)
        k_val.append(k)

    ax1.plot(step, U, '-+', label='$ PA^{*}$ distributed')

    ax2.plot(step, L_val, '-+')

    ax3.plot(step, k_val)

    ax1.set(
        ylabel='$ u^{*} $',
        title = 'mean to Umax as a function of the step of the Uzawa iteration'
    )

    ax2.set(
        ylabel='$ L $',
        title='Lagrangian multiplier as a function of the step of the Uzawa iteration'
    )

    ax3.set(
        ylabel='number of iteration',
        title='number of iteration as a function of the step of the Uzawa iteration'
    )

    plt.show()
    fig.savefig('step_opt.png', dpi=200, bbox_inches='tight')
    fig.savefig('step_opt.pdf', bbox_inches='tight')
    return fig, (ax1, ax2, ax3)