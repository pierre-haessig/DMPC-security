from __future__ import division, print_function
from cvxopt import matrix, solvers
import numpy as np
import matplotlib.pyplot as plt

solvers.options['show_progress'] = False


def plot_T_tot(pb, pb2, T_opt, u_sol, T_opt2, u_sol2, x_min=0, x_max=24):
    """
    DynamicOpt.plot_T(object)
    Parameters : dictionary of the variables, number of the user, the vector of all optimal temperature
    from DynamicOpt.get_temp_op_OL and the vector of optimal power.
    returns : graph of the ideal temperature and the optimum temperature.
    """
    T_id_pred = pb['T_id_pred']
    dt = pb['dt']
    m = pb['m']
    alpha_eq = pb['alpha']
    alpha = pb2['alpha']
    N_sim = pb['N_sim']
    t = np.arange(N_sim) * dt

    fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, figsize=(9, 6))

    ax1.plot(t, [T_id_pred[j] for j in range(N_sim)], 'k:',
             label='T_id')
    ax1.plot(t, T_opt[:, 0], 'b', label=r'$\alpha_{eq}=$ %s' % alpha_eq[0])
    ax1.plot(t, T_opt2[:, 0], 'g', label=r'$\alpha_1=$ %s' % alpha[0])
    ax1.plot(t, T_opt2[:, 1], 'r', label=r'$\alpha_2=$ %s' % alpha[1])
    ax1.legend()


    ax2.plot(t, u_sol[:, 0], 'b', label='$U_{eq}$')
    ax2.plot(t, u_sol2[:, 0], 'g', label='$U_1$')
    ax2.plot(t, u_sol2[:, 1], 'r', label='$U_2$')
    ax2.legend()

    ax1.set(
        ylabel=u'T ($^{\circ}$C)'
    )

    ax1.set_ylim([min(T_id_pred)-1, max(T_id_pred)+1])
    ax1.set_xlim([x_min, x_max])
    ax2.set(
        xlabel='t (h)',
        ylabel=u'Power kW'
    )



    fig.tight_layout()

    return fig, (ax1, ax2)

def plot_2usr(pb, T_opt, u_sol, i, lab1, T_opt2, u_sol2, z, lab2, x_min=6, x_max=7):
    """
    DynamicOpt.plot_T(object)
    Parameters : dictionary of the variables, number of the user, the vector of all optimal temperature
    from DynamicOpt.optim_central_OL and the vector of optimal power.
    returns : graph of the ideal temperature and the optimum temperature for usr i.
    """
    T_id_pred = pb['T_id_pred']
    dt = pb['dt']
    m = pb['m']

    N_sim = pb['N_sim']
    t = np.arange(N_sim) * dt

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9, 6))

    ax1.plot(t, [T_id_pred[j] for j in range(N_sim)], 'k:', label='T_id')
    ax1.plot(t, [T_id_pred[N_sim + j] for j in range(N_sim)], 'k:')
    ax1.plot(t, T_opt[:, i], 'r', label=lab1)
    ax1.plot(t, T_opt2[:, z], 'g', label=lab2)
    ax1.legend()

    ax2.plot(t, u_sol[:, i], 'r', label=lab1)
    ax2.plot(t, u_sol2[:, z], 'g', label=lab2)
    ax2.legend()

    ax1.set(
        ylabel=u'Temperature ($^{\circ}$C)'

    )
    ax1.set_ylim([min(T_id_pred) - 1, max(T_id_pred) + 1])
    ax1.set_xlim([x_min, x_max])
    ax2.set(
        xlabel='t (h)',
        ylabel='Power (kW)'
    )

    fig.tight_layout()
    plt.show()
    return fig, (ax1, ax2)

