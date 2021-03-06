# Sylvain Chatel - July 2016

from __future__ import division, print_function
from cvxopt import matrix, solvers
import numpy as np
import matplotlib.pyplot as plt

solvers.options['show_progress'] = False


def plot_T_tot(pb, pb2, T_opt, u_sol, T_opt2, u_sol2, x_min=0, x_max=24):
    """
        Returns the graph of temperature and power consumption for user i in two cases.

        keyword arguments:
        pb -- dictionary of the problem (nbr of users, time step, max resources, max admissible power, thermal resistance,
         Thermal capacity, vector of the init temperature, value of the exterior temperature, reference temperature,
          comfort factor, size of the prediction horizon)
        i -- number of the user to plot
        T_opt -- optimal temperature profile in case 1
        u_sol -- optimal power distribution in case 1
        lab1 -- label of case 1
        T_opt2 -- optimal temperature profile in case 2
        u_sol2 -- optimal power distribution in case 2
        lab2 -- label of case 2
        x_min -- axis 0 minimum value (default 0)
        x_max -- axis 0 maximum value (default 24)

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
    Returns the graph of temperature and power consumption for two users.

    keyword arguments:
    pb -- dictionary of the problem (nbr of users, time step, max resources, max admissible power, thermal resistance,
     Thermal capacity, vector of the init temperature, value of the exterior temperature, reference temperature,
      comfort factor, size of the prediction horizon)
    i -- number of the user to plot
    T_opt -- optimal temperature profile in user 1
    u_sol -- optimal power distribution in user 1
    lab1 -- label of user 1
    T_opt2 -- optimal temperature profile user 2
    u_sol2 -- optimal power distribution user 2
    lab2 -- label of user 2
    x_min -- axis 0 minimum value (default 6)
    x_max -- axis 0 maximum value (default 7)

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

"""""" """""" """""" """"""

def plot_alpha1(pb, _U, _DT, alpha_ratio):
    """
        Plots the optimal power distribution and the temperature as a parametric study of the comfort factor for one
        users.

        keyword arguments:
        pb -- dictionary of the problem (nbr of users, time step, max resources, max admissible power, thermal
        resistance, Thermal capacity, vector of the init temperature, value of the exterior temperature,
        reference temperature, comfort factor, size of the prediction horizon)
        _U -- optimal power distribution as a function of the comfort factor ratio
        -DT -- optimal temperature deviation as a function of the comfort factor ratio
        alpha_ratio -- vector of the comfort factor ratio
    """
    Umax = pb['Umax']
    alpha = pb['alpha']
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    ax1.plot(alpha_ratio, _U, 'r-+', label='1')

    ax1.hlines(Umax, alpha_ratio[0], alpha_ratio[-1], linestyles='--', label='')

    ax1.set(
        title=r'Parametric study of the comfort factor for $\alpha_=$ %s' % alpha[0],
        ylabel='$u^{*}$',
        xscale='log'
    )
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ax2.plot(alpha_ratio, _DT, 'r')

    ax2.set(
        xlabel=r'$ \alpha*x$',
        ylabel=r'$\Delta T = T -T_{id}$',
    )

    fig.tight_layout()

    plt.show()

    return fig, (ax1, ax2)

def plot_alpha2(pb, _U, _DT, alpha_ratio):
    """
        Plots the optimal power distribution and the temperature as a parametric study of the comfort factor for two
        users.

        keyword arguments:
        pb -- dictionary of the problem (nbr of users, time step, max resources, max admissible power, thermal
        resistance, Thermal capacity, vector of the init temperature, value of the exterior temperature,
        reference temperature, comfort factor, size of the prediction horizon)
        _U -- optimal power distribution as a function of the comfort factor ratio
        -DT -- optimal temperature deviation as a function of the comfort factor ratio
        alpha_ratio -- vector of the comfort factor ratio
    """
    Umax = pb['Umax']
    alpha = pb['alpha']


    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    ax1.plot(alpha_ratio, _U[0, :], 'g-+', label='1')
    ax1.plot(alpha_ratio, _U[1, :], 'r-+', label='2')

    ax1.hlines(Umax , alpha_ratio[0], alpha_ratio[-1], linestyles='--', label='')

    ax1.set(
        title=r'Parametric study of the comfort factor for $\alpha_1=$ %s' % alpha[0],
        xlabel=r'$ \alpha_{2} / \alpha_{1} $',
        ylabel='$u^{*}$',
        xscale='log'
    )
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ax2.plot(alpha_ratio, _DT[0, :], 'g')
    ax2.plot(alpha_ratio, _DT[1, :], 'r')

    ax2.set(
        xlabel=r'$ \alpha_{2} / \alpha_{1} $',
        ylabel=r'$\Delta T = T -T_{id}$',
    )

    fig.tight_layout()

    return fig, (ax1, ax2)

def plot_alpha(pb, _U, _DT, alpha_ratio):
    """
        Plots the optimal power distribution and the temperature as a parametric study of the comfort factor for
        more than two users.

        keyword arguments:
        pb -- dictionary of the problem (nbr of users, time step, max resources, max admissible power, thermal
        resistance, Thermal capacity, vector of the init temperature, value of the exterior temperature, reference
        temperature, comfort factor, size of the prediction horizon)
        _U -- optimal power distribution as a function of the comfort factor ratio
        -DT -- optimal temperature deviation as a function of the comfort factor ratio
        alpha_ratio -- vector of the comfort factor ratio
    """
    Umax = pb['Umax']
    m = pb['m']
    alpha = pb['alpha']

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    for k in range(m - 1):
        ax1.plot(alpha_ratio, _U[k, :], 'g', label='users')
    ax1.plot(alpha_ratio, _U[m - 1, :], 'r', label='user2 cheater')

    ax1.hlines(Umax / m, alpha_ratio[0], alpha_ratio[-1], linestyles='--', label='')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.set(
        title=r'Parametric study of the comfort factor for $\alpha=$ %s and Umax = %s.' % (alpha[0], Umax),
        xlabel=r'$ \alpha_{%s} / \alpha_{1} $' % m,
        ylabel='$u^{*}$',
        xscale='log'
    )
    for k in range(m - 1):
        ax2.plot(alpha_ratio, _DT[k, :], 'g')
    ax2.plot(alpha_ratio, _DT[m - 1, :], 'r')

    ax2.set(
        xlabel=r'$ \alpha_{%s} / \alpha_{1} $' % m,
        ylabel=r'$\Delta T = T -T_{id}$',
    )

    fig.tight_layout()

    plt.show()

    return fig, (ax1, ax2)

"""""" """""" """""" """"""

def plot_Tbc(pb, _U, _DT, T_sup, ):
    """
        Plots the optimal power distribution and the temperature deviation as a function of the broadcasted temperature.

        keyword arguments:
     pb -- dictionary of the problem (nbr of users, time step, max resources, max admissible power, thermal resistance,
         Thermal capacity, vector of the init temperature, value of the exterior temperature, reference temperature,
          comfort factor, size of the prediction horizon)
     _U -- optimal power distribution when the broadcasted temperature differ from the wanted temperature
     _DT -- optimal temperature deviation when the broadcasted temperature differ from the wanted temperature
     T_sup -- broadcasted temperature

    """

    Umax = pb['Umax']
    m = pb['m']
    alpha = pb['alpha']
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    ax1.plot(T_sup, _U[0, :], 'r-+', label='user1 cheater')
    ax1.plot(T_sup, _U[1, :], 'g-+', label='user2')
    ax1.plot(T_sup, _U[2, :], 'g-+', label='user3')

    ax1.hlines(Umax, T_sup[0], T_sup[-1], linestyles='--', label='')

    ax1.set(
        title=r'Parametric study of the broadcasted temperature for $\alpha_1=$ %s' % alpha[0],
        xlabel=r'$ T_{sup} $',
        ylabel='$u^{*}$',
    )
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ax2.plot(T_sup, _DT[0, :], 'r')
    ax2.plot(T_sup, _DT[1, :], 'g')
    ax2.plot(T_sup, _DT[2, :], 'g')
    ax2.plot(T_sup, T_sup,'--')

    ax2.set(
        xlabel=r'$ T_{sup} $',
        ylabel=r'$\Delta T = T -T_{id}$',
    )

    fig.tight_layout()

    return fig, (ax1, ax2)

"""""" """""" """""" """"""

def plot_Rth(pb, _U, _DT, varRth, Rth, figsize=(9, 6)):
    """
        Plots the optimal power distribution and the temperature deviation  as a parametric study of the thermal
         resistance.

        keyword arguments:
        pb -- dictionary of the problem (nbr of users, time step, max resources, max admissible power, thermal
         resistance, Thermal capacity, vector of the init temperature, value of the exterior temperature, reference
         temperature, comfort factor, size of the prediction horizon)
        _U -- optimal power distribution when the thermal resistance is the parameter
        _DT -- optimal temperature deviation when the thermal resistance is the parameter
        varRth -- vector of the parametric thermal resistance
        Rth -- real thermal resistance

    """

    Umax = pb['Umax']
    m = pb['m']
    alpha = pb['alpha']
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    ax1.plot(varRth, _U[0, :], 'r', label='use1 cheater')
    ax1.plot(varRth, _U[1, :], 'g', label='user2')
    ax1.plot(varRth, _U[2, :], 'b--', label='user3')

    ax1.hlines(Umax, varRth[0], varRth[-1], linestyles='--', label='')

    ax1.set(
        title=r'Parametric study of the broacasted Rth for $Rth_{real}$= %s' % Rth[0],
        xlabel=r'$ T_{sup} $',
        ylabel='$u^{*}$',
        #xscale='log',
    )
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ax2.plot(varRth, _DT[0, :], 'r')
    ax2.plot(varRth, _DT[1, :], 'g')
    ax2.plot(varRth, _DT[2, :], 'b--')

    ax2.set(
        xlabel=r'$Rth$',
        ylabel=r'$\Delta T = T -T_{id}$',
    )

    fig.tight_layout()

    return fig, (ax1, ax2)