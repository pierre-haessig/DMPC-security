# Sylvain Chatel - July 2016

from __future__ import division, print_function
from cvxopt import matrix, solvers
import numpy as np
import matplotlib.pyplot as plt

from tabulate import tabulate
solvers.options['show_progress'] = False

"""
Centralized Optimization  version v4.0
"""

"""""" """""" """""" """"""
def optim_central(pb):
    """
        Returns the power distribution u_sol of the central static QP.

        keyword arguments:
        pb -- dictionary of the problem (nbr of users, time step, max resources, max admissible power, thermal resistance,
         Thermal capacity, vector of the init temperature, value of the exterior temperature, reference temperature,
          comfort factor, size of the prediction horizon)
    """
    Rth = pb['Rth']
    Text = pb['Text']
    T_id = pb['T_id']
    Umax = pb['Umax']
    u_m = pb['u_m']
    alpha = pb['alpha']

    u_id = (T_id - Text) /Rth

    print(np.shape(alpha*u_id))
    print(np.shape(2*Rth**2))

    # Matrix definition
    P = matrix(2 * np.diag(alpha*Rth ** 2), tc='d')
    q = matrix(1 - 2*alpha*u_id * (Rth ** 2), tc='d')
    G = matrix(np.vstack((np.ones(len(Rth)), -np.identity(len(Rth)), np.identity(len(Rth)))), tc='d')
    h = matrix(np.hstack((Umax, np.zeros(len(Rth)), u_m)), tc='d')

    # Resolution
    sol = solvers.qp(P, q, G, h)

    # Solution
    u_sol = np.asarray(sol['x']).T[0]

    return u_sol,

def optim_decen(pb, step, e, k_max=1000, count=True):
    """
    Returns power distribution u_sol, the lagrangian multiplier, the number of Uzawa iterations and
     the value of the cost function in the static distributed optimization

    keyword arguments:
    pb -- dictionary of the problem (nbr of users, time step, max resources, max admissible power, thermal resistance,
     Thermal capacity, vector of the init temperature, value of the exterior temperature, reference temperature,
      comfort factor, size of the prediction horizon)
    step -- step of the Lagrangian in the Uzawa method
    e -- value of the maxim gap tolerable
    k_max -- max number of iteration in the Uzawa method (default 1000)
    count -- boolean to print the count of breaks (default True)
    """

    Rth = pb['Rth']
    Text = pb['Text']
    T_id = pb['T_id']
    Umax = pb['Umax']
    u_m = pb['u_m']
    alpha = pb['alpha']

    u_id = (T_id - Text) / Rth

    L = 0
    m = len(Rth)
    u_sol = np.zeros(m)
    k_uzw = 0

    for k in range(k_max):
        assert L >= 0, "u_id can be reached for all users"
        for j in range(m):

            Pj = matrix(2 * alpha[j] * Rth[j] ** 2, tc='d')
            qj = matrix(1 - 2 * alpha[j] * Rth[j]**2 * u_id[j] + L, tc='d')
            Gj = matrix([-1, 1], tc='d')
            hj = matrix([0, u_m[j]], tc='d')

            solj = solvers.qp(Pj, qj, Gj, hj)
            u_sol[j] = solj['x'][0]

        k_uzw = k

        L = L + step * (u_sol.sum() - Umax)

        if u_sol.sum() - Umax < e:
            if count == True:
                print('break at %s.' % k)
            break
    cost = u_sol.sum()
    deltaT_opt = Rth * (u_sol.T - u_id)
    J_u = cost + alpha.dot((deltaT_opt**2))

    return u_sol, L, k_uzw, J_u

def optim_CHT_decen(pb, step, e, user, ratio=0.,  k_max=1000):
    """
    Returns power distribution u_sol, the lagrangian multiplier, the number of Uzawa iterations and
     the value of the cost function in the static distributed optimization with a user cheating with his comfort factor.

    keyword arguments:
    pb -- dictionary of the problem (nbr of users, time step, max resources, max admissible power, thermal resistance,
     Thermal capacity, vector of the init temperature, value of the exterior temperature, reference temperature,
      comfort factor, size of the prediction horizon)
    step -- step of the Lagrangian in the Uzawa method
    e -- value of the maxim gap tolerable
    user -- number of the user cheating
    ratio -- ratio of the comfort factor in comparison to the one of the other users (default 0.0)
    k_max -- max number of iteration in the Uzawa method (default 1000)
    """

    Rth = pb['Rth']
    Text = pb['Text']
    T_id = pb['T_id']
    Umax = pb['Umax']
    u_m = pb['u_m']
    alpha = pb['alpha']

    u_id = (T_id - Text) / Rth

    L = 0
    m = len(Rth)
    u_sol = np.zeros(m)


    for k in range(k_max):
        assert L >= 0, "u_id can be reached for all users"
        for j in range(m):

            Pj = matrix(2 * alpha[j] * Rth[j] ** 2, tc='d')
            qj = matrix(1 - 2 * alpha[j] * Rth[j]**2 * u_id[j] + L, tc='d')
            Gj = matrix([-1, 1], tc='d')
            hj = matrix([0, u_m[j]], tc='d')

            solj = solvers.qp(Pj, qj, Gj, hj)
            u_sol[j] = solj['x'][0]

        Pj = matrix(2 * alpha[user] * Rth[user] ** 2, tc='d')
        qj = matrix(1 - 2 * alpha[user] * Rth[user] ** 2 * u_id[user] + (1-ratio)*L, tc='d')
        Gj = matrix([-1, 1], tc='d')
        hj = matrix([0, u_m[user]], tc='d')

        solj = solvers.qp(Pj, qj, Gj, hj)
        u_sol[user] = solj['x'][0]

        L = L + step * (u_sol.sum() - Umax)


        if u_sol.sum() - Umax < e:
            break
    return u_sol, L, k, 'tba'

def print_sol(pb, u_sol):
    """
        Returns a table of the average deviation of the temperature, the maximal power admissible, the ideal power
         distribution and the optimal power distribution

        keyword arguments:
        pb -- dictionary of the problem (nbr of users, time step, max resources, max admissible power, thermal resistance,
         Thermal capacity, vector of the init temperature, value of the exterior temperature, reference temperature,
          comfort factor, size of the prediction horizon)
        u_sol -- optimal power distribution

    """
    Rth = pb['Rth']
    Text = pb['Text']
    T_id = pb['T_id']
    u_m = pb['u_m']
    u_sol_v = u_sol[0]



    u_id = (T_id - Text) / Rth
    deltaT_opt = Rth * (u_sol_v.T - u_id)

    table = tabulate([deltaT_opt, u_m, u_id, u_sol_v], floatfmt=".6f")
    print(table)
    total = u_sol_v.sum()
    print(total)

def plot_sol(pb, u_sol, decen=True):
    """
        Plots the optimal power distribution

        keyword arguments:
        pb -- dictionary of the problem (nbr of users, time step, max resources, max admissible power, thermal resistance,
         Thermal capacity, vector of the init temperature, value of the exterior temperature, reference temperature,
          comfort factor, size of the prediction horizon)
        u_sol -- optimal power distribution

    """

    Rth = pb['Rth']
    Text = pb['Text']
    T_id = pb['T_id']
    Umax = pb['Umax']
    u_m = pb['u_m']
    i = np.arange(len(Rth))
    n = len(Rth)
    u_sol_v = u_sol[0]
    if decen==True:
        J_u = u_sol[3]
        kUzawa = u_sol[2]

    u_id = (T_id - Text) / Rth

    # graph colors (hexadecimal RRGGBB)
    c = {
        'max':'#bbd5f0', # white blue
        'id':'#4582c2', # blue
        'opt':'#f9d600' # golden yellow
    }

    fig, ax1 = plt.subplots(1,1)

    ax1.bar(i, u_m,  label='$u_{max}$',
            align='center',color=c['max'], linewidth=0)
    ax1.bar(i, u_id, label='$u_{id}$',
            align='center', color=c['id'], width=0.5, linewidth=0)
    ax1.plot(i, u_sol_v, 'D', label='$u^{*}$',
             color=c['opt'], markersize=20)
    ax1.hlines(Umax/n, -1/2, n -1/2, linestyles = '--', label='$U_{max}/n$')
    ax1.hlines(u_sol_v.mean(), -1/2, n-1/2, linestyles=':', label='$<u^{*}>$')

    deltaT_opt = Rth*(u_sol_v.T - u_id)

    for l in range(0, len(u_m)):
        ax1.annotate("%.1f" % float(deltaT_opt[l]), xy=(l, u_sol_v[l]), xycoords='data',
                     size='small', ha='center', va='center')

    if decen==True:
        ax1.annotate('J_u = %s' % J_u, xy=(0.055, -0.1), xycoords='data',
                     size='small', ha='center', va='center', annotation_clip=False)
        ax1.annotate(r'$k_{\;Uzawa}$ = %s' % kUzawa, xy=(-0.5, -0.12), xycoords='data',
                     size='small', ha='center', va='center', annotation_clip=False)

    ax1.set(
        title='Power allocation = {:.5f}/{}'.format(u_sol_v.sum(), Umax),
        xticks=i,
        xlabel='sub system',
        ylabel='heating power (kW)',
    )

    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #ax1.legend(loc='upper left', markerscale=0.4)

    fig.tight_layout()

    fig.savefig('power_bars.png', dpi=200, bbox_inches='tight')
    fig.savefig('power_bars.pdf', bbox_inches='tight')
    #plt.show()

    return fig, ax1

"""""" """""" """""" """"""

def plot_step1(pb, range, pas, e):

    """
        Returns the graph of the distributed optimal power allocation as a function of the Uzawa step.

        keyword arguments:
        pb -- dictionary of the problem (nbr of users, time step, max resources, max admissible power, thermal resistance,
         Thermal capacity, vector of the init temperature, value of the exterior temperature, reference temperature,
          comfort factor, size of the prediction horizon)
        range -- range of the Uzawa step
        pas -- step of the Uzawa step variation
        e -- admissible gap
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
        Returns the graph of the distributed optimal power allocation to Umax as a function of the
        Uzawa step, the value of the Lagrangian multiplier and the number of Uzawa iterations.

        keyword arguments:
        pb -- dictionary of the problem (nbr of users, time step, max resources, max admissible power, thermal resistance,
         Thermal capacity, vector of the init temperature, value of the exterior temperature, reference temperature,
          comfort factor, size of the prediction horizon)
        step_min -- minimal Uzawa step
        step_max -- maximal Uzawa step
        nbr -- number of point of the Uzawa step
        e -- admissible gap
    """

    fig, (ax1, ax2, ax3) = plt.subplots(3,1)
    fig.tight_layout()
    step = np.logspace(np.log10(step_min), np.log10(step_max), nbr)
    U = []
    k_val = []
    L_val = []

    for x in np.nditer(step):
        u_sol, L, k, J_u = optim_decen(pb, x, e)
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

"""""" """""" """""" """"""

def param_alpha(pb, a_beg, a_end, nbr):
    """
        Returns the optimal power distribution, the temperature deviation as a parametric study of the comfort factor
        and the comfort factor ratio vector.

        keyword arguments:
        pb -- dictionary of the problem (nbr of users, time step, max resources, max admissible power, thermal resistance,
         Thermal capacity, vector of the init temperature, value of the exterior temperature, reference temperature,
          comfort factor, size of the prediction horizon)
        a_min -- minimal comfort factor
        a_max -- maximal comfort factor
        nbr -- number of point of the comfort factor ratio
    """

    assert m==2, "illegal number of users. Expecting 2 and received %s." % m
    Rth=pb['Rth']
    Text=pb['Text']
    T_id=pb['T_id']
    Umax=pb['Umax']
    u_m=pb['u_m']
    alpha=pb['alpha']

    u_id = (T_id - Text) / Rth

    alpha_ratio = np.logspace(a_beg, a_end, nbr)

    _U = np.zeros((2, len(alpha_ratio)))
    _DT = np.zeros((2, len(alpha_ratio)))

    for z, ratio in enumerate(alpha_ratio):

        alpha[1] = ratio*alpha[0]
        print(alpha)

        _pb = dict(Rth=Rth, Text=Text, T_id=T_id, Umax=Umax, u_m=u_m, alpha=alpha)

        u_sol_d = optim_central(_pb)[0]
        _U[0, z] = u_sol_d[0]
        _U[1, z] = u_sol_d[1]

        _DT[0,z] = Rth[0] * (u_sol_d[0] - u_id[0])
        _DT[1, z] = Rth[1] * (u_sol_d[1] - u_id[1])

    return _U, _DT, alpha_ratio

def plot_alpha(_U, _DT, alpha_ratio):
    """
        Plots the optimal power distribution and the temperature as a parametric study of the comfort factor.

        keyword arguments:
        _U -- optimal power distribution as a function of the comfort factor ratio
        -DT -- optimal temperature deviation as a function of the comfort factor ratio
        alpha_ratio -- vector of the comfort factor ratio
    """


    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    ax1.plot(alpha_ratio, _U[0, :], 'r-+', label='1')
    ax1.plot(alpha_ratio, _U[1, :], 'g-+', label='2')

    ax1.hlines(Umax , alpha_ratio[0], alpha_ratio[-1], linestyles='--', label='')

    ax1.set(
        title=r'Parametric study of the comfort factor for $\alpha_1=$ %s' % alpha[0],
        xlabel=r'$ \alpha_{2} / \alpha_{1} $',
        ylabel='$u^{*}$',
        xscale='log'
    )

    ax1.legend(loc='upper left', markerscale=0.4)

    ax2.plot(alpha_ratio, _DT[0, :], 'r')
    ax2.plot(alpha_ratio, _DT[1, :], 'g')

    ax2.set(
        xlabel=r'$ \alpha_{2} / \alpha_{1} $',
        ylabel=r'$\Delta T = T -T_{id}$',
    )

    fig.tight_layout()

    plt.show()

    return fig, (ax1, ax2)

"""""" """""" """""" """"""

def param_Tbc(pb):
    """
        Returns the optimal power distribution, the temperature deviation and the supposed temperature as a parametric
         study of the broadcasted temperature.

        keyword arguments:
        pb -- dictionary of the problem (nbr of users, time step, max resources, max admissible power, thermal resistance,
         Thermal capacity, vector of the init temperature, value of the exterior temperature, reference temperature,
          comfort factor, size of the prediction horizon)

    """
    assert m == 3, "illegal number of users. Expecting 3 and received %s." % m
    Rth = pb['Rth']
    Text = pb['Text']
    T_id = pb['T_id']
    Umax = pb['Umax']
    u_m = pb['u_m']
    alpha = pb['alpha']

    u_id_real = (T_id - Text) / Rth

    T_sup = np.linspace(-1, 4, 20)

    _U = np.zeros((3, len(T_sup)))
    _DT = np.zeros((3, len(T_sup)))

    for z, sup in enumerate(T_sup):

        T_id[0] = T_id[0] + sup

        _pb = dict(Rth=Rth, Text=Text, T_id=T_id, Umax=Umax, u_m=u_m, alpha=alpha)

        u_sol_d = optim_central(_pb)[0]
        _U[0, z] = u_sol_d[0]
        _U[1, z] = u_sol_d[1]
        _U[2, z] = u_sol_d[2]
        _DT[0, z] = Rth[0] * (u_sol_d[0] - u_id_real[0])
        _DT[1, z] = Rth[1] * (u_sol_d[1] - u_id_real[1])
        _DT[2, z] = Rth[2] * (u_sol_d[2] - u_id_real[2])

    return _U, _DT, T_sup

def plot_Tbc(_U, _DT, T_sup):
    """
    Plots the optimal power distribution and the temperature deviation as a function of the broadcasted temperature.

    keyword arguments:
    _U -- optimal power distribution when the broadcasted temperature differ from the wanted temperature
    _DT -- optimal temperature deviation when the broadcasted temperature differ from the wanted temperature
    T_sup -- broadcasted temperature

    """
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    ax1.plot(T_sup, _U[0, :], 'r-+', label='1')
    ax1.plot(T_sup, _U[1, :], 'g-+', label='2')
    ax1.plot(T_sup, _U[2, :], 'g-+', label='3')

    ax1.hlines(Umax, T_sup[0], T_sup[-1], linestyles='--', label='')

    ax1.set(
        title=r'Parametric study of the broacasted temperature',
        xlabel=r'$ T_{sup} $',
        ylabel='$u^{*}$',
        #xscale='log',
    )

    ax1.legend(loc='upper left', markerscale=0.4)

    ax2.plot(T_sup, _DT[0, :], 'r')
    ax2.plot(T_sup, _DT[1, :], 'g')
    ax2.plot(T_sup, _DT[2, :], 'b')

    ax2.set(
        xlabel=r'$ T_{sup} $',
        ylabel=r'$\Delta T = T -T_{id}$',
    )

    fig.tight_layout()

    plt.show()

    return fig, (ax1, ax2)

"""""" """""" """""" """"""

def param_Rth(pb):
    """
        Returns the optimal power distribution, the temperature deviation and the supposed temperature as a parametric
         study of the thermal resistance.

        keyword arguments:
        pb -- dictionary of the problem (nbr of users, time step, max resources, max admissible power, thermal resistance,
         Thermal capacity, vector of the init temperature, value of the exterior temperature, reference temperature,
          comfort factor, size of the prediction horizon)

    """
    assert m == 3, "illegal number of users. Expecting 3 and received %s." % m
    Rth = pb['Rth']
    Rth_real = pb['Rth']
    _Rth = Rth
    Text = pb['Text']
    T_id = pb['T_id']
    Umax = pb['Umax']
    u_m = pb['u_m']
    alpha = pb['alpha']


    u_id = (T_id - Text)*1.0 / Rth

    varRth = np.linspace(1.0, 4.0, 100)

    _U = np.zeros((3, len(varRth)))
    _DT = np.zeros((3, len(varRth)))

    for z, sup in enumerate(varRth):

        _Rth[0] = sup

        _pb = dict(Rth=_Rth, Text=Text, T_id=T_id, Umax=Umax, u_m=u_m, alpha=alpha)

        u_sol_d = optim_central(_pb)[0]
        _U[0, z] = u_sol_d[0]
        _U[1, z] = u_sol_d[1]
        _U[2, z] = u_sol_d[2]
        _DT[0, z] = Rth_real[0] * (u_sol_d[0] - u_id[0])
        _DT[1, z] = Rth_real[1] * (u_sol_d[1] - u_id[1])
        _DT[2, z] = Rth_real[2] * (u_sol_d[2] - u_id[2])

    return _U, _DT, varRth, Rth_real

def plot_Rth(_U, _DT, varRth, Rth_real):
    """
    Plots the optimal power distribution and the temperature deviation  as a parametric study of the thermal
     resistance.

    keyword arguments:
    _U -- optimal power distribution when the thermal resistance is the parameter
    _DT -- optimal temperature deviation when the thermal resistance is the parameter
    varRth -- vector of the parametric thermal resistance
    Rth_real -- real thermal resistance

    """
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    ax1.plot(varRth, _U[0, :], 'r-+', label='1')
    ax1.plot(varRth, _U[1, :], 'g-+', label='2')
    ax1.plot(varRth, _U[2, :], 'g-+', label='3')

    ax1.hlines(Umax, varRth[0], varRth[-1], linestyles='--', label='')

    ax1.set(
        title=r'Parametric study of the broacasted Rth for $Rth_{real}$= %s' % Rth_real[0],
        xlabel=r'$ T_{sup} $',
        ylabel='$u^{*}$',
        #xscale='log',
    )

    ax1.legend(loc='upper left', markerscale=0.4)

    ax2.plot(varRth, _DT[0, :], 'r')
    ax2.plot(varRth, _DT[1, :], 'g')
    ax2.plot(varRth, _DT[2, :], 'b')

    ax2.set(
        xlabel=r'$Rth$',
        ylabel=r'$\Delta T = T -T_{id}$',
    )

    fig.tight_layout()

    plt.show()

    return fig, (ax1, ax2)

"""""" """""" """""" """"""

def param_mult(pb, l, step, e, user):
    """
        Plots the DeltaT as a function of the deafness of the user regarding the Lagrangian multiplier
        result of the talk between all users

        keyword arguments:
        pb -- dictionary of the problem (nbr of users, time step, max resources, max admissible power, thermal resistance,
         Thermal capacity, vector of the init temperature, value of the exterior temperature, reference temperature,
          comfort factor, size of the prediction horizon)
        l -- number of points for the parametric study
        step -- value of the Uzawa step
        e -- maximal admissible gap
        user -- number of the user studied
    """

    fig, (ax1) = plt.subplots(1, 1, sharex=True, figsize=(9, 6))

    Rth = pb['Rth']
    Text = pb['Text']
    T_id = pb['T_id']

    hor = np.linspace(0, 1, l)
    res = np.zeros_like(hor)
    res2 = np.zeros_like(hor)

    if user == 0:
        user2 = m
    else:
        user2 = user-1

    for z, ratio in enumerate(hor):
        u_sol = optim_CHT_decen(pb, step, e, user, ratio)
        u_sol_v = u_sol[0]

        u_id = (T_id - Text) / Rth

        deltaT_opt = Rth * (u_sol_v.T - u_id)

        res[z] = deltaT_opt[user]
        res2[z] = deltaT_opt[user2]

    ax1.plot(hor*100, res, 'r-+', label='user %s' % user)
    ax1.plot(hor * 100, res2, 'g-+', label='user %s' % user2)

    ax1.legend(loc='center right', markerscale=0.4)
    ax1.set(
        xlabel='percentage of deafness',
        ylabel=r'$\Delta T$'
    )
    fig.tight_layout()
    plt.show()

    return res


"""""" """""" """""" """"""
def pwrdist(pb,  step, e, k_max=1000):
    """
        Plots the ideal power distribution for the user from pb as function of the number of Uzawa iteration and the
        Lagrangian multiplier

        keyword arguments:
        pb -- dictionary of the problem (nbr of users, time step, max resources, max admissible power, thermal resistance,
         Thermal capacity, vector of the init temperature, value of the exterior temperature, reference temperature,
          comfort factor, size of the prediction horizon)
        step -- value of the Uzawa step
        e -- maximal admissible gap
        k_max = maximum Uzawa iteration (default 1000)

    """

    m = pb['m']
    Rth = pb['Rth']
    Text = pb['Text']
    T_id = pb['T_id']
    Umax = pb['Umax']
    u_m = pb['u_m']
    alpha = pb['alpha']

    k_Uzw = np.arange(k_max)
    u_opt = np.zeros(shape=(k_max, m))
    L_opt = np.zeros(shape=(k_max+1, m))

    u_id = (T_id - Text) / Rth

    L = 0
    m = len(Rth)
    u_sol = np.zeros(m)

    for k in range(k_max):
        assert L >= 0, "u_id can be reached for all users"
        for j in range(m):
            Pj = matrix(2 * alpha[j] * Rth[j] ** 2, tc='d')
            qj = matrix(1 - 2 * alpha[j] * Rth[j] ** 2 * u_id[j] + L, tc='d')
            Gj = matrix([-1, 1], tc='d')
            hj = matrix([0, u_m[j]], tc='d')

            solj = solvers.qp(Pj, qj, Gj, hj)
            u_sol[j] = solj['x'][0]
            u_opt[k, j] = u_sol[j]

        L = L + step * (u_sol.sum() - Umax)


        L_opt[k+1] = L

        if u_sol.sum() - Umax < e:
            print('break at %s.' % k)
            break

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6))

    L_opt = L_opt[0:k_max]
    for j in range(m):
        ax1.plot(k_Uzw, u_opt[:, j], '--', label='%s' % j)
        ax2.plot(L_opt, u_opt[:, j], '--')

    #ax1.plot(k_Uzw, u_opt[:, user], 'b', label='user %s' % user)
    #ax2.plot(L_opt, u_opt[:, user], 'b')

    #ax1.legend(loc='lower left', markerscale=0.4)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.set(
        xlabel=r'$k_{Uzawa}$',
        ylabel=r'$u^{*}$'
    )
    ax1.set_xlim([0, k+5])
    ax1.set_ylim([0, 1.05])

    ax1.annotate(r'%s' % k, xy=(k, -0.12), xycoords='data',
                 size='small', ha='center', va='center', annotation_clip=False)
    ax2.set(
        xlabel=r'$\lambda}$',
        ylabel=r'$u^{*}$'
    )
    ax2.set_xlim([0, L.round(0)+10])
    ax2.set_ylim([0, 1.05])

    ax2.annotate(r'%s' % L.round(1), xy=(L, -0.12), xycoords='data',
                 size='small', ha='center', va='center', annotation_clip=False)

    fig.tight_layout()

    return fig, (ax1, ax2)


###########################################################
if __name__ == '__main__':
    """
    Test bench
    """
    ## variable definition

    # number of users
    m = 3
    i = np.arange(m)

    # max energy in kW
    Umax = 2
    u_m = np.array([1, 1, 1], dtype=float)
    assert len(u_m) == m, "illegal number of users. Expecting %s. and received %s." % (m, len(u_m))

    # thermal resistance
    Rth =np.array([10, 10, 10])

    # Exterior temperature
    Text = 10

    # Ideal temperature in degrees
    T_id = np.array([21, 21, 21], dtype=float)
    assert len(T_id) == m, "illegal number of users. Expecting %s. and received %s." % (m, len(T_id))

    # Ideal energy
    deltaT = (T_id - Text)

    # comfort factor
    alpha = np.asarray([10, 10, 100], dtype=float)
    #assert len(alpha) == m, "illegal number of alpha. Expecting %s. and received %s." % (m, len(alpha))

    pb = dict(Rth=Rth, Text=Text, T_id=T_id, Umax=Umax, u_m=u_m, alpha=alpha, m=m)



    #u_sol_d = optim_decen(pb, 15, 1.0e-2)
    #print_sol(pb, u_sol_d)
    #plot_sol(pb, u_sol_d)

    #param_mult(pb, 10, 15, 1.0e-2, 1)
    pwrdist(pb, 15, 1.0e-2, k_max=1000)
    plt.show()

