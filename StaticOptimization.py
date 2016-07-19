from __future__ import division, print_function
from cvxopt import matrix, solvers
import numpy as np
import matplotlib.pyplot as plt

from tabulate import tabulate
solvers.options['show_progress'] = False

"""
Centralized Optimization  version v1.5
TODO = MPC in version v2.0
"""

def optim_central(pb):
    """ Centralized optimization for power allocation

    StaticOptimization.optim_central(object)
    Parameters : dictionary of parameters Rth, Text, T_id, Umax, u_m, alpha.
    Returns : out ndarray of the optimal solution for each user

    """
    Rth = pb['Rth']
    Text = pb['Text']
    T_id = pb['T_id']
    Umax = pb['Umax']
    u_m = pb['u_m']
    alpha = pb['alpha']

    u_id = (T_id - Text) /Rth

    # Matrix definition
    P = matrix(2 * np.diag(alpha*Rth ** 2), tc='d')
    q = matrix(1 - 2 * alpha*u_id * (Rth ** 2), tc='d')
    G = matrix(np.vstack((np.ones(len(Rth)), -np.identity(len(Rth)), np.identity(len(Rth)))), tc='d')
    h = matrix(np.hstack((Umax, np.zeros(len(Rth)), u_m)), tc='d')

    # Resolution
    sol = solvers.qp(P, q, G, h)

    # Solution
    u_sol = np.asarray(sol['x']).T[0]

    return u_sol,

def optim_decen(pb, step, e, k_max=1000):
    """ Distributed optimization for power allocation

       StaticOptimization.optim_decen(object)
       Parameters : dictionary of parameters (Rth, Text, T_id, Umax, u_m, alpha), step for the Uzawa method, error, max
       iterration.
       Returns : out ndarray of the optimal solution for each user, value of the Lagrangian multiplier and number of
       iterations

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


        L = L + step * (u_sol.sum() - Umax)

        if u_sol.sum() - Umax < e:
            #print('break at %s.' %(k))
            break
    return u_sol, L, k

def optim_CHT_decen(pb, step, e, user, ratio=0.,  k_max=1000):
    """ Distributed optimization for power allocation

       StaticOptimization.optim_decen(object)
       Parameters : dictionary of parameters (Rth, Text, T_id, Umax, u_m, alpha), step for the Uzawa method, error, max
       iterration.
       Returns : out ndarray of the optimal solution for each user, value of the Lagrangian multiplier and number of
       iterations

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
        qj = matrix(1 - 2 * alpha[user] * Rth[user] ** 2 * u_id[user] + ratio*L, tc='d')
        Gj = matrix([-1, 1], tc='d')
        hj = matrix([0, u_m[user]], tc='d')

        solj = solvers.qp(Pj, qj, Gj, hj)
        u_sol[user] = solj['x'][0]

        L = L + step * (u_sol.sum() - Umax)


        if u_sol.sum() - Umax < e:
            break
    return u_sol, L, k

def print_sol(pb, u_sol):
    """StaticOptimization.print_sol(param, sol )
       Parameters : dictionary of parameters (Rth, Text, T_id, Umax, u_m, alpha), solution of the optimization QP as an
       array.
       Returns : out table of deltaT, u_m, u_id and u_sol for each user
    """
    Rth = pb['Rth']
    Text = pb['Text']
    T_id = pb['T_id']
    u_m = pb['u_m']
    u_sol_v = u_sol[0]


    u_id = (T_id - Text) / Rth
    deltaT = T_id - Text

    table = tabulate([deltaT, u_m, u_id, u_sol_v], floatfmt=".6f")
    print(table)
    total = u_sol_v.sum()
    print(total)

def plot_sol(pb, u_sol):
    """StaticOptimization.plot_sol(param, sol )
       Parameters : dictionary of parameters (Rth, Text, T_id, Umax, u_m, alpha), solution of the optimization QP as an
       array.
       Returns : out bar graph of the power allocation, max power and ideal power  for each user.
    """

    Rth = pb['Rth']
    Text = pb['Text']
    T_id = pb['T_id']
    Umax = pb['Umax']
    u_m = pb['u_m']
    i = np.arange(len(Rth))
    n = len(Rth)
    u_sol_v = u_sol[0]


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

    ax1.set(
        title='Power allocation = {:.5f}/{}'.format(u_sol_v.sum(), Umax),
        xticks=i,
        xlabel='sub system',
        ylabel='heating power (kW)',
    )



    ax1.legend(loc='upper left', markerscale=0.4)

    fig.tight_layout()

    fig.savefig('power_bars.png', dpi=200, bbox_inches='tight')
    fig.savefig('power_bars.pdf', bbox_inches='tight')
    #plt.show()

    return fig, ax1

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

def param_alpha(pb, a_beg, a_end, nbr):
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

def param_Tbc(pb):
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

def param_Rth(pb):
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

def param_mult(pb, l, step, e, user):

    fig, (ax1) = plt.subplots(1, 1, sharex=True, figsize=(9, 6))

    Rth = pb['Rth']
    Text = pb['Text']
    T_id = pb['T_id']

    hor = np.linspace(0, 1, l)
    res = np.zeros_like(hor)

    for z, ratio in enumerate(hor):
        u_sol = optim_CHT_decen(pb, step, e, user, ratio)
        u_sol_v = u_sol[0]

        u_id = (T_id - Text) / Rth

        deltaT_opt = Rth * (u_sol_v.T - u_id)

        res[z] = deltaT_opt[user]

    ax1.plot(hor*100, res, '-+')
    ax1.set(
        xlabel='percentage of multiplier taken into account',
        ylabel=r'$\Delta T$'
    )
    fig.tight_layout()
    plt.show()

    return res

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
    u_m = np.array([1.5, 1.5, 1.5], dtype=float)
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
    alpha = np.asarray([10, 10, 10], dtype=float)
    #assert len(alpha) == m, "illegal number of alpha. Expecting %s. and received %s." % (m, len(alpha))

    pb = dict(Rth=Rth, Text=Text, T_id=T_id, Umax=Umax, u_m=u_m, alpha=alpha)



    #u_sol_d = optim_CHT_decen(pb, 1.5, 1.0e-2, 1, 0.25)
    #print_sol(pb, u_sol_d)
    #plot_sol(pb, u_sol_d)

    param_mult(pb, 10, 1.5, 1.0e-2, 1)



