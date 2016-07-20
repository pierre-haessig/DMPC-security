from __future__ import division, print_function
from cvxopt import matrix, solvers
import numpy as np
import StaticPlot as SP
import StaticOptimization as SO
import matplotlib.pyplot as plt

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
