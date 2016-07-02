from __future__ import division, print_function
from cvxopt import matrix, solvers
import numpy as np
from itertools import repeat
import matplotlib.pyplot as plt

from tabulate import tabulate
solvers.options['show_progress'] = False
np.set_printoptions(threshold=np.nan)
"""
Centralized MPC Optimization  version v2.0
"""

def mat_def(pb):
    """
    DynamicOptimisation.mat_def(object)
    parameters : dictionary of all the variables
    returns : dictionary of all the matrix needed in the optimization problem
    """
    # parameters
    m = pb['m']
    dt = pb['dt']
    Umax = pb['Umax']
    u_m = pb['u_m']
    Text = pb['Text']
    Rth = pb['Rth']
    Cth = pb['Cth']
    T_id_pred = pb['T_id_pred']
    alpha = pb['alpha']
    N = pb['N']


    # Local variables
    tau_th = Rth * Cth


    # Matrix definition
    D = np.diag([x for item in alpha for x in repeat(item, N)])

    h = matrix(np.hstack((Umax * np.ones(N), np.zeros(m * N), np.tile(u_m, N))), tc='d')

    c_t = np.ones(m * N)

    A = np.identity(m) + dt * np.diag(1 / tau_th)

    F = np.hstack([np.linalg.matrix_power(A, i + 1) for i in range(N)]).T

    C = 1

    G1 = np.zeros(shape=(N, m * N))
    for l in range(N):
        for k in range(m):
            G1[l, l * (m) + k] = 1

    G = matrix(np.vstack((G1, -np.identity(m * N), np.identity(m * N))), tc='d')

    B = np.diag(dt / Cth)

    H_init = np.bmat(B)
    for l in range(N - 1):
        H_init = np.asarray(np.bmat([[H_init, np.zeros(shape=(m * (l + 1), m))], [
            np.hstack([np.linalg.matrix_power(A, l + 1 - x).dot(B) for x in range(0, l + 2)])]]))

    H = H_init

    W = np.zeros(m * N)

    for k in range(N):
        for l in range(m):
            W[l + m * k] = T_id_pred[l, k]

    P = matrix((H.transpose().dot(D)).dot(H), tc='d')

    q = matrix((c_t + 2 * ((X.transpose()).dot(F.T)).dot(D.dot(H)) - 2 * (W.transpose()).dot(D.dot(H))).transpose(),
               tc='d')



    mat = dict(A=A, B=B, C=C, F=F, H=H, G=G, c_t=c_t, h=h, D=D, P=P, q=q, W=W)

    return mat

def occupancy(t, t_switch=((6.5, 8), (18, 22))):
    '''boolean occupancy vector for each instant in vector `t` (in hours)

    occupancy is True between switching hours `t_switch`,
    which is a list of pairs of hours (t_on, t_off).

    By default, occupancy is True:

    * in the morning 6:30 to 8:00
    * in the evening: 18:00 to 22:00
    '''
    h = t % 24
    occ = np.zeros_like(t, dtype=bool)
    for (t_on, t_off) in t_switch:
        assert t_off >= t_on
        occ |= (h>=t_on) & (h<=t_off)
    return occ

def optim_central(mat):
    """
    DynamicOptimization.optim_central(object)
    Parameters : dictionary of all the matrix needed cf. DynamicOpt.mat_def
    Returns : out ndarray of the optimal solution for each user

    """
    # parameters
    G = mat['G']
    h = mat['h']
    P = mat['P']
    q = mat['q']

    # Resolution
    sol = solvers.qp(P, q, G, h)
    u_sol = np.asarray(sol['x']).T[0]
    prime_obj = sol['primal objective']

    return u_sol, prime_obj

def get_temp_op_OL(pb, mat,  u_sol):
    """
    DynamicOpt.get_temp_opt_OL(object)
    Parameters : dictionary of all the variables, dictionary of all the matrix, the vector solution of the power
    optimization problem.
    returns : vector Y of all the temperature in open loop control

    """
    # parameters
    Text = pb['Text']
    F = mat['F']
    H = mat['H']

    X = np.ones(m)*Text

    Y = F.dot(X) + H.dot(u_sol)

    return Y

def plot_t(pb, i, T_opt, u_sol):
    """
    DynamicOpt.plot_traj(object)
    Parameters : dictionary of the variables, number of the user, the vector of all optimal temperature
    from DynamicOpt.get_temp_op_OL and the vector of optimal power.
    returns : graph of the ideal temperature and the optimum temperature.
    """

    T_id_pred = pb['T_id_pred']
    Text = pb['Text']
    dt = pb['dt']
    N = pb['N']
    m = pb['m']

    t = np.arange(N) * dt

    fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, figsize=(6,4))

    ax1.plot(t, [T_id_pred[i, j] for j in range(N)])
    ax1.plot(t, [T_opt[m*j+i] for j in range(N)])

    ax2.plot(t, [u_sol[m*j+i] for j in range(N)], 'r')

    ax1.set(
        ylabel='Temperature (deg C)',
        #ylim=(Text-0.5, T_id_pred.max()+0.5)
    )

    ax2.set(
        xlabel='t(h)',
        ylabel='Power(kW)',
        ylim=(0, max([u_sol[m*j+i] for j in range(N)])+0.5)
    )

    fig.tight_layout()
    plt.show()

    return fig, (ax1, ax2)

def temp_id(T_abs, T_pres):

    t = np.arange(N_sim) * dt
    occ = occupancy(t)
    T_min = np.zeros(N_sim) + T_abs  # degC
    T_min[occ] = T_pres

    return T_min

if __name__ == '__main__':

    # number of users
    m = 2
    i = np.arange(m)

    # Time step
    dt = 0.1  # h

    # Horizon
    N_sim = int(24/dt)
    N = 5


    # max energy in kW
    Umax = 30

    # max admissible energy
    u_m = np.array([20, 10], dtype=float)
    assert len(u_m) == m, "illegal number of users. Expecting %s. and received %s." % (m, len(u_m))

    # thermal parameters
    Text = 18
    Rth = np.array([20, 20], dtype=float)
    Cth = np.array([2, 1], dtype=float)
    assert len(Rth) == m, "illegal number of Rth. Expecting %s. and received %s." % (m, len(Rth))
    assert len(Cth) == m, "illegal number of Cth. Expecting %s. and received %s." % (m, len(Cth))


    T_id_pred = np.array([[18, 20, 20, 20, 18], [18, 30, 30, 18, 18]])


    # comfort factor
    alpha = np.array([0, 0], dtype=float)



    # Initial state
    X = np.ones(m)*Text

    pb = dict(m=m, dt=dt, Umax=Umax, u_m=u_m, Text=Text, Rth=Rth, Cth=Cth, T_id_pred=T_id_pred, alpha=alpha, N=N)


    mat = mat_def(pb)
    u_sol = optim_central(mat)[0]
    Ju_opt = optim_central(mat)[1]
    T_opt = get_temp_op_OL(pb, mat, u_sol)

    print(mat['H'])





