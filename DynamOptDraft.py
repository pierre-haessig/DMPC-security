from __future__ import division, print_function
from cvxopt import matrix, solvers
import numpy as np
from itertools import repeat
import matplotlib.pyplot as plt

from tabulate import tabulate
solvers.options['show_progress'] = False


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
    Rth = pb['Rth']
    Cth = pb['Cth']
    T_init = pb['T_init']
    Text = pb['Text']
    T_id_pred = pb['T_id_pred']
    alpha = pb['alpha']
    N = pb['N']


    # Local variables
    tau_th = Rth * Cth


    # Matrix definition
    X = T_init
    ###
    D = np.diag([x for item in alpha for x in repeat(item, N)])
    ###
    h = matrix(np.hstack((Umax * np.ones(N), np.zeros(m * N), np.tile(u_m, N))), tc='d')
    ###
    c_t = np.ones(m * N)
    ###
    A = np.identity(m) + dt * np.diag(-1 / tau_th)
    ###
    F = np.hstack([np.linalg.matrix_power(A, i + 1) for i in range(N)]).T
    ###
    C = 1
    ###
    G1 = np.zeros(shape=(N, m * N))
    for l in range(N):
        for k in range(m):
            G1[l, l * (m) + k] = 1

    G = matrix(np.vstack((G1, -np.identity(m * N), np.identity(m * N))), tc='d')
    ###
    B = np.diag(dt / Cth)
    ###
    H_init = np.bmat(B)
    for l in range(N - 1):
        H_init = np.asarray(np.bmat([[H_init, np.zeros(shape=(m * (l + 1), m))], [
            np.hstack([np.linalg.matrix_power(A, l + 1 - x).dot(B) for x in range(0, l + 2)])]]))
    H = H_init
    ###
    B_Text = np.diag(dt/tau_th)
    ###
    H_init_Text = np.bmat(B_Text)
    for l in range(N - 1):
        H_init_Text = np.asarray(np.bmat([[H_init_Text, np.zeros(shape=(m * (l + 1), m))], [
            np.hstack([np.linalg.matrix_power(A, l + 1 - x).dot(B_Text) for x in range(0, l + 2)])]]))
    H_ext = H_init_Text
    ###
    Y_c = np.zeros(m * N)

    for k in range(N):
        for l in range(m):
            Y_c[l + m * k] = T_id_pred[l, k]
    ###
    cte = ((F.dot(X)).T).dot(D.dot(F)).dot(X) + 2*(((H_ext.dot(Text)).T).dot(D.dot(F)) - Y_c.T.dot(D.T.dot(F))).dot(X) + (H_ext.dot(Text)).T.dot(D.dot(H_ext.dot(Text))) - 2*Y_c.T.dot(D.T.dot(H_ext.dot(Text))) + Y_c.T.dot(D.dot(Y_c))
    ###
    P_mat = (H.T).dot(D.dot(H))
    P = matrix(P_mat, tc='d')
    ###
    q_mat = (c_t + 2 * ((F.dot(X)).T.dot(D.dot(H))) + 2 * (((H_ext.dot(Text)).T).dot(D.dot(H))) - 2 * (Y_c.T).dot(D.dot(H)))
    q = matrix(q_mat.T,
               tc='d')
    ###
    ###
    mat = dict(A=A, B=B, C=C, F=F, H=H, G=G, H_ext=H_ext, c_t=c_t, h=h, D=D, P=P, q=q, Y_c=Y_c, cte=cte, P_mat=P_mat, q_mat=q_mat)

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
    T_init = pb['T_init']
    Text = pb['Text']
    m = pb['m']
    N = pb['N']
    F = mat['F']
    H = mat['H']
    H_ext = mat['H_ext']

    X = T_init

    Y = F.dot(X) + H.dot(u_sol) + H_ext.dot(Text)

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
        #ylim=(0, max([u_sol[m*j+i] for j in range(N)])+0.5)
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
    m = 1
    i = np.arange(m)

    # Time step
    dt = 0.1  # h

    # Horizon
    N_sim = int(24/dt)
    N = N_sim


    # max energy in kW
    Umax = 10

    # max admissible energy
    u_m = np.array([2], dtype=float)
    assert len(u_m) == m, "illegal number of users. Expecting %s. and received %s." % (m, len(u_m))

    # thermal parameters
    Text = np.ones(m*N)*0
    T_init = np.array([0], dtype=float)
    Rth = np.array([25], dtype=float)
    Cth = np.array([0.28], dtype=float)
    assert len(T_init) == m, "illegal number of T_init. Expecting %s. and received %s." % (m, len(T_init))
    assert len(Rth) == m, "illegal number of Rth. Expecting %s. and received %s." % (m, len(Rth))
    assert len(Cth) == m, "illegal number of Cth. Expecting %s. and received %s." % (m, len(Cth))


    T_id_pred = np.array([temp_id(18, 22)])


    # comfort factor
    alpha = np.array([10000], dtype=float)


    pb = dict(m=m, dt=dt, Umax=Umax, u_m=u_m, Text=Text, T_init=T_init, Rth=Rth, Cth=Cth, T_id_pred=T_id_pred, alpha=alpha, N=N)


    mat = mat_def(pb)
    u_sol = optim_central(mat)[0]
    #u_sol = np.ones(m*N)*3
    #u_sol = np.zeros(m * N)
    P_mat = mat['P_mat']
    q_mat = mat['q_mat']
    cte = mat['cte']

    #print(mat)

    #Ju_opt = u_sol.T.dot(P_mat.dot(u_sol)) + q_mat.dot(u_sol) + cte
    T_opt = get_temp_op_OL(pb, mat, u_sol)
    plot_t(pb, 0, T_opt, u_sol)

    print('o')





