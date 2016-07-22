# Sylvain Chatel - July 2016

from __future__ import division, print_function
from cvxopt import matrix, solvers
import numpy as np
from itertools import repeat
import matplotlib.pyplot as plt

solvers.options['show_progress'] = False


"""
Centralized MPC Optimization  Closed Loop version v2.3
"""

def mat_def(pb):
    """
        Forms all the matrix needed for the QP.

        keyword arguments:
        pb -- dictionary of the problem (nbr of users, time step, max resources, max admissible power, thermal resistance,
         Thermal capacity, vector of the init temperature, value of the exterior temperature, reference temperature,
          comfort factor, size of the prediction horizon)
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
            G1[l, l * m + k] = 1

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
            Y_c[l + m * k] = T_id_pred[l*N + k]
    ###
    cte = ((F.dot(X)).T).dot(D.dot(F)).dot(X) + 2*(((H_ext.dot(Text)).T).dot(D.dot(F)) - Y_c.T.dot(D.T.dot(F))).dot(X) + (H_ext.dot(Text)).T.dot(D.dot(H_ext.dot(Text))) - 2*Y_c.T.dot(D.T.dot(H_ext.dot(Text))) + Y_c.T.dot(D.dot(Y_c))
    ###
    P_mat = 2*(H.T).dot(D.dot(H))
    P = matrix(P_mat, tc='d')
    ###
    q_mat = (c_t + 2 * ((F.dot(X)).T.dot(D.dot(H))) + 2 * (((H_ext.dot(Text)).T).dot(D.dot(H))) - 2 * (Y_c.T).dot(D.dot(H)))
    q = matrix(q_mat.T,
               tc='d')
    ###
    ###
    mat = dict(A=A, B=B, C=C, F=F, H=H, G=G, B_Text = B_Text, H_ext=H_ext, c_t=c_t, h=h, D=D, P=P, q=q, Y_c=Y_c, cte=cte, P_mat=P_mat, q_mat=q_mat)

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
    Returns the vector of optimal power for the pb with central open loop control, and the value of the primal objective

    keyword arguments:
    mat -- result of mat_def(pb)
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
    Returns the optimal temperature in the centralized open loop control.

    keyword arguments:
    pb -- dictionary of the problem (nbr of users, time step, max resources, max admissible power, thermal resistance,
     Thermal capacity, vector of the init temperature, value of the exterior temperature, reference temperature,
      comfort factor, size of the prediction horizon)
    mat -- dictionary of the matrix result of mat_def (F, H, H_ext)
    u_sol -- optimal power distribution, result of optim_central
    """
    # parameters
    T_init = pb['T_init']
    Text = pb['Text']
    F = mat['F']
    H = mat['H']
    H_ext = mat['H_ext']

    X = T_init

    Y = F.dot(X) + H.dot(u_sol) + H_ext.dot(Text)

    return Y

def get_Opt_CL(pb):
    """
        Returns the optimal temperature in central closed loop control, the optimal power distribution,
         the power cost and the value of the cost function.

        keyword arguments:
        pb -- dictionary of the problem (nbr of users, time step, max resources, max admissible power, thermal resistance,
         Thermal capacity, vector of the init temperature, value of the exterior temperature, reference temperature,
          comfort factor, size of the prediction horizon, size of the simulation horizon)
    """
    pb_k = pb
    mat_k = mat_def(pb_k)

    A = mat_k['A']
    B = mat_k['B']
    D = mat_k['D']
    c_t = mat_k['c_t']
    F = mat_k['F']
    H = mat_k['H']
    H_ext = mat_k['H_ext']
    B_Text = mat_k['B_Text']
    T_init = pb['T_init']
    P = mat_k['P']
    q = mat_k['q']
    Y_c = mat_k['Y_c']
    T_mod = pb['T_mod']

    U = np.zeros(N_sim * m)
    T_res = np.zeros(N_sim * m)

    for k in range(N_sim):

        pb_k['T_init'] = T_init

        for j in range(N):
            for i in range(m):
                Y_c[j*m + i] = T_mod[i*(N_sim + N) + k + j]


        q_mat = (c_t + 2 * ((F.dot(T_init)).T.dot(D.dot(H))) + 2 * (((H_ext.dot(Text)).T).dot(D.dot(H))) - 2 * (Y_c.T).dot(D.dot(H)))
        q = matrix(q_mat.T, tc='d')

        mat_k['q_mat'] = q_mat
        mat_k['q'] = q

        uk_sol = optim_central(mat_k)


        U[k*m:k*m+m] = uk_sol[0][0:m]

        T_init = A.dot(T_init) + B.dot(uk_sol[0][0:m]) + B_Text.dot(Text_sim[k*m:k*m + m])
        T_res[k*m:(k+1)*m] = T_init

    return T_res, U

def plot_t(pb, i, T_opt, u_sol):
    """
    Returns the graph of temperature and power consumption for user i.

    keyword arguments:
    pb -- dictionary of the problem (nbr of users, time step, max resources, max admissible power, thermal resistance,
     Thermal capacity, vector of the init temperature, value of the exterior temperature, reference temperature,
      comfort factor, size of the prediction horizon)
    i -- number of the user to plot
    T_opt -- optimal temperature profile
    u_sol -- optimal power distribution


    """

    T_id_pred = pb['T_id_pred']

    Text = pb['Text']
    dt = pb['dt']
    N = pb['N']
    m = pb['m']

    t = np.arange(N_sim) * dt

    fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, figsize=(6,4))

    ax1.plot(t, [T_id_pred[i*N_sim + j] for j in range(N_sim)])
    ax1.plot(t, [T_opt[m*j+i] for j in range(N_sim)])

    ax2.plot(t, [u_sol[m*j+i] for j in range(N_sim)], 'r')

    ax1.set(
        ylabel='Temperature (deg C)'
    )

    ax2.set(
        xlabel='t(h)',
        ylabel='Power(kW)'
    )

    fig.tight_layout()
    plt.show()

    return fig, (ax1, ax2)

def temp_id(size, T_abs, T_pres):
    """
        Parameters : size of the horizon, temperature when absent, temperature when present
        Returns : temperature profile
    """

    t = np.arange(size) * dt
    occ = occupancy(t)
    T_min = np.zeros(size) + T_abs  # degC
    T_min[occ] = T_pres

    return T_min

def get_Cost(mat, u_sol):
    """
        Returns the value of the cost function after the optimization.

        keyword arguments:
        mat -- result of mat_def(pb)
        u_sol -- optimal power distribution

    """

    P_mat = mat['P_mat']
    q_mat = mat['q_mat']
    cte = mat['cte']

    Ju_opt = u_sol.T.dot(P_mat.dot(u_sol)) + q_mat.dot(u_sol) + cte

    return Ju_opt

if __name__ == '__main__':
    """
    Test bench
    """

    # number of users
    m = 2
    i = np.arange(m)

    # Time step
    dt = 0.1  # h

    # Horizon
    N_sim = int(24/dt)
    N = int(5/dt)

    # max energy in kW
    Umax = 10

    # max admissible energy
    u_m = np.array([1, 1], dtype=float)
    assert len(u_m) == m, "illegal number of users. Expecting %s. and received %s." % (m, len(u_m))

    # thermal parameters
    Text_sim = np.ones(m*(N_sim + N))*0
    Text = Text_sim[0:m*N]
    Tpres = 22
    Tabs = 18
    T_init = np.array([0, 0], dtype=float)
    Rth = np.array([50, 50], dtype=float)
    Cth = np.array([0.056, 0.056], dtype=float)
    assert len(T_init) == m, "illegal number of T_init. Expecting %s. and received %s." % (m, len(T_init))
    assert len(Rth) == m, "illegal number of Rth. Expecting %s. and received %s." % (m, len(Rth))
    assert len(Cth) == m, "illegal number of Cth. Expecting %s. and received %s." % (m, len(Cth))


    T_mod = np.hstack((temp_id(N_sim+N, 17, 25), temp_id(N_sim+N, Tabs, Tpres))) ## ATTENTION : defined user after user

    T_id_pred = np.hstack((temp_id(N_sim, 17, 25), temp_id(N_sim, Tabs, Tpres)))  ## ATTENTION : defined user after user


    # comfort factor
    alpha = np.array([100, 100], dtype=float)

    pb = dict(m=m, dt=dt, Umax=Umax, u_m=u_m, Text=Text, Text_sim=Text_sim, T_mod=T_mod, T_init=T_init, Rth=Rth, Cth=Cth,
              T_id_pred=T_id_pred, alpha=alpha, N=N, N_sim=N_sim)


    #mat = mat_def(pb)
    #u_sol = optim_central(mat)[0]
    #T_opt = get_temp_op_OL(pb, mat, u_sol)
    #plot_t(pb, 0, T_opt, u_sol)  ## be careful and set N=N_sim otherwise error
    #T_res, U = optim_central_CL(pb)
    #plot_t(pb, 0, T_res, U)









