from __future__ import division, print_function
from cvxopt import matrix, solvers
import numpy as np
from itertools import repeat
import matplotlib.pyplot as plt

from tabulate import tabulate
solvers.options['show_progress'] = False


"""
Distributed MPC Optimization Closed Loop version v2.0
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

def optim_decen(pb, step, e, k_max=100):
    """ Distributed optimization for power allocation

          optim_decen(object)
          Parameters : dictionary of matrix
          Returns : out ndarray of the optimal solution for each user, value of the Lagrangian multiplier and number of
          iterations

    """
    m = pb['m']
    Umax = pb['Umax']
    u_m = pb['u_m']
    Text = pb['Text']
    Text_sim = pb['Text_sim']
    T_mod = pb['T_mod']
    T_init = pb['T_init']
    Rth = pb['Rth']
    Cth = pb['Cth']
    T_id_pred = pb['T_id_pred']
    alpha = pb['alpha']
    N = pb['N']
    N_sim = pb['N_sim']

    mat = mat_def(pb)

    # Local variables
    tau_th = Rth * Cth

    U = np.ones(N_sim * m)
    T_res = np.zeros((N_sim+1) * m)
    Lgrn_mult = np.zeros(N_sim)
    cte_N = np.zeros(N_sim)
    J_u = np.zeros(N_sim)
    cost = np.zeros(N_sim)

    T_res[0:m] = T_init

    for k in range(N_sim):
        L = np.zeros(N)

        for i_u in range(k_max):

            Y_cj = np.zeros(N)
            for j in range(m):
                Xj = T_res[k*m + j]
                for l_hor in range(N):
                    Y_cj[l_hor] = T_mod[j * (N_sim + N) + k + l_hor]


                Aj = 1 - dt * (1 / tau_th[j])
                Fj = np.asarray([Aj ** (i + 1) for i in range(N)]).T
                Textj = np.asarray([np.zeros(N)]).T
                Bj = (dt / Cth[j])
                Cj = 1
                ###
                H_initj = [[Bj]]

                for l_h in range(N - 1):
                    H_initj = np.vstack((np.bmat([H_initj, np.zeros(shape=((l_h + 1), 1))]),
                                         [[Aj ** (l_h + 1 - x) * (Bj) for x in range(0, l_h + 2)]]))
                Hj = H_initj
                ###
                Gj = matrix(np.vstack((-np.identity(N), np.identity(N))), tc='d')
                G1 = np.zeros(shape=(N, m * N))
                for l in range(N):
                    for nb_user in range(m):
                        G1[l, l * m + nb_user] = 1
                ###
                Dj = np.identity(N) * alpha[j]
                ###
                hj = matrix(np.hstack((np.zeros(N), np.ones(N) * u_m[j])), tc='d')
                ###
                c_tj = np.ones(N)
                ###
                B_Textj = (dt / tau_th[j])
                ###
                H_init_Textj = [[B_Textj]]
                for l_hp in range(N - 1):
                    H_init_Textj = np.vstack((np.bmat([H_init_Textj, np.zeros(shape=((l_hp + 1), 1))]),
                                              [[Aj ** (l_hp + 1 - x) * (B_Textj) for x in range(0, l_hp + 2)]]))
                H_extj = H_init_Textj
                ###


                ###
                P_matj = 2 * (Hj.T).dot(Dj.dot(Hj))
                Pj = matrix(P_matj, tc='d')

                q_matjk = (c_tj + 2 * ((Fj * Xj).T.dot(Dj.dot(Hj))) + 2 * (
                    ((H_extj.dot(Textj)).T).dot(Dj.dot(Hj))) - 2 * (
                               Y_cj.T).dot(Dj.dot(Hj))) + L

                qj = matrix(q_matjk.T,
                            tc='d')

                ctej = ((Fj*Xj).T).dot(Dj.dot(Fj))*Xj + 2 * (
                ((H_extj*Textj).T).dot(Dj.dot(Fj)) - Y_cj.T.dot(Dj.T.dot(Fj)))*Xj + (H_extj*Textj).T.dot(
                    Dj.dot(H_extj*Textj)) - 2 * Y_cj.T.dot(Dj.T.dot(H_extj*Textj)) + Y_cj.T.dot(Dj*Y_cj)

                uk_sol = solvers.qp(Pj, qj, Gj, hj)

                U[k * m + j] = np.asarray(uk_sol['x']).T[0][0]



            Delta = G1.dot(U[0:N * m]) - Umax
            Delta[Delta < 0] = 0
            L = L + step * (Delta)


            for j in range(m):

                Aj = 1 - dt * (1 / tau_th[j])
                Bj = (dt / Cth[j])
                B_Textj = (dt / tau_th[j])
                T_res[(k+1) * m + j] = Aj * T_res[k * m + j] + Bj * U[k * m + j] + B_Textj * Text_sim[k * m + j]

            if all(x < e for x in Delta):
                print('break at %s.' % (i_u))
                break

        Lgrn_mult[k] = L[0]
        cost[k] = sum(U[k*m:k*m + m])

        Gu = np.zeros(shape=(N_sim, m * N_sim))
        for l in range(N_sim):
            for nb_user in range(m):
                Gu[l, l * m + nb_user] = 1

        comfort = np.zeros(N_sim)
        for hor in range(N_sim):
            comfort[hor] = sum((T_res[k_room +  hor* m] - T_id_pred[k_room + hor * m]) ** 2 for k_room in range(m))

        J_u = Gu.dot(U) + comfort

    return U, T_res, Lgrn_mult, cost, J_u

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
    F = mat['F']
    H = mat['H']
    H_ext = mat['H_ext']

    X = T_init

    Y = F.dot(X) + H.dot(u_sol) + H_ext.dot(Text)

    return Y

def get_Opt_CL(pb):
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

def plot_t(pb, i, T_opt, u_sol, lab1, T_opt2, u_sol2, lab2):
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

    t = np.arange(N_sim) * dt

    fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, figsize=(6,4))

    ax1.plot(t, [T_id_pred[i*N_sim + j] for j in range(N_sim)], 'k:',
             label='T_id')
    ax1.plot(t, [T_opt[m*j+i] for j in range(N_sim)], label=lab1)
    ax1.plot(t, [T_opt2[m * j + i] for j in range(N_sim)], '--', label=lab2)
    ax1.legend()

    ax2.plot(t, [u_sol[m*j+i] for j in range(N_sim)], label=lab1)
    ax2.plot(t, [u_sol2[m * j + i] for j in range(N_sim)], '--', label=lab2)
    ax2.legend()
    
    ax1.set(
        ylabel=u'Temperature (C)'
    )

    ax2.set(
        xlabel='t (h)',
        ylabel='Power (kW)'
    )

    fig.tight_layout()
    plt.show()

    return fig, (ax1, ax2)

def temp_id(size, T_abs, T_pres):

    t = np.arange(size) * dt
    occ = occupancy(t)
    T_min = np.zeros(size) + T_abs  # degC
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
    N = int(5/dt)

    # max energy in kW
    Umax = 3

    # max admissible energy
    u_m = np.array([2, 2], dtype=float)
    assert len(u_m) == m, "illegal number of users. Expecting %s. and received %s." % (m, len(u_m))

    # thermal parameters
    Text_sim = np.ones(m*(N_sim + N))*0
    Text = Text_sim[0:m*N]
    Tpres = 22
    Tabs = 18
    T_init = np.array([10, 10], dtype=float)
    Rth = np.array([50, 50], dtype=float)
    Cth = np.array([0.056, 0.056], dtype=float)
    assert len(T_init) == m, "illegal number of T_init. Expecting %s. and received %s." % (m, len(T_init))
    assert len(Rth) == m, "illegal number of Rth. Expecting %s. and received %s." % (m, len(Rth))
    assert len(Cth) == m, "illegal number of Cth. Expecting %s. and received %s." % (m, len(Cth))


    T_mod = np.hstack((temp_id(N_sim+N, Tabs, Tpres), temp_id(N_sim+N, Tabs, Tpres))) ## ATTENTION : defined user after user

    T_id_pred = np.hstack((temp_id(N_sim, Tabs, Tpres), temp_id(N_sim, Tabs, Tpres)))  ## ATTENTION : defined user after user


    # comfort factor
    alpha = np.array([10, 10], dtype=float)

    pb = dict(m=m, dt=dt, Umax=Umax, u_m=u_m, Text=Text, Text_sim=Text_sim, T_mod=T_mod, T_init=T_init, Rth=Rth, Cth=Cth,
              T_id_pred=T_id_pred, alpha=alpha, N=N, N_sim=N_sim)


    T_cen, U_cen =get_Opt_CL(pb)
    pb['T_init'] = T_init
    U, T_res, L, cost, J_u= optim_decen(pb, 0.15, 1.0e-1)
    plot_t(pb, 0, T_res, U, 'dist.', T_cen, U_cen, 'cent.')
    print(J_u)

    #mat = mat_def(pb)
    #print(mat)
    #u_sol = optim_central(mat)[0]
    #T_opt = get_temp_op_OL(pb, mat, u_sol)
    #plot_t(pb, 0, T_opt, u_sol)  ## be careful and set N=N_sim otherwise error









