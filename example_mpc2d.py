#!/usr/bin/python
# -*- coding: utf-8 -*-
# Pierre Haessig — July 2016
""" An usage example of the `mpc` module with a 2d thermal system
(2 independant rooms for example)
"""

from __future__ import division, print_function, unicode_literals
import numpy as np

import mpc

n_sys = 2
dyn2 = mpc.dyn_from_thermal([20,20], [0.05, 0.05], 0.1)


nh = 20
c2 = mpc.MPC(dyn2, nh, u_min=0, u_max=1.5, u_cost=1, track_weight=100)

zh = np.zeros((nh*n_sys,1))
p_hor = 2+zh # °C

# Output prediction

u_hor = np.zeros((nh, n_sys))
u_hor[10:] = 1

u_hor_flat = u_hor.reshape((-1, 1))


T0 = np.array([20,10]).reshape((n_sys,1))

#print(c2.pred_output(T0, u_hor_flat, p_hor))

T = c2.pred_output(T0, u_hor_flat, p_hor, reshape_2d=True)
print(T)


# MPC open loop

Ts_hor = 18 + zh # °C
Ts_hor = Ts_hor.reshape((nh, n_sys))
Ts_hor[nh//2:] = 22 # °C
Ts_hor = Ts_hor.reshape((nh * n_sys, 1))

c2.set_xyp(T0, Ts_hor, p_hor)

u_opt = c2.solve_u_opt(reshape_2d=True)

print(u_opt)

# MPC Closed loop # TO BE CONTINUED


n_sim = 30

Ts_fcast_arr = 18 + np.zeros(n_sim+nh) # °C
Ts_fcast_arr[n_sim//2:] = 22 # °C
Ts_fcast = mpc.Oracle(Ts_fcast_arr)

T_ext_fcast = mpc.ConstOracle([2, 2]) # °C



c2.set_oracles(Ts_fcast, T_ext_fcast)

# Broken line:
u, p, x, y, ys = c2.closed_loop_sim(T0, n_sim)
    
t_sim = np.arange(n_sim)*dt
