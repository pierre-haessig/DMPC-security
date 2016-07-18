#!/usr/bin/python
# -*- coding: utf-8 -*-
# Pierre Haessig — July 2016
""" Usage example of the `mpc` module with a 2d thermal system
(the system represents 2 independant rooms to be heated)
"""

from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt

import mpc

## System definition
n_sys = 2
dt = 0.1
dyn2 = mpc.dyn_from_thermal([20,20], [0.05, 0.05], dt)

# Controler:
nh = 20 # MPC horizon, as a number of timesteps
c2 = mpc.MPC(dyn2, nh, u_min=0, u_max=1.5, u_cost=1, track_weight=100)
# (breaks when u_max or track_weight is set to a list)


## Output prediction

zh = np.zeros((nh*n_sys,1))
T_ext_hor = 2+zh # °C

u_hor = np.zeros((nh, n_sys))
u_hor[10:] = 1

u_hor_flat = u_hor.reshape((-1, 1))

T0 = np.array([20,10]).reshape((n_sys,1))

T_pred = c2.pred_output(T0, u_hor_flat, T_ext_hor, reshape_2d=True)
print(T_pred)


## MPC open loop

Ts_hor = 18 + zh # °C
Ts_hor = Ts_hor.reshape((nh, n_sys))
Ts_hor[nh//2:] = 22 # °C
Ts_hor = Ts_hor.reshape((nh * n_sys, 1))

c2.set_xyp(T0, Ts_hor, T_ext_hor)

u_opt = c2.solve_u_opt(reshape_2d=True)

print(u_opt)


## MPC Closed loop

n_sim = 30

Ts_fcast_arr = 18 + np.zeros((n_sim+nh, n_sys)) # °C
Ts_fcast_arr[n_sim//2:] = 22 # °C
Ts_fcast = mpc.Oracle(Ts_fcast_arr)

T_ext_fcast = mpc.ConstOracle([2, 2]) # °C

c2.set_oracles(Ts_fcast, T_ext_fcast)

u, p, x, T_sim, Ts = c2.closed_loop_sim(T0, n_sim)

# plot the closed loop simulation
t_sim = np.arange(n_sim)*dt

fig, (ax1, ax2) = plt.subplots(2,1, sharex=True)

col = ['b', 'g']

for i in range(n_sys):
    ax1.plot(t_sim, Ts[:,i], 'k:', label=r'$T^{{set}}_{}$'.format(i+1))
    ax1.plot(t_sim, T_sim[:,i], '+-', label=r'$T_{}$'.format(i+1), color=col[i])
    ax1.plot(0, T0[i], 'D', color=col[i])
    
    ax2.plot(t_sim, u[:,i], '+-', label=r'$u_{}$'.format(i+1), color=col[i])

ax1.set(
    title = 'MPC in closed loop (dt={:.1f}h, nh={}, n_sim={})'.format(dt,nh, n_sim),
    ylabel = u'temperature (°C)'
    )
ax1.legend(ncol=2, loc='lower right')

ax2.set(
    xlabel = 'time (h)',
    ylabel = u'heating (kW)'
    )
ax2.legend(ncol=2, loc='lower right')

plt.show()
