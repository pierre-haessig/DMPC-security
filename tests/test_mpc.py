#!/usr/bin/python
# -*- coding: utf-8 -*-
# Pierre Haessig — July 2016
""" test the MPC toolbox

"""

from __future__ import division, print_function, unicode_literals
from nose.tools import assert_true, assert_equal
from numpy.testing import assert_allclose
import numpy as np

def assert_allclose9(a,b):
    return assert_allclose(a, b, 1e-9, 1e-9)

import mpc

def test_dyn_from_thermal():
    dyn = mpc.dyn_from_thermal(5, 1, dt=0.1)
    assert_equal(dyn.A, 0.98)
    assert_equal(dyn.Bu, 0.1)
    assert_equal(dyn.Bp, 0.02)
    assert_equal(dyn.C, 1)

def test_block_toeplitz():
    assert_allclose9(
        mpc.block_toeplitz([1,2,3], [1,4,5,6]),
        np.array([[1, 4, 5, 6],
                  [2, 1, 4, 5],
                  [3, 2, 1, 4]])
        )
    
    I2 = np.eye(2)
    assert_allclose9(
        mpc.block_toeplitz([1*I2,2*I2,3*I2], [1*I2,4*I2,5*I2,6*I2]),
        np.array([[1, 0, 4, 0, 5, 0, 6, 0],
                  [0, 1, 0, 4, 0, 5, 0, 6],
                  [2, 0, 1, 0, 4, 0, 5, 0],
                  [0, 2, 0, 1, 0, 4, 0, 5],
                  [3, 0, 2, 0, 1, 0, 4, 0],
                  [0, 3, 0, 2, 0, 1, 0, 4]])
        )
    
    assert_allclose9(
        mpc.block_toeplitz([1*I2,2*I2,3*I2], [1,4,5,6]),
        np.array([[1, 0, 4, 4, 5, 5, 6, 6],
                  [0, 1, 4, 4, 5, 5, 6, 6],
                  [2, 0, 1, 0, 4, 4, 5, 5],
                  [0, 2, 0, 1, 4, 4, 5, 5],
                  [3, 0, 2, 0, 1, 0, 4, 4],
                  [0, 3, 0, 2, 0, 1, 4, 4]])
        )

def test_pred_mat():
    '''test prediction matrices on a 1D thermal system'''
    r_th = 20
    c_th= 0.02
    assert r_th * c_th == 0.4 # h
    dt = 0.2 #h
    dyn = mpc.dyn_from_thermal(r_th, c_th, dt, "thermal subsys")

    n_hor = int(2.5/dt)
    assert n_hor == 12
    t = np.arange(1, n_hor+1)*dt
    F, Hu, Hp = mpc.pred_mat(n_hor, dyn.A, dyn.C, dyn.Bu, dyn.Bp)

    zn = np.zeros(n_hor)[:,None]
    T_ext_hor = 2 + zn # °C
    u_hor = 0 + zn # kW
    u_hor[t>1] = 1 #kW
    T0 = 20 # °C

    T_hor = np.dot(F, T0) + np.dot(Hu,u_hor) + np.dot(Hp, T_ext_hor)
    
    assert_equal(T_hor.shape, (12,1))
    
    assert_allclose9(
        T_hor,
        np.array([[ 11.        ],
                  [  6.5       ],
                  [  4.25      ],
                  [  3.125     ],
                  [  2.5625    ],
                  [ 12.28125   ], # u becomes 1: T goes up
                  [ 17.140625  ],
                  [ 19.5703125 ],
                  [ 20.78515625],
                  [ 21.39257812],
                  [ 21.69628906],
                  [ 21.84814453]])
        )
    # Also check the method of LinDyn
    F1, Hu1, Hp1 = dyn.pred_mat(n_hor)
    assert_true(np.all(F==F1))
    assert_true(np.all(Hu==Hu1))
    assert_true(np.all(Hp==Hp1))
