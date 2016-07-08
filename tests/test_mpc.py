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
