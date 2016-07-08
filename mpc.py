#!/usr/bin/python
# -*- coding: utf-8 -*-
# Pierre Haessig — July 2016
""" A toolbox for Model Predictive Control (MPC) and Distributed MPC.
"""

from __future__ import division, print_function, unicode_literals
import numpy as np


class LinDyn(object):
    "Dynamics of a discrete time linear system (LTI)"
    def __init__(self, A, Bu, Bp, C, name=None):
        '''
        A, Bu, Bp, C: 2d arrays representing dynamics
        
          x⁺ = A.x + Bᵤ.u + Bₚ.p
          y  = C.x
        
        A:  (nx,nx) array
        Bu: (nx,nu) array
        Bu: (nx,np) array
        C:  (ny,nx) array
        '''
        self.name = name
        
        nx = A.shape[0]
        assert A.shape[0] == A.shape[1]
        
        nu = Bu.shape[1]
        assert Bu.shape[0] == A.shape[0]
        
        np = Bp.shape[1]
        assert Bp.shape[0] == A.shape[0]
        
        ny = C.shape[0]
        assert C.shape[1] == A.shape[1]
        
        
        self.A = A
        self.Bu = Bu
        self.Bp = Bp
        self.C = C
        
        self.nx = nx
        self.nu = nu
        self.np = np
        self.ny = ny

    def get_AB(self):
        'stacked A,Bu,Bp matrices'
        return np.hstack([self.A, self.Bu, self.Bp])
    
    def __str__(self):
        name = '' if self.name is None else "'{}'".format(self.name)
        return 'LinDyn {name} \n  dims x:{0.nx}, u:{0.nu}, p:{0.np}, y:{0.ny}'.format(self, name=name)


def dyn_from_thermal(r_th, c_th, dt, name=None):
    r_th = np.asarray(r_th, dtype=float).ravel()
    c_th = np.asarray(c_th, dtype=float).ravel()
    assert r_th.shape == c_th.shape
    
    tau = r_th*c_th

    A = np.diag(1 -dt/tau)
    Bu = np.diag(dt/c_th)
    B_Text = np.diag(dt/tau)
    C = np.atleast_2d(1.)
    
    return LinDyn(A, Bu, B_Text, C, name=name)


def block_toeplitz(c, r=None):
    '''
    Construct a block Toeplitz matrix, with blocks having the same shape
    
    Signature is compatible with ``scipy.linalg.toeplitz``
    
    Parameters
    ----------
    c : list of 2d arrays
        First column of the matrix.
        Each item of the list should have same shape (mb,nb)
    r : list of 2d arrays
        First row of the matrix. If None, ``r = conjugate(c)`` is assumed;
        in this case, if c[0] is real, the result is a Hermitian matrix.
        r[0] is ignored; the first row of the returned matrix is
        made of blocks ``[c[0], r[1:]]``.
    
    c and r can also be lists of scalars; if so they will be broadcasted
    to the fill the blocks
    
    Returns
    -------
    A : (len(c)*mb, len(r)*nb) ndarray
        The block Toeplitz matrix.
    
    Examples
    --------
    Compatible with ``scipy.linalg.toeplitz`` (but less optimized):
    >>> block_toeplitz([1,2,3], [1,4,5,6])
    array([[1, 4, 5, 6],
           [2, 1, 4, 5],
           [3, 2, 1, 4]])
    
    Regular usage:
    >>> I2 = np.eye(2)
    >>> block_toeplitz([1*I2,2*I2,3*I2], [1*I2,4*I2,5*I2,6*I2])
    array([[1, 0, 4, 0, 5, 0, 6, 0],
           [0, 1, 0, 4, 0, 5, 0, 6],
           [2, 0, 1, 0, 4, 0, 5, 0],
           [0, 2, 0, 1, 0, 4, 0, 5],
           [3, 0, 2, 0, 1, 0, 4, 0],
           [0, 3, 0, 2, 0, 1, 0, 4]])
    
    Usage with broadcasting of scalar blocks:
    >>> block_toeplitz([1*I2,2*I2,3*I2], [1,4,5,6])
    array([[1, 0, 4, 4, 5, 5, 6, 6],
           [0, 1, 4, 4, 5, 5, 6, 6],
           [2, 0, 1, 0, 4, 4, 5, 5],
           [0, 2, 0, 1, 4, 4, 5, 5],
           [3, 0, 2, 0, 1, 0, 4, 4],
           [0, 3, 0, 2, 0, 1, 4, 4]])
    '''
    c = [np.atleast_2d(ci) for ci in c]
    if r is None:
        r = [np.conj(ci) for ci in c]
    else:
        r = [np.atleast_2d(rj) for rj in r]
    
    mb,nb = c[0].shape
    dtype = (c[0]+r[0]).dtype
    m = len(c)
    n = len(r)
    
    A = np.zeros((m*mb, n*nb), dtype=dtype)
    
    for i in range(m):
        for j in range(n):
            # 1. select the Aij block from c or r:
            d = i-j
            if d>=0:
                Aij = c[d]
            else:
                Aij = r[-d]
            # 2. paste the block
            A[i*mb:(i+1)*mb, j*mb:(j+1)*mb] = Aij
    
    return A


class MPC(object):
    '''MPC controller
    
    with cost function c'.u + α ‖y-y*‖²
    constraints u_min ≤ u ≤ u_max
    '''
    def __init__(self, dyn, nh, u_min, u_max, u_cost, track_weight):
        self.dyn = dyn
        self.nh = nh
        self.u_min = u_min
        self.u_max = u_max
        self.u_cost = u_cost
        self.track_weight = track_weight
        
        # Prediction matrices
        F, Hu, Hp = self.pred_mat(nh, dyn.A, dyn.C, dyn.Bu, dyn.Bp)
        self.F = F
        self.Hu = Hu
        self.Hp = Hp
        
        # Precompute quadprog matrices which are independent of time
        self._update_P()
        self._update_Gh()
    
    @staticmethod
    def pred_mat(n, A, C, *B_list):
        '''
        Construct prediction matrices F, H for horizon n
        such that y = Fx + H.u
        
        Any number of B matrices can be given, which will return an equal
        number of H matrices
        
        TODO: implement case C≠[1]
        
        Examples
        --------
        >>> F, H = MPC.pred_mat(3, np.atleast_2d(0.9), np.atleast_2d(1), np.atleast_2d(0.2))
        >>> F
        array([[ 0.9  ],
               [ 0.81 ],
               [ 0.729]])
        >>> H
        array([[ 0.2  ,  0.   ,  0.   ],
               [ 0.18 ,  0.2  ,  0.   ],
               [ 0.162,  0.18 ,  0.2  ]])
        with 2 B matrices:
        >>> F, Hu, Hp = MPC.pred_mat(3, np.atleast_2d(0.9), np.atleast_2d(1), np.atleast_2d(0.2), np.atleast_2d(0.5))
        >>> Hp
        array([[ 0.5  ,  0.   ,  0.   ],
               [ 0.45 ,  0.5  ,  0.   ],
               [ 0.405,  0.45 ,  0.5  ]])
        '''
        # [A^i] for i=1:n
        A_pow = [A]
        for i in range(n-1):
            A_pow.append(A_pow[-1].dot(A))
        F = np.vstack(A_pow) 

        H_list = []
        for B in B_list:
            # [A^i.Bu] for i=0:n-1
            AB_pow = [B]
            AB_pow.extend([Ai.dot(B) for Ai in A_pow[:-1]])
            
            H = block_toeplitz(AB_pow, np.zeros(n))
            H_list.append(H)
        
        return (F,) + tuple(H_list)
    
    def _update_P(self):
        track_weight = self.track_weight
        Hu = self.Hu
        self.P = 2*track_weight * (Hu.T).dot(Hu)
    
    def _update_q(self, u_cost, x0=None, ys_hor=None, p_hor=None):
        if x0 is not None:
            track_weight = self.track_weight
            F = self.F
            Hu = self.Hu
            Hp = self.Hp
            F_x = F.dot(x0)
            Hp_p = Hp.dot(p_hor)
            self._q0 = 2*track_weight*(Hu.T).dot(F_x + Hp_p - ys_hor)
        
        self.q = u_cost + self._q0
    
    def _update_j0(self, x0, ys_hor, p_hor):
        j0 = ys_hor.T.dot(ys_hor - 2*(F_x + Hp_p)) + \
             F_x.T.dot(F_x) + Hp_p.T.dot(Hp_p) + 2*F_x.T.dot(Hp_p)
        self.j0 = j0*track_weight
    
    def _update_Gh(self):
        F = self.F
        u_max = self.u_max
        # TODO: fix u_min != 0
        n = F.shape[0]
        In = np.identity(n)
        self.G = np.vstack((-In, In))
        self.h = np.hstack((np.zeros(n), np.ones(n) * u_max))
    
#    @staticmethod
#    def qp_mat(F, Hu, Hp, x0, u_cost, track_weight):
#        '''
#        Construct quadprog matrices P, q, j0, G, h
#        
#        corresponding to ``cvxopt.solvers.qp`` notation, with constant:
#        
#            minimize    (1/2) x'.P.x + q'.x + j0
#            subject to  G.x <= h
#        '''
#        # P
#        P = 2*track_weight * (Hu.T).dot(Hu)
#        # q
#        F_x = F.dot(x0)
#        Hp_p = Hp.dot(p_hor)
#        q = u_cost + 2*track_weight*(Hu.T).dot(F_x + Hp_p - ys_hor)
#        
#        #j0
#        j0 = ys_hor.T.dot(ys_hor - 2*(F_x + Hp_p)) + \
#             F_x.T.dot(F_x) + Hp_p.T.dot(Hp_p) + 2*F_x.T.dot(Hp_p)
#        j0 = j0*track_weight
#        # G
#        n = F.shape[0]
#        In = np.identity(n)
#        G = np.vstack((-In, In))
#        # h
#        h = np.hstack((np.zeros(n), np.ones(n) * u_max))
#        
#        return P, q, j0, G, h
    
    def set_xyp(self, x0, ys_hor, p_hor):
        '''sets the state measurement x0, and forecasts on the horizon
        set point ys_hor and perturbation p_hor
        '''
        u_cost = self._u_cost
        self._update_q(u_cost, x0, ys_hor, p_hor)
        self._update_j0(u_cost, x0, ys_hor, p_hor)
    
    def set_u_cost(self, u_cost):
        '''recompute quadprog matrices with new cost for u'''
        self._u_cost = u_cost
        self._update_q(u_cost)
    
    def solve_u_opt(self):
        ''''''
        u_opt = solvers.qp(self.P, self.q, self.G, self.h)
        return u_opt


if __name__ == '__main__':
    A = np.array([[1]], dtype=float)
    Bu = np.array([[1]], dtype=float)
    Bp = np.array([[1]], dtype=float)
    C = np.array([[1]], dtype=float)
    
    dyn = LinDyn(A, Bu, Bp, C, name='SISO')
    print(dyn)
    
    dyn = dyn_from_thermal(5, 1, 0.1)
    
    mpc = MPC(dyn, nh=3, u_min=0, u_max=1, u_cost=1, track_weight=100)
    
    MPC.pred_mat(2, dyn.A, dyn.C, dyn.Bu, dyn.Bp)
