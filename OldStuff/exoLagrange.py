from cvxopt import *
import numpy as np
import sympy as sp
from scipy.optimize import fsolve


# definition of the lagrangian function
def func(X):
    x1 = X[0]
    x2 = X[1]
    L = X[2]

    return x1**2 - 2*x1 + 1 + 2*x2**2 - 4*x2 + 2 + L * (x1 + x2 - 1)

# derivate of the lagrangian function
def dfunc(X) :
    dLambda = np.zeros(len(X))
    h = 1e-3
    for i in range(len(X)):
        dX = np.zeros(len(X))
        dX[i] = h
        dLambda[i] = (func(X+dX)-func(X-dX))/(2*h)
    return dLambda

# this is the min
X1 = fsolve(dfunc, [0, 0, 0])
print X1, func(X1)

