from cvxopt import matrix, solvers
import numpy
"""
Optimisation centralisee version v1.0
"""

## variable definition

# max energy in kW
Umax = 12
u1m = 4
u2m = 7
u3m = 3

# thermal resistance
Rth1 = 1/float(u1m)
Rth2 = 1/float(u2m)
Rth3 = 1/float(u3m)

# Exterior temperature
Text = 20

# Ideal temperature in degrees
T1id = 25
T2id = 25
T3id = 25

# Ideal energy
u1id = (T1id - Text)/Rth1
print u1id

u2id = (T2id - Text)/Rth2
print u2id

u3id = (T3id - Text)/Rth3
print u3id

# comfort factor
alpha = 100

print(alpha)

## Quadratic problem resolution
## min 1/2*u'*P*u + q'u
##  st Gu < h
## and Au=b

# Matrix definition
P = matrix(numpy.diag([2*alpha*Rth1**2, 2*alpha*Rth2**2, 2*alpha*Rth3**2]), tc='d')
print(P)

q = matrix(numpy.array([1-2*alpha*(Rth1**2)*u1id, 1-2*alpha*(Rth2**2)*u2id, 1-2*alpha*(Rth3**2)*u3id]), tc='d')
print(q)

G = matrix(numpy.array([[1, 1, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]), tc='d')
print(G)

h = matrix(numpy.array([Umax, 0.0, 0.0, 0.0, u1m, u2m, u3m]), tc='d')
print(h)

# Resolution
sol = solvers.qp(P, q, G, h)

# Solution
print sol['x']

print(matrix([1, 1, 1]).T*sol['x'])

print type(sol['x'])
