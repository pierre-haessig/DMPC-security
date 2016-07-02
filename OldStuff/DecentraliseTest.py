from cvxopt import matrix, solvers
import numpy
import math
from decimal import *


solvers.options['show_progress'] = False

"""
Optimisation decentralisee version v1.0

"""


## number of user
n = 3

## variable definition

# pas
pas = 0.1

# mas iteration
Kmax = 1000

# threshold value
e = 1.0e-8

# max energy in kW
Umax = 10
um = numpy.zeros(n)
um[0] = 4
um[1] = 7
um[2] = 3

# thermal resistance
Rth = numpy.zeros(n)
Rth[0] = 1/float(um[0])
Rth[1] = 1/float(um[1])
Rth[2] = 1/float(um[2])

# Exterior temperature
Text = 20

# Ideal temperature in degrees
Tid = numpy.zeros(n)
Tid[0] = 25
Tid[1] = 25
Tid[2] = 20

# Ideal energy
uid = numpy.zeros(n)

uid[0] = (Tid[0] - Text)/Rth[0]
uid[1] = (Tid[1] - Text)/Rth[1]
uid[2] = (Tid[2] - Text)/Rth[2]

# comfort factor
alpha = 100

## Quadratic problem resolution
## min 1/2*u'*P*u + q'u
##  st Gu < h
## and Au=b

# Matrix definition
P = matrix(numpy.diag([2*alpha*Rth[0]**2, 2*alpha*Rth[1]**2, 2*alpha*Rth[2]**2]), tc='d')

q = matrix(numpy.array([1-2*alpha*(Rth[0]**2)*uid[0]**2, 1-2*alpha*(Rth[1]**2)*uid[1]**2, 1-2*alpha*(Rth[2]**2)*uid[2]**2]), tc='d')

G = matrix(numpy.array([[1, 1, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]), tc='d')

h = matrix(numpy.array([Umax, 0.0, 0.0, 0.0, um[0], um[1], um[2]]), tc='d')


## distributed optimization. with Uzawa method

# init lambda
L = 0
uOpt = numpy.zeros(n)


# Recursion

for k in range(0, Kmax):



    for j in range(0, n):

        Pj = matrix([2*alpha*Rth[j]**2], tc='d')
        qj = matrix(numpy.array([1 - 2*Rth[j]**2 * uid[j] + L]), tc='d')
        Gj = matrix(numpy.array([-1, 1]), tc='d')
        hj = matrix(numpy.array([0, um[j]]), tc='d')

        solj = solvers.qp(Pj, qj, Gj, hj)

        uOpt[j] = list(matrix([1]).T*solj['x'])[0]

    L = L + pas * (sum(uOpt) - Umax )

    k = k+1

    print "iteration number %s." % k
    print "."


    if math.fabs(list(matrix([1, 1, 1]).T*solj['x'] - Umax)[0]) < e:
        print('break')
        break



print "value of the lagrangian multiplier L =%s." % L
print "."
print "optimal vector U =%s. " % uOpt
print "."
print "total consumption %s." % sum(uOpt)
print "."




