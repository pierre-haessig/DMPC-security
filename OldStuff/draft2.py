from __future__ import division, print_function
from cvxopt import matrix, solvers
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from tabulate import tabulate


solvers.options['show_progress'] = False
"""

Centralized MPC Optimization  version v2.0


"""

## variable definition

# number of users
n = 6
i = np.arange(n)


# Definition of the horizon
p = 10

# max energy in kW
Umax = 10
"""To be change to take more users"""
u_m = np.array([3, 3, 3, 3, 3, 3])

# thermal resistance
"""beta can be taken into account to modify the importance of Rth"""

beta = 6 # if higher than 6, the optimization process considers than it isn't necessary to spend enery on room with high Rth
Rth = beta*1./u_m

# Exterior temperature
Text = 20

# Ideal temperature in degrees
"""To be change to take more users"""
T_id = np.array([21, 21, 28, 21, 25, 21])

# Ideal energy
deltaT = (T_id - Text)
u_id = (T_id - Text) * 1./Rth
#print(u_id)


# comfort factor
alpha = 100

#print(alpha)



""" Quadratic problem resolution
 min 1/2*u'*P*u + q'u
  st Gu < h
 and Au=b
"""

# Matrix definition
P = matrix(2*alpha*(Rth.T)*np.identity(n)*Rth, tc='d')
#print(P)

q = matrix(1 - 2*alpha*u_id*(Rth**2), tc='d')
#print(q)

G = matrix(np.vstack((np.zeros(n)+1, -1*np.identity(n), np.identity(n))), tc='d')
#print(G)

h = matrix(np.hstack((Umax, np.zeros(n), u_m)), tc='d')
#print(h)


# Resolution
sol = solvers.qp(P, q, G, h)

# Solution
print(sol['x'])

print(matrix(np.zeros(n)+1).T*sol['x'])

solution = np.asarray(list(sol['x']))

print(solution)

print(".")

table = tabulate([deltaT, u_m, u_id, solution],  floatfmt=".10f")

print(table)
total = sum(list(matrix(np.zeros(n)+1).T*sol['x']))
print(total)

"""
AFFICHAGE
"""
# graph colors (hexadecimal RRGGBB)
c = {
    'max':'#bbd5f0', # white blue
    'id':'#4582c2', # blue
    'opt':'#f9d600' # golden yellow
}


fig, ax1 = plt.subplots(1,1)


ax1.bar(i, u_m,  label='$u_{max}$', # legend
        align='center',color=c['max'], linewidth=0)
ax1.bar(i, u_id, label='u_id',
        align='center', color=c['id'], width=0.5, linewidth=0)
ax1.plot(i, sol['x'], 'D', label='u*',
         color=c['opt'], markersize=20)

ax1.set(
    title='Power allocation Umax = %s.' %total,
    xticks=i,
    xlabel='sub system',
    ylabel='heating power (kW)',
)

ax1.legend(loc='upper left', markerscale=0.4)

fig.tight_layout()

fig.savefig('power_bars.png', dpi=200, bbox_inches='tight')
fig.savefig('power_bars.pdf', bbox_inches='tight')
plt.show()


