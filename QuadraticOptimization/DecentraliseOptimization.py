from __future__ import division, print_function
from cvxopt import matrix, solvers
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math

from tabulate import tabulate

solvers.options['show_progress'] = False


"""
Distributed Optimisation  version v1.1

TODO = Predictive in version 2.0
"""

## variable definition

# number of users
"""To be change to take more users"""
n = 6
i = np.arange(n)

# pas
pas = 1.5

# mas iteration
Kmax = 1000

# threshold value
e = 1.0e-5


# max energy in kW
Umax = 10
"""To be change to take more users"""
u_m = np.array([3, 3, 3, 3, 3, 3])
assert len(u_m) == n

# thermal resistance
"""beta can be taken into account to modify the importance of Rth"""

beta = 6 # if higher than 6, the optimization process considers than it isn't necessary to spend enery on room with high Rth
Rth = beta*1./u_m

# Exterior temperature
Text = 10

# Ideal temperature in degrees
"""To be change to take more users"""
T_id = np.array([21, 21, 21, 21, 21, 21])

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
P = matrix(2*alpha*np.diag(Rth**2), tc='d')
#print(P)

q = matrix(1 - 2*alpha*u_id*(Rth**2), tc='d')
#print(q)

G = matrix(np.vstack((np.ones(n), -1*np.identity(n), np.identity(n))), tc='d')
#print(G)

h = matrix(np.hstack((Umax, np.zeros(n), u_m)), tc='d')
#print(h)



# init lambda
L = 0
u_sol = np.zeros(n)


for k in range(0, Kmax):



    for j in range(0, n):
        u = min(u_id[j], u_m[j])


        Pj = matrix([2*alpha*Rth[j]**2], tc='d')
        qj = matrix(np.array([1 - 2*alpha*Rth[j]**2 * u_id[j] + L]), tc='d')
        Gj = matrix(np.array([-1, 1]), tc='d')
        hj = matrix(np.array([0, u_m[j]]), tc='d')

        solj = solvers.qp(Pj, qj, Gj, hj)

        u_sol[j] = list(matrix([1]).T * solj['x'])[0]

    L = L + pas * (u_sol.sum() - Umax)

    print("iteration number %s." % (k+1))
    print("")


    if u_sol.sum()- Umax < e:
        print('break')
        break

#print(solution)

"""This table will print the temperature delta between the Text and T*, u_m, u_id and the optimal solution. The next line presents Jnrj(u)"""


table = tabulate([deltaT, u_m, u_id, u_sol], floatfmt=".10f")

print(table)
total = sum(u_sol)
print(total)


"""
PLOTTING
"""
# Graph colors (hexadecimal RRGGBB)
c = {
    'max':'#bbd5f0', # bleu clair
    'id':'#4582c2', # bleu
    'opt':'#f9d600' # jaune dore
}


fig, ax1 = plt.subplots(1,1)


ax1.bar(i, u_m,  label='$u_{max}$', # legend
        align='center',color=c['max'], linewidth=0)
ax1.bar(i, u_id, label='u_id',
        align='center', color=c['id'], width=0.5, linewidth=0)
ax1.plot(i, u_sol, 'D', label='u*',
         color=c['opt'], markersize=20)

ax1.set(
    title='Power allocation %s.' %sum(u_sol),
    xticks=i,
    xlabel='sub system',
    ylabel='heating power (kW)',
)

ax1.legend(loc='upper left', markerscale=0.4)

fig.tight_layout()

fig.savefig('power_bars.png', dpi=200, bbox_inches='tight')
fig.savefig('power_bars.pdf', bbox_inches='tight')
plt.show()


