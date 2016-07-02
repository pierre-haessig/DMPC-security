from __future__ import division, print_function
from cvxopt import matrix, solvers
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math

from tabulate import tabulate

solvers.options['show_progress'] = False


"""

Optimisation Distribuee version v1.1

Mise en place de la notation vectorielle

Mise en place du tableau et des schemas

"""

## variable definition

# number of users
n = 3
i = np.arange(n)

# pas
pas = 1.5

# mas iteration
Kmax = 1000

# threshold value
e = 1.0e-5


# max energy in kW
Umax = 10
u_m = np.array([4, 7, 3])

# thermal resistance
"""On peut ouer sur le parametre beta"""

beta = 6 # au dela de 6 l'optimisation considere qu cela ne sert a rien de trop chauffer
Rth = beta*1./u_m

# Exterior temperature
Text = 0

# Ideal temperature in degrees
T_id = np.array([25, 25, 25])

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



# init lambda
L = 0
uOpt = np.zeros(n)


for k in range(0, Kmax):



    for j in range(0, n):

        Pj = matrix([2*alpha*Rth[j]**2], tc='d')
        qj = matrix(np.array([1 - 2*Rth[j]**2 * u_id[j] + L]), tc='d')
        Gj = matrix(np.array([-1, 1]), tc='d')
        hj = matrix(np.array([0, u_m[j]]), tc='d')

        solj = solvers.qp(Pj, qj, Gj, hj)

        uOpt[j] = list(matrix([1]).T*solj['x'])[0]

    L = L + pas * (sum(uOpt) - Umax )

    k = k+1

    print("iteration number %s." % k)
    print(".")


    if math.fabs(list(matrix([1, 1, 1]).T*solj['x'] - Umax)[0]) < e:
        print('break')
        break





# Solution
print(solj['x'])

print(matrix([1, 1, 1]).T*uOpt)

solution = np.asarray(list(uOpt))

print(solution)

print(".")

table = tabulate([deltaT, u_m, u_id, solution],  floatfmt=".10f")

print(table)
print(sum(uOpt))










"""
AFFICHAGE
"""
# couleurs pour le graph (hexadecimal RRGGBB)
c = {
    'max':'#bbd5f0', # bleu clair
    'id':'#4582c2', # bleu
    'opt':'#f9d600' # jaune dore
}


fig, ax1 = plt.subplots(1,1)


ax1.bar(i, u_m,  label='$u_{max}$', # on peut mettre du tex dans les 'label', cad la legende
        align='center',color=c['max'], linewidth=0)
ax1.bar(i, u_id, label='u_id',
        align='center', color=c['id'], width=0.5, linewidth=0)
ax1.plot(i, uOpt, 'D', label='u*',
         color=c['opt'], markersize=20)

ax1.set(
    title='Allocation de puissance',
    xticks=i,
    xlabel='sub system',
    ylabel='heating power (kW)',
)

ax1.legend(loc='upper left', markerscale=0.4)

fig.tight_layout()

fig.savefig('power_bars.png', dpi=200, bbox_inches='tight')
fig.savefig('power_bars.pdf', bbox_inches='tight')
plt.show()


