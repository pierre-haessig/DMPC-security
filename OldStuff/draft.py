#!/usr/bin/python
# -*- coding: utf-8 -*-
# Pierre Haessig — June 2016
""" Qq lignes pour créer, afficher et tracer
des vecteurs issus de l'optim statique

"""

from __future__ import division, print_function
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


n = 3
# vecteur constant:
u_m = np.zeros(n)+3
# vecteurs custom
u_i = np.array([2,2,2.5])
u = np.array([1,1.9,2])

i = np.arange(n) # version numpy de range(n) qui renvoie un array

# Les tableaux numpy peuvent servir en calcul terme à terme (pas besoind de "for")
(u_m + u_i*i)**2 # (calcul bidon).
# Rem: l'opérateur puissance se note "**" et pas "^"


# Empilement vertical (pour affichage, ou pour fabriquer des matrices par bloc)
# 2 écritures au résultat identique:
tab = np.vstack((u, u_m))
tab = np.vstack((u[None,:], u_m[None,:]))
# (en fait l'astuce dont je parlais pour avoir un tableau 2D
#  n'est pas nécessaire avec vstack, mais elle l'est pour hstack)
print(tab)




### Représentation graphique (essai) ###

# petite touche de style (facultative) : grille en gris clair
""" mpl.style.use('whitegrid.mplstyle') # nécessite le fichier idoine
"""

# couleurs pour le graph (hexadécimal RRGGBB)
c = {
    'max':'#bbd5f0', # bleu clair
    'id':'#4582c2', # bleu
    'opt':'#f9d600' # jaune doré
}


fig, ax1 = plt.subplots(1,1)


ax1.bar(i,u_m,  label='$u_{max}$', # on peut mettre du tex dans les 'label', cad la légende
        align='center',color=c['max'], linewidth=0)
ax1.bar(i,u_i, label='u_id',
        align='center', color=c['id'], width=0.5, linewidth=0)
ax1.plot(i,u, 'D', label='u*',
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



