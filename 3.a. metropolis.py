#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
Este script genera una muestra aleatoria con
PDF \propto x / (1 + x**2)**

usando el algoritmo de Metropolis
'''

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import uniform
from numpy.random import seed


def W(x):
    return x / (1 + x**2)**2

# def W(x):
#     return 3.5 * np.exp(-(x-3.)**2/3.) + 2 * np.exp(-(x+1.5)**2/0.5)

x = np.linspace(0, 8, 10000)

plt.figure(1)
plt.clf()
# plt.plot(x, W(x) / 13.251318549)
plt.plot(x, W(x) / 0.49)


plt.xlim(-2, 10)
plt.xlabel(r"$\{x_i\}$", fontsize=18)
plt.ylabel(r"$\propto W(x)$", fontsize=18)
# plt.show()
# plt.draw()


from scipy.integrate import trapz
print "La integral de W(x) es = {}".format(trapz(W(x), x))


# Metropolis
seed(123)

d = 1.
N_muestra = int(1e5)
x0 = 8

# un paso

def paso_metropolis(x0):
    r = uniform(low=-1, high=1)
    xp = x0 + d * r
    if W(xp) / W(x0) > uniform(0, 1):
        x0 = xp
    return x0

muestra = np.zeros(N_muestra)
muestra[0] = x0
for i in range(1, N_muestra):
    muestra[i] = paso_metropolis(muestra[i-1])


plt.hist(muestra, bins=np.arange(-4, 8, 0.1), normed=True, histtype='step')
plt.show()
plt.draw()
