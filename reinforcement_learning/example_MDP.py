#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 23:51:45 2017

@author: RitaRamos
"""

import numpy as np
import matplotlib.pyplot as plt
import model_based



WIND = (0, 0, 0, 1, 1, 1, 2, 2, 1, 0)
nrows = 7
ncols = 10
init = [3, 0]
goal = [3, 7]

# States
X = [[x, y] for x in range(nrows) for y in range(ncols)]
nX = len(X)

# Actions
A = ['U', 'D', 'L', 'R']
nA = len(A)

# Transition probabilities
P = dict()
P[0] = np.zeros((nX, nX))
P[1] = np.zeros((nX, nX))
P[2] = np.zeros((nX, nX))
P[3] = np.zeros((nX, nX))

for i in range(len(X)):
    x = X[i]
    y = dict()
    
    y[0] = [x[0] - WIND[x[1]] - 1, x[1]]
    y[1] = [x[0] - WIND[x[1]] + 1, x[1]]
    y[2] = [x[0] - WIND[x[1]], x[1] - 1]
    y[3] = [x[0] - WIND[x[1]], x[1] + 1]
    
    for k in y:
        y[k][0] = max(min(y[k][0], nrows - 1), 0)
        y[k][1] = max(min(y[k][1], ncols - 1), 0)
        j = X.index(y[k])
        P[k][i, j] = 1

c = np.ones((nX, nA))
c[X.index(goal), :] = 0

gamma = 0.99

# -- Pretty print

print('\n- MDP problem specification: -\n')

print('States:')
print(np.array(X))

print('\nActions:')
print(A)

print('\nTransition probabilities:')
for a in range(nA):
    print('Action', a)
    print(P[a])
    
print('\ncost:')
print(c)

print('\nStart state:', init)
print('\nGoal state:', goal)


Q, chart_values, index=model_based.get_Q(X, nA, P, c, gamma, init,goal,  100000, 500)
print("Q:\n", Q)
plt.figure(1)
plt.plot(np.arange(0,index), chart_values)
plt.xlabel('Steps')
plt.ylabel('Norm')
plt.title('Model-based')
plt.show()