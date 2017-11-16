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
states = [[x, y] for x in range(nrows) for y in range(ncols)]
n_states = len(states)

# Actions
actions = ['U', 'D', 'L', 'R']
n_actions = len(actions)

# Transition probabilities
P = dict()
for action in range(n_actions):
    P[action] = np.zeros((n_states, n_states))

for i in range(len(states)):
    x = states[i]
    y = dict()
    
    y[0] = [x[0] - WIND[x[1]] - 1, x[1]]  
    y[1] = [x[0] - WIND[x[1]] + 1, x[1]]
    y[2] = [x[0] - WIND[x[1]], x[1] - 1]
    y[3] = [x[0] - WIND[x[1]], x[1] + 1]
    
    for k in y:
        y[k][0] = max(min(y[k][0], nrows - 1), 0)
        y[k][1] = max(min(y[k][1], ncols - 1), 0)
        j = states.index(y[k])
        P[k][i, j] = 1

c = np.ones((n_states, n_actions))
c[states.index(goal), :] = 0

gamma = 0.99



print('\n- MDP problem specification: -\n')

print('States:')
print(np.array(states))

print('\nActions:')
print(actions)

print('\nTransition probabilities:')
for a in range(n_actions):
    print('Action', a)
    print(P[a])
    
print('\ncost:')
print(c)

print('\nStart state:', init)
print('\nGoal state:', goal)


Q, chart_values, index=model_based.get_Q(states, n_actions, P, c, gamma, init,goal,  100000, 500)
print("Q:\n", Q)
plt.figure(1)
plt.plot(np.arange(0,index), chart_values)
plt.xlabel('Steps')
plt.ylabel('Norm')
plt.title('Model-based')
plt.show()