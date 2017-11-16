#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: RitaRamos
"""

'''
given a Q-function  Q and a state  x, 
selects a random action using the  ϵ-greedy policy
obtained from  Q for state  x,
'''

import random 
import numpy as np

def select_action(Q, state, n_actions, eps = 0.1):
    if(random.random() < eps): #with a small prob
        return random.randint(0, n_actions-1) #choose a random action.
    min_values= np.argwhere(Q[state,:] == np.amin(Q[state,:])) #Else:
    return min_values[random.randint(0, len(min_values)-1)][0]  #greedy action