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

def select_action(q, x, nA, eps = 0.1):
    if(random.random() < eps):
        return random.randint(0, nA-1)  
    min_values= np.argwhere(q[x,:] == np.amin(q[x,:]))
    return min_values[random.randint(0, len(min_values)-1)][0]