#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 23:26:13 2017

@author: RitaRamos
"""

'''Optimal Q-function for a given MDP, using value iteration.'''

import numpy as np

def get_Q(n_states, n_actions,Prob, c, gamma):
    J=np.zeros((n_states,1))
    err1=1
    Q=[]
    while err1 > 1e-8:
        Q=[c[:,i].reshape((n_states,1))+gamma*Prob[i].dot(J) for i in range(n_actions)]
        Qmin=np.min(Q,axis=0)
        Jnew=Qmin
        err1=np.linalg.norm(Jnew-J)
        J=Jnew  
    return np.concatenate((Q), axis=1)
