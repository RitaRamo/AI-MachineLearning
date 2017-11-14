#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 23:26:13 2017

@author: RitaRamos
"""

'''Optimal Q-function for a given MDP, using value iteration.'''

import numpy as np

def get_Q( nX, nA, Prob, c, gamma):
    J=np.zeros((nX,1))
    err1=1
    qs=[]
    while err1 > 1e-8:
        qs=[c[:,i].reshape((nX,1))+gamma*Prob[i].dot(J) for i in range(nA)]
        Qmin=np.min(qs,axis=0)
        Jnew=Qmin
        err1=np.linalg.norm(Jnew-J)
        J=Jnew  
    return np.concatenate((qs), axis=1)
