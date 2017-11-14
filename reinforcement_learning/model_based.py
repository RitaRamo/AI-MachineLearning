#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 22:11:44 2017

@author: RitaRamos"""

import Q_optimal_values
import e_greedy
import numpy as np
import random


def get_Q(X, nA, Prob, c, gamma, initial_state,goal_state,  n_interactions, plot_x_interactions):
    nX=len(X)
    C_estimated=np.zeros((nX,nA))
    Q=np.zeros((nX,nA))
    Prob_estimated={}
    for action in range(nA):
        Prob_estimated[action]=np.eye(nX)
    currentState=X.index(initial_state)
    visits=np.zeros((nX,nA))
    chart_norm=np.zeros(int(n_interactions/plot_x_interactions))
    index=0
    Q_optimal=Q_optimal_values.get_Q(nX, nA, Prob, c, gamma)
    for i in range(n_interactions):  
        action=e_greedy.select_action(Q,currentState, nA)
        
        step=1/(visits[currentState,action] + 1)
        visits[currentState,action] += 1
        
        C_estimated[currentState,action] = C_estimated[currentState,action]+ step*(c[currentState, action]- C_estimated[currentState,action])   #ponho gama
        
        next_state=np.random.choice(nX,p=Prob[action][currentState,:])
        
        for s in range(nX):  
            if(s== next_state):
                Prob_estimated[action][currentState, s]=Prob_estimated[action][currentState, s] + step*(1.0-Prob_estimated[action][currentState, s])
            else:
                Prob_estimated[action][currentState, s]=Prob_estimated[action][currentState, s] + step*(-Prob_estimated[action][currentState, s])
        
        Qt_new=C_estimated[currentState,action] + gamma*np.sum( (Prob_estimated[action][currentState, :])*np.min(Q, axis=1))
        Q[currentState, action]=Qt_new
        
    
        if currentState==X.index(goal_state):
            currentState=random.randint(0, nX-1)
        else:
            currentState=next_state
    
        
        if(i%plot_x_interactions==0):
            chart_norm[index]=np.linalg.norm(Q_optimal-Q)
            index+=1
    
    return Q, chart_norm, index


