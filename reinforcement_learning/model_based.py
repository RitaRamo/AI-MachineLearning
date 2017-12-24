#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 22:11:44 2017

@author: RitaRamos"""

import Q_optimal_values
import e_greedy
import plot_charts
import numpy as np
import random


def get_Q(states, n_actions, Prob, c, gamma, initial_state,goal_state, n_interactions, plot_x_interactions, Q_optimal):
    n_states=len(states)
    C_estimated=np.zeros((n_states,n_actions))
    Q=np.zeros((n_states,n_actions))
    Prob_estimated={}
    for action in range(n_actions):
        Prob_estimated[action]=np.eye(n_states)
    current_state=states.index(initial_state)
    goal=states.index(goal_state)
    visits=np.zeros((n_states,n_actions))
    chart_norm=np.zeros(int(n_interactions/plot_x_interactions))
    counter_chart=0
    #Q_optimal=Q_optimal_values.get_Q(n_states, n_actions, Prob, c, gamma)
    for i in range(n_interactions):  
        action=e_greedy.select_action(Q,current_state, n_actions)
        step=1/(visits[current_state,action] + 1)
        visits[current_state,action] += 1
        C_estimated[current_state,action]=C_estimated[current_state,action]+step*(c[current_state, action]- C_estimated[current_state,action])
        
        next_state=np.random.choice(n_states,p=Prob[action][current_state,:])
        for s in range(n_states):  
            if(s== next_state):
                Prob_estimated[action][current_state, s]=Prob_estimated[action][current_state, s] + step*(1.0-Prob_estimated[action][current_state, s])
            else:
                Prob_estimated[action][current_state, s]=Prob_estimated[action][current_state, s] + step*(-Prob_estimated[action][current_state, s])
        
        Qt_new=C_estimated[current_state,action] + gamma*np.sum( (Prob_estimated[action][current_state, :])*np.min(Q, axis=1))
        Q[current_state, action]=Qt_new
        
        if current_state==goal:
            current_state=random.randint(0, n_states-1)
        else:
            current_state=next_state
    
        chart_norm, counter_chart=plot_charts.savePlots(i,plot_x_interactions, chart_norm, counter_chart, Q_optimal, Q)
        
    return Q, chart_norm, counter_chart


