#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 18:13:14 2017

@author: RitaRamos
"""
import e_greedy
import plot_charts
import numpy as np
import random



def get_Q_learning(states, n_actions, Prob, c, gamma, initial_state,goal_state, n_interactions, plot_x_interactions, step,Q_optimal):
    n_states=len(states)
    Q_learning=np.zeros((n_states,n_actions))
    current_state=states.index(initial_state)
    goal=states.index(goal_state)
    chart_norm=np.zeros(int(n_interactions/plot_x_interactions))
    counter_chart=0
    for i in range(n_interactions):  
        action=e_greedy.select_action(Q_learning,current_state,n_actions)
    
        next_state=np.random.choice(n_states,p=Prob[action][current_state,:])
        
        qmax=np.max(Q_learning[next_state, :])
        Qt_new=Q_learning[current_state, action] + step*(c[current_state, action]+gamma*qmax -  Q_learning[current_state, action])
        Q_learning[current_state, action]=Qt_new
        
        if current_state==goal:
            current_state=random.randint(0, n_states-1)
        else:
            current_state=next_state
          
        chart_norm, counter_chart=plot_charts.savePlots(i,plot_x_interactions, chart_norm, counter_chart, Q_optimal, Q_learning)

    return Q_learning, chart_norm, counter_chart
    