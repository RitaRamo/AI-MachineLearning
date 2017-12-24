#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 19:30:43 2017

@author: RitaRamos
"""

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



def get_Q_SARSA(states, n_actions, Prob, c, gamma, initial_state,goal_state, n_interactions, plot_x_interactions, Q_optimal):
    n_states=len(states)
    Q_sarsa=np.zeros((n_states,n_actions))
    current_state=states.index(initial_state)
    goal=states.index(goal_state)
    chart_norm=np.zeros(int(n_interactions/plot_x_interactions))
    counter_chart=0
            
    for i in range(100000):
        action=e_greedy.select_action(Q_sarsa,current_state,n_actions)
        
        next_state=np.random.choice(n_states,p=Prob[action][current_state,:])
        
        next_action=e_greedy.select_action(Q_sarsa,next_state,n_actions)
        
        Q_sarsa[current_state, action]=Q_sarsa[current_state, action] + 0.3*(c[current_state, action]+gamma*Q_sarsa[next_state,next_action ]- Q_sarsa[current_state, action])
        
        if current_state==goal:
            current_state=random.randint(0, n_states-1)
        else:
            current_state=next_state
            action=next_action
    
        chart_norm, counter_chart=plot_charts.savePlots(i,plot_x_interactions, chart_norm, counter_chart, Q_optimal, Q_sarsa)

    return Q_sarsa, chart_norm, counter_chart

    