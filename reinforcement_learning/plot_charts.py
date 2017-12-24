#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 18:24:37 2017

@author: RitaRamos
"""

import numpy as np

def savePlots(interaction,plot_x_interactions, chart_norm, counter_chart, Q_optimal, Q):
    if(interaction%plot_x_interactions==0):
        chart_norm[counter_chart]=np.linalg.norm(Q_optimal-Q)
        counter_chart+=1
    return chart_norm, counter_chart

