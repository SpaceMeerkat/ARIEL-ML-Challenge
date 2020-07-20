# -*- coding: utf-8 -*-
"""
Created on Mon May 20 14:27:48 2019

@author: c1307135
"""

import pickle

_scalers_path = '/home/anubis/c1307135/ARIEL/scalers.pkl'

with open(_scalers_path,'rb') as handle:
    _scalers = pickle.load(handle)
    
def radius_scaler(radius):
    radius = radius**2
    radius -= _scalers[0]
    radius /= _scalers[1]
    return radius