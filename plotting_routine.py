# -*- coding: utf-8 -*-
"""
Created on Fri May  3 14:55:40 2019

@author: c1307135
"""

#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#

class epoch_plotter:
    
    def __init__(self,train_epoch,train_mu,train_std,
                 test_epoch,test_mu,test_std,save_path):
        
        self.train_epoch = train_epoch
        self.train_mu = train_mu
        self.train_std = train_std
        self.test_epoch = test_epoch
        self.test_mu = test_mu
        self.test_std = test_std
        self.save_path = save_path
        
    def filler(self,mu,std):
        _up = np.array(mu) + np.array(std)
        _down = np.array(mu) - np.array(std)
        return np.array(_up), np.array(_down)
        
    def plotter(self):
        _train_up, _train_down = self.filler(self.train_mu,self.train_std)
        _test_up, _test_down = self.filler(self.test_mu,self.test_std)
        plt.figure()
        plt.plot(self.train_epoch,self.train_mu,'b-')
        plt.fill_between(self.train_epoch,_train_up,_train_down,color='b',alpha=0.5)
        plt.plot(self.test_epoch,self.test_mu,'r-')
        plt.fill_between(self.test_epoch,_test_up,_test_down,color='r',alpha=0.5)
        plt.ylim(0,0.0002)
        plt.title('Accuracy: '+str(self.test_mu[-1]))
        plt.savefig(self.save_path)
        plt.close('all')
        return
        
        