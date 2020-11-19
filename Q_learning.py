# -*- coding: utf-8 -*-
"""
@author: Po-Kan (William) Shih
@advisor: Dr. Bahman Moraffah
Variational inference function object
"""

import numpy as np
import itertools as itt
import Variables as var

# build Q-table
class Q_learner(object):
    gamma = var.gamma  # discount factor
    T = var.T
    
    def __init__(self, L, W, A, O):
        self.L = L  # number of LTE agents
        self.W = W  # number of WiFi agents
        self.N = self.L + self.W
        self.A = A  # size of action set
        self.O = O  # size of observation set
        
        self.q_table = np.zeros((self.N, self.O, self.A))
        
    
    def learn(self, data, epsilon):
        
        self.ep = len(data) # get number of episodes
        
        # extract rewards from each episode
        rewards = np.array(list(map(lambda t: t[2], data)), dtype = np.float)
        # extract action & obv histories of each agent
        action = []
        obv = []
        
        for n in range(self.N):
            # get effective actions
            action_n = np.array(list(map(lambda x: x[0][n,:], data)))
            action.append(action_n)
            
            # extract action & observation
            obv_n = np.array(list(map(lambda x: x[1][n,:], data)))
            obv.append(obv_n)
            
        n_k_pair = itt.product(range(self.N), range(self.ep))
        for (n, k) in n_k_pair:
            act_n_k = action[n][k]
            obv_n_k = obv[n][k]
            
            for t in range(self.T):
                if act_n_k[t] >= 0:
                    
                    q_value = self.q_table[n, obv_n_k[t], act_n_k[t]]
                    rwd = rewards[k, t]
            
        
            
