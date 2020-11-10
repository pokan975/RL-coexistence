# -*- coding: utf-8 -*-
"""
@author: Po-Kan (William) Shih
@advisor: Dr. Bahman Moraffah
Finite state controller policy
"""
import numpy as np


# =============================================================================
# def SBPR(Z, sigma, rho, size_A, size_O, c = 0.1, d = 10**-6):
#     '''
#     Parameters
#     ----------
#     Z : int
#         truncation level of SB process (number of nodes).
#     sigma : float array
#         SB parameter.
#     rho : float vector (length = size of action set)
#         parameter of Dirichlet distribution.
#     size_A : int
#         size of action set.
#     size_O : int
#         size of observation set.
#     c : float, optional
#         parameter of Gamma. The default is 0.1.
#     d : float, optional
#         parameter of Gamma. The default is 10**-6.
#     Returns
#     -------
#     None.
#     '''
#     # if rho vector != size of action set, alarm
#     assert rho.shape[0] == size_A
#     # create action distribution at each node
#     action_prob = np.random.default_rng().dirichlet(rho, size = Z)
#     
#     # create node transition probabilities for each (action, observation) set
#     assert sigma.shape == (Z, Z)
#     W = []
#     for o in range(size_O):
#         W_temp = []
#         for a in range(size_A):
#             # generate eta for Beta distribution
#             eta = np.random.default_rng().gamma(c, d, size = (Z ,Z))
#             V = np.random.default_rng().beta(sigma, eta)  # Beta r.v. array
#             node_prob = np.empty_like(V)
#             node_prob[:, 0] = V[:, 0]
#             node_prob[:, 1:] = V[:, 1:] * (1 - V[:, :-1]).cumprod(axis = 1)
#             W_temp.append(node_prob)
#             
#         W.append(W_temp)
#     
#     # create FSC policy
#     policy = FSC_policy(action_prob, W)
#     
#     return policy
# =============================================================================


class uniform_policy(object):
    '''
    For collecting node info for behavior policy
    '''
    def __init__(self, act_set):
        self.act_setsize = len(act_set)
        self.action_prob = np.array([1/self.act_setsize] * self.act_setsize)
        
    def select_action(self, node: int):
        # the input argument is not functional, just to make class structure similar
        # to FSC policy
        act = np.random.default_rng().multinomial(1, self.action_prob, size = 1)[0]
        
        return np.where(act == 1)[0][0]
    
    def next_node(self, cur_node: int, act: int, obs: int):
        # the input argument is not functional, just to make class structure similar
        # to FSC policy
        return 0


class behavior_policy(object):
    def __init__(self, A, O, Z, epsilon, posterior = None):
        self.A = A
        self.O = O
        self.Z = Z
        self.epsilon = epsilon
        self.posterior = posterior
        self.prob_table()
        self.t = 0  # indicator for initial state
        
    def prob_table(self):
        # initial node distribution
        self.eta = np.zeros(self.Z)
        self.eta[0] = 1
        
        self.pz = np.array(self.eta)
        
        # uniform node transition prob
        self.node_prob = np.ones((self.A, self.O, self.Z, self.Z))
        self.node_prob /= np.sum(self.node_prob, axis = 3)[...,np.newaxis]
        
        # uniform exploration action probabilities
        explore_act = np.ones((self.Z, self.A))
        explore_act /= np.sum(explore_act, axis = 1)[:, np.newaxis]
        
        if self.epsilon <= 0:
            # initial exploitation action probabilities
            exploit_act = np.array(explore_act)
            
        else:
            assert self.posterior.shape == explore_act.shape
            exploit_act = self.posterior / np.sum(self.posterior, axis = -1)[..., np.newaxis]
            
        # build action probability given node
        self.action_prob = self.epsilon * explore_act + (1 - self.epsilon) * exploit_act
    
    
    def select_action(self, node: int):
        '''
        Parameters
        ----------
        node : int
            current node.
        Returns
        -------
        output the index of action to take given node.
        '''
        # extract probability vector for given node
        prob_set = self.action_prob[node, :]
        # pick an action
        act = np.random.default_rng().multinomial(1, prob_set, size = 1)[0]
        
        return np.where(act == 1)[0][0]
        
    
    def next_node(self, cur_node: int, act: int, obs: int):
        '''
        Parameters
        ----------
        act : int
            index of action taken.
        obs : int
            index of observation received after act is taken.
        cur_node : int
            index of current node.
        Returns
        -------
        output node distribution for next step given action, obseration, and 
        current node.
        '''
        # extract probability vector for given node, action, observation
        prob_set = self.node_prob[act, obs, cur_node, :]
        # pick a noode
        node = np.random.default_rng().multinomial(1, prob_set, size = 1)[0]
        
        return np.where(node == 1)[0][0]
    
    
    def update_action(self):
        if self.t == 0:
             
        