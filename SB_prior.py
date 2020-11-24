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


# =============================================================================
# class behavior_policy(object):
#     def __init__(self, A, O, Z, epsilon):
#         self.A = A
#         self.O = O
#         self.Z = Z
#         self.epsilon = epsilon
#         
#         self.prob_table()
#         
#     
#     def prob_table(self):
#         # initial node distribution
#         self.eta = np.zeros(self.Z)
#         self.eta[0] = 1
#         
#         # initialize p(z_t|a_0, ..., a_{t-1}, o_0, ..., o_t)
#         self.pz = np.array(self.eta)
#         
#         # uniform node transition prob
#         self.node_prob = np.ones((self.A, self.O, self.Z, self.Z))
#         self.node_prob /= np.sum(self.node_prob, axis = 3)[...,np.newaxis]
#         
#         # uniform exploration action probabilities
#         self.action_prob = np.random.default_rng().dirichlet(np.ones(self.A), self.Z)
#     
#     
#     def select_action(self, node: int):
#         '''
#         Parameters
#         ----------
#         node : int
#             current node.
#         Returns
#         -------
#         output the index of action to take given node.
#         '''
#         # extract probability vector for given node
#         prob_set = self.action_prob[node, :]
#         # pick an action
#         act = np.random.default_rng().multinomial(1, prob_set, size = 1)[0]
#         
#         return np.where(act == 1)[0][0]
#         
#     
#     def next_node(self, cur_node: int, act: int, obs: int):
#         '''
#         Parameters
#         ----------
#         act : int
#             index of action taken.
#         obs : int
#             index of observation received after act is taken.
#         cur_node : int
#             index of current node.
#         Returns
#         -------
#         output node distribution for next step given action, obseration, and 
#         current node.
#         '''
#         # extract probability vector for given node, action, observation
#         prob_set = self.node_prob[act, obs, cur_node, :]
#         # pick a noode
#         node = np.random.default_rng().multinomial(1, prob_set, size = 1)[0]
#         
#         return np.where(node == 1)[0][0]
# =============================================================================
class behavior_policy(object):
    def __init__(self, A, O, Z, epsilon, phi = None, sigma = None, lambda_ = None):
        self.A = A
        self.O = O
        self.Z = Z
        self.epsilon = epsilon
        # pi array
        self.phi = phi
        self.sigma = sigma
        self.lambda_ = lambda_
        
        self.prob_table()
        
    
    def prob_table(self):
        # initial node distribution
        self.eta = np.zeros(self.Z)
        self.eta[0] = 1
        
        # uniform exploration action probabilities
        explore_act = np.ones((self.Z, self.A))
        explore_act /= np.sum(explore_act, axis = 1)[:, np.newaxis]
        
        if self.phi is None:
            # initial exploitation action probabilities
            exploit_act = np.array(explore_act)
            # initial node transition prob
            self.node_prob = np.ones((self.A, self.O, self.Z, self.Z))
            self.node_prob /= np.sum(self.node_prob, axis = 3)[...,np.newaxis]
            
        else:
            assert self.phi.shape == explore_act.shape
            exploit_act = self.phi / np.sum(self.phi, axis = -1)[..., np.newaxis]
            
            v = self.sigma / (self.sigma + self.lambda_)
            vv = (1 - v[..., :-1]).cumprod(axis = -1)
            self.node_prob = np.empty_like(self.sigma)
            self.node_prob[..., 0] = v[..., 0]
            self.node_prob[..., 1:-1] = v[..., 1:-1] * vv[..., :-1]
            self.node_prob[..., -1] = vv[..., -1]
            self.node_prob /= np.sum(self.node_prob, axis = -1)[..., np.newaxis]
            
        self.action_prob = self.epsilon * explore_act + (1 - self.epsilon) * exploit_act
            
    
    def refresh_prob(self):
        # initialize p(z_t|a_0, ..., a_{t-1}, o_0, ..., o_t)
        self.pz = np.array(self.eta)
        self.t = 0  # indicator for initial state
        
    
    def select_action(self, act_pre: int = -1, obv_pre: int = -1):
        '''
        Parameters
        ----------
        act_pre : int, optional
            previous action index. The default is -1.
        obv_pre : int, optional
            previous observation index. The default is -1.
        Returns
        -------
        TYPE
            DESCRIPTION.
        '''
        if self.t > 0:
            self.update_action(act_pre, obv_pre)
        
        self.t += 1
        
        # update action probability marginalizing nodes
        # p(z_t) * p(a|z)
        prob_set = self.pz[..., np.newaxis] * self.action_prob
        # marginalize z
        prob_set = np.sum(prob_set, axis = 0)
        # normalize to valid probability
        prob_set = prob_set / np.sum(prob_set)
        
        # pick an action
        act = np.random.default_rng().multinomial(1, prob_set, size = 1)[0]
        
        return np.where(act == 1)[0][0]
    
    
    def update_action(self, act, obv):
        assert act >= 0 and obv >= 0
        # p(z_t|z_{t-1}, a_{t-1}, o_t)
        p_z_zao = self.node_prob[act, obv]
        
        # p(z_{t-1}) * p(a_{t-1}|z)
        p_az = self.pz * self.action_prob[:, act]
        # p(z_t, z_{t-1}, a_{t-1}|o_t)
        p_azz_o = p_az[..., np.newaxis] * p_z_zao
        # marginalize z_{t-1}, get p(z_t, a_{t-1}|o_t)
        p_az_o = np.sum(p_azz_o, axis = 0)
        # normalize, get p(z_t|a_{t-1}, o_t)
        self.pz = p_az_o / np.sum(p_az_o)

    

class behavior_policy2(object):
    def __init__(self, A, O, Z, epsilon, phi = None, sigma = None, lambda_ = None):
        self.A = A
        self.O = O
        self.Z = Z
        self.epsilon = epsilon
        # pi array
        self.phi = phi
        self.sigma = sigma
        self.lambda_ = lambda_
        
        self.prob_table()
        
    
    def prob_table(self):
        # initial node distribution
        self.eta = np.zeros(self.Z)
        self.eta[0] = 1
        
        # uniform exploration action probabilities
        self.explore_act = np.ones((self.Z, self.A))
        self.explore_act /= np.sum(self.explore_act, axis = 1)[:, np.newaxis]
        
        if self.phi is None:
            # initial exploitation action probabilities
            self.exploit_act = np.array(self.explore_act)
            # initial node transition prob
            self.node_prob = np.ones((self.A, self.O, self.Z, self.Z))
            self.node_prob /= np.sum(self.node_prob, axis = 3)[...,np.newaxis]
            
        else:
            assert self.phi.shape == self.explore_act.shape
            self.exploit_act = self.phi / np.sum(self.phi, axis = -1)[..., np.newaxis]
            
            v = self.sigma / (self.sigma + self.lambda_)
            vv = (1 - v[..., :-1]).cumprod(axis = -1)
            self.node_prob = np.empty_like(self.sigma)
            self.node_prob[..., 0] = v[..., 0]
            self.node_prob[..., 1:-1] = v[..., 1:-1] * vv[..., :-1]
            self.node_prob[..., -1] = vv[..., -1]
            self.node_prob /= np.sum(self.node_prob, axis = -1)[..., np.newaxis]
            
        self.action_prob = self.epsilon * self.explore_act + (1 - self.epsilon) * self.exploit_act
            
    
    def refresh_prob(self):
        # initialize p(z_t|a_0, ..., a_{t-1}, o_0, ..., o_t)
        self.pz = np.array(self.eta)
        self.t = 0  # indicator for initial state
        self.u = 0  # exploration factor
        
    
    def select_action(self, act_pre: int = -1, obv_pre: int = -1):
        '''
        Parameters
        ----------
        act_pre : int, optional
            previous action index. The default is -1.
        obv_pre : int, optional
            previous observation index. The default is -1.
        Returns
        -------
        TYPE
            DESCRIPTION.
        '''
        if self.t > 0:
            self.update_action(act_pre, obv_pre, self.u)
        
        self.t += 1
        
        # determine exploration or exploitation
        self.u = np.random.default_rng().uniform()
        
        if self.u > self.epsilon:
            act_prob = self.exploit_act
        else:
            act_prob = self.explore_act
        
        # update action probability marginalizing nodes
        # p(z_t) * p(a|z)
        prob_set = self.pz[..., np.newaxis] * act_prob
        # marginalize z
        prob_set = np.sum(prob_set, axis = 0)
        # normalize to valid probability
        prob_set = prob_set / np.sum(prob_set)
        
        # pick an action
        act = np.random.default_rng().multinomial(1, prob_set, size = 1)[0]
        
        return np.where(act == 1)[0][0]
    
    
    def update_action(self, act, obv, u):
        assert act >= 0 and obv >= 0
        # p(z_t|z_{t-1}, a_{t-1}, o_t)
        p_z_zao = self.node_prob[act, obv]
        
        if u > self.epsilon:
            act_prob = self.exploit_act[:, act]
        else:
            act_prob = self.explore_act[:, act]
        
        # p(z_{t-1}) * p(a_{t-1}|z)
        p_az = self.pz * act_prob
        # p(z_t, z_{t-1}, a_{t-1}|o_t)
        p_azz_o = p_az[..., np.newaxis] * p_z_zao
        # marginalize z_{t-1}, get p(z_t, a_{t-1}|o_t)
        p_az_o = np.sum(p_azz_o, axis = 0)
        # normalize, get p(z_t|a_{t-1}, o_t)
        self.pz = p_az_o / np.sum(p_az_o)
        
        

class eval_policy(object):
    
    def __init__(self, theta, phi, sigma, lambda_):
        # remove nodes
        removed_nodes = np.sum(phi - theta, axis = -1)
        removed_nodes = tuple(np.where(removed_nodes < 1e-6)[0])
        
        self.action_prob = np.delete(phi, removed_nodes, axis = 0)
        self.action_prob = self.action_prob / np.sum(self.action_prob, axis = -1)[..., np.newaxis]
        
        t1 = np.delete(sigma, removed_nodes, axis = 2)
        t1 = np.delete(sigma, removed_nodes, axis = 3)
        t2 = np.delete(lambda_, removed_nodes, axis = 2)
        t2 = np.delete(lambda_, removed_nodes, axis = 3)
        
        v = t1 / (t1 + t2)
        self.node_prob = np.empty_like(v)
        self.node_prob[..., 0] = v[..., 0]
        self.node_prob[..., 1:] = v[..., 1:] * (1 - v[..., :-1]).cumprod(axis = -1)
        
    
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