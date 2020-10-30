# -*- coding: utf-8 -*-
"""
@author: Po-Kan (William) Shih
@advisor: Dr. Bahman Moraffah
Variational inference function object
"""
import numpy as np
import itertools as itt
import Variables as var
from scipy.special import digamma, loggamma

class CAVI(object):
    gamma = var.gamma  # discount factor
    T = var.T
    
    def __init__(self, L, W, Z, A, O):
        self.L = L  # number of LTE agents
        self.W = W  # number of WiFi agents
        self.N = self.L + self.W
        self.A = A  # size of action set
        self.O = O  # size of observation set
        self.Z = Z  # truncation level for q
        #initialize first prior models
        self.prior_param()
        
    
    def prior_param(self):
        '''
        Returns
        -------
        define prior distributions p.
        '''
        # parameters of p(pi|z, theta) for (n, i) LTE & WiFi agents
        self.theta = np.ones((self.N, self.Z, self.A))
        
        # parameters of p(alpha|c, d) for (n, a, o, i) LTE & WiFi agents
        self.c = np.ones((self.N, self.A, self.O, self.Z))
        self.d = np.ones((self.N, self.A, self.O, self.Z))
        
        # define initial node distribution (deterministic) for all agents
        # p(z = 1) = 1 & p(z = 2, ..., |Z|) = 0 for all agents
        self.eta = np.zeros((self.N, self.Z))
        self.eta[:, 0] = 1
    
    
    def init_q_param(self):
        '''
        Returns
        -------
        initialize variational distributions q.
        '''
        # parameter of q(V|sigma, lambda) for (n, a, o, i, j) agent
        self.sigma = np.ones((self.N, self.A, self.O, self.Z, self.Z))
        self.lambda_ = np.ones((self.N, self.A, self.O, self.Z, self.Z))
        
        # parameter of q(pi|z, phi) for (n, i) agent
        self.phi = np.ones((self.N, self.Z, self.A))
        
        # parameters of q(alpha| a, b) for each (n, a, o, i) agent
        self.a = np.ones((self.N, self.A, self.O, self.Z))
        self.b = 0.1 * np.ones((self.N, self.A, self.O, self.Z))
        
        # initialize marginal q(z) for each (n, k) indices
        # each element = arrays of q(z) for each (n, k) agent
        # each array has dimension T_k * |Z|
        self.qz = [[] for i in range(self.N)]
        
        
    def fit(self, data, policy_list, max_iter = 150, tol = 1e-8):
        '''
        Parameters
        ----------
        max_iter : int, optional
            max iteration times for CAVI. The default is 150.
        tol : float, optional
            tolerance value for CAVI iteration. The default is 1e-8.
        Returns
        -------
        Perform CAVI.
        '''
        self.policy_list = policy_list  # behavior policy
        self.data = data  # trajectories
        self.ep = len(self.data) # get number of episodes
        
        # extract action & obv histories of each agent
        self.action = []
        self._action = []
        self.obv = []
        self._obv = []
        
        for n in range(self.N):
            # get effective actions
            action_n = np.array(list(map(lambda x: x[0][n,:], self.data)))
            self.action.append(action_n)
            self._action.append(list(map(lambda x: x[x >= 0], action_n)))
            
            # extract action & observation
            obv_n = np.array(list(map(lambda x: x[1][n,:], self.data)))
            self.obv.append(obv_n)
            self._obv.append(list(map(lambda x: x[x >= 0], obv_n)))
        
        self.init_q_param()
        # compute reweighted reward
        self.reweight_reward()
        # calc initial ELBO(q)
        self.elbo_values = [-np.inf]
        self.update_z()
        self.update_pi()
# =============================================================================
#         # CAVI iteration
#         for it in range(1, max_iter + 1):
#             # CAVI update
#             self.update_z()     # update each q(z) distribution
#             self.update_v()     # update each q(v) distribution
#             self.update_pi()    # update each q(pi) distribution
#             self.update_alpha() # update each q(alpha) distribution
#             # calc ELBO(q) at the end of all updates
#             self.elbo_values.append(self.calc_ELBO())
#             
#             # if converged, stop iteration
#             if np.abs(self.elbo_values[-2] - self.elbo_values[-1]) <= tol:
#                 # print('CAVI converged with ELBO(q) %.3f at iteration %d'%(self.elbo_values[-1], it))
#                 break
#         
#         # iteration terminates but still cannot converge
#         if it == max_iter:
#             print('CAVI ended with ELBO(q) %.f'%(self.elbo_values[-1]))
# =============================================================================
        
    
    def calc_ELBO(self):
        # pre-compute some values since they are being used later
        d1 = digamma(self.sigma)
        d2 = digamma(self.lambda_)
        d12 = digamma(self.sigma + self.lambda_)
        d2d12 = d2 - d12
        
        dalnb = digamma(self.a) - np.log(self.b)
        dalnb = digamma(self.a) - np.log(self.b)
        
        dphi = digamma(self.phi) - digamma(np.sum(self.phi, axis = 2))[..., np.newaxis]
        
        lowerbound = 0
        
        # (1) E[lnp(alpha| c, d)]
        palpha = (self.c - 1) * dalnb
        palpha -= (self.d * self.a / self.b)
        lowerbound += palpha.sum()
        
        # (2) E[lnp(V| alpha)]
        pv = d2d12 * ((self.a / self.b) - 1)[..., np.newaxis]
        pv += dalnb[..., np.newaxis]
        lowerbound += pv.sum()
        
        # (3) E[lnp(pi| theta)]
        pphi = (self.theta - 1) * dphi
        lowerbound += pphi.sum()
        
        # (4) E[lnq(alpha| a, b)]
        qalpha = self.a * (digamma(self.a) - 1)
        qalpha -= loggamma(self.a)
        qalpha -= dalnb
        lowerbound -= qalpha.sum()
        
        # (5) E[lnq(V| sigma, lambda)]
        t1 = (self.sigma - 1) * (d1 - d12)
        t2 = (self.lambda_ - 1) * d2d12
        qv = t1 + t2 + loggamma(self.sigma + self.lambda_)
        qv -= loggamma(self.sigma)
        qv -= loggamma(self.lambda_)
        lowerbound -= qv.sum()
        
        # (6) E[lnq(pi| phi)]
        qphi = (self.phi - 1) * dphi
        qphi += loggamma(np.sum(self.phi, axis = 2))[..., np.newaxis]
        qphi -= loggamma(self.phi)
        lowerbound -= qphi.sum()
        
        # (7) ln(r) term in E[ln(q(z))]
        lnr = np.sum(np.log(self.reweight_r)) / self.ep
        lowerbound += lnr
        
        # (8) E[ln(pi)] term in E[ln(q(z))]
        
        
        # (9) E[ln(omega)] term in E[ln(q(z))]
        
        
        assert np.isscalar(lowerbound)
        return lowerbound
    
    
    def update_z(self):
        # compute expected values of action prob pi
        self.pi = self.phi / np.sum(self.phi, axis = 2)[..., np.newaxis]
        
        # compute expected values of V
        # build node trans prob omega from expected V
        V = self.sigma / (self.sigma + self.lambda_)
        # prob weight of last node = 1
        V[...,-1] = 1
        self.omega = np.empty_like(V)
        # prob to node 1
        self.omega[..., 0] = V[..., 0]
        # prob to node 2, ..., |Z|
        self.omega[..., 1:] = V[..., 1:] * (1 - V[..., :-1]).cumprod(axis = -1)
        
        n_k_pair = itt.product(range(self.N), range(self.ep))
        
        for (n, k) in n_k_pair:
            # extract action history for agent n at episode k
            temp = self.action[n][k, :]
            index = np.where(temp >= 0)[0]
            assert len(index) > 0
            act_n_k = temp[index]
            
            temp = self.obv[n][k, :]
            obv_n_k = temp[index]
            
            # create q(z) table for agent n at episode k
            assert len(index) <= self.T
            qz_n_k = np.zeros((len(index), self.Z))
            qz_n_k[0, :] = self.eta[n, :]
            assert np.sum(qz_n_k[0, :]) > 0
            
            for i, t in enumerate(index):
                # if this is the 1st action
                if i == 0:
                    t1 = qz_n_k[i, :]
                else:
                    t1 = qz_n_k[i - 1, :]
                    t1 = t1[...,np.newaxis] * self.omega[n, act_n_k[i-1], obv_n_k[i-1], ...]
                    t1 = np.sum(t1, axis = 0)
                
                t1 *= self.pi[n, :, act_n_k[i]]
                t1 *= self.reweight_r[k, t]
                assert np.sum(t1) > 0
                qz_n_k[i, :] = t1 / np.sum(t1)
                
            
            self.qz[n].append(qz_n_k)
    
    
    def update_v(self):
        self.sigma = np.ones((self.N, self.A, self.O, self.Z, self.Z))
        

    
    def update_pi(self):
        self.phi[...] = self.theta[...]
        update = np.zeros(self.phi.shape)
        
        n_k_pair = itt.product(range(self.N), range(self.ep))
        for (n, k) in n_k_pair:
            
            aa = self._action[n][k]
            q = self.qz[n][k]
            for i, a in enumerate(aa):
                update[n, :, a] += (q[i] * (q.shape[0] - i + 1))
                
        update /= self.ep
        self.phi += update
            
    
    def update_alpha(self):
        assert self.a.shape == self.c.shape
        assert self.b.shape == self.d.shape
        
        d2d12 = digamma(self.lambda_) - digamma(self.sigma + self.lambda_)
        
        self.a = self.c + self.Z
        self.b = self.d - np.sum(d2d12, axis = 4)
                
    
    def reweight_reward(self):
        
        # extract rewards from each episode
        rewards = np.array(list(map(lambda t: t[2], self.data)))
        self.reweight_r = np.zeros(rewards.shape)
        
        # rescale rewards
        r_max = np.max(rewards)
        r_min = np.min(rewards)
        rewards = (rewards - r_min + 1) / (r_max - r_min + 1)
        
        # impose discount factor
        reward_ = rewards.transpose()
        reward_ = np.array(list(map(lambda x, t: x*(self.gamma**t), reward_, range(self.T))))
        rewards = reward_.transpose()
        
        # in each episode, compute reweighted rewards
        for k in range(self.ep):
            # initialize p(z) array to track p(z_{t-1}, a_{t-1}, ..., a_0)
            pz = np.array(list(map(lambda x: x.eta, self.policy_list)))
            # action tracker array, used to track the order of action for each agent
            action_num = np.zeros(self.N, dtype = np.int)
            
            # reward array for episode k
            rwd_k = rewards[k, :]
            # get action & observation arrays
            act_array = self.data[k][0]
            obv_array = self.data[k][1]
            for t in range(self.T):
                # extract the agents who contribute reward at time k
                agent_idx = np.where(act_array[:, t] >= 0)[0]
                # extract their initial node prob
                p_eta = pz[agent_idx]
                # extract their action prob given node
                act_idx = act_array[agent_idx, t]
                # extract corresponding taken action prob given all nodes
                p_act = np.array(list(map(lambda n,a: self.policy_list[n].action_prob[:,a],agent_idx,act_idx)))
                # compute p(a|z)
                p_eta *= p_act
                
                # if t > 0, need to times p(z|z, a, o) extra
                for ii, nn in enumerate(agent_idx):
                    if action_num[nn] > 0:
                        assert act_array[nn, t] >= 0
                        assert obv_array[nn, t] >= 0
                        p_node = self.policy_list[nn].node_prob[act_array[nn, t], obv_array[nn, t],...]
                        temp = p_node * p_eta[ii, np.newaxis]
                        p_eta[ii] = np.sum(temp, axis = -2)
                    # replace p(z)
                    pz[nn, :] = p_eta[ii, :]
                        
                    action_num[nn] += 1
                                    
                assert np.prod(np.sum(p_eta, axis = -1)) > 0
                self.reweight_r[k, t] = rwd_k[t] / np.prod(np.sum(p_eta, axis = -1))
        