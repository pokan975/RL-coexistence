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
    
    def __init__(self, L, W, A):
        self.L = L  # number of LTE agents
        self.W = W  # number of WiFi agents
        self.N = self.L + self.W
        self.A = A  # size of action set
        
    
    def prior_param(self, O, Z):
        '''
        Parameters
        ----------
        O : int
            size of observation set.
        Z : int
            truncation level for q.
        Returns
        -------
        Initialize the prior distributions.
        '''
        self.O = O  # size of observation set
        self.Z = Z  # truncation level for q
        
        # parameters of p(pi|z, theta) for (n, i) LTE & WiFi agents
        self.theta = np.ones((self.N, self.Z, self.A))
        
        # parameters of p(alpha|c, d) for (n, a, o, i) LTE & WiFi agents
        self.c = 0.1 * np.ones((self.N, self.A, self.O, self.Z))
        self.d = 1e-6 * np.ones((self.N, self.A, self.O, self.Z))
        
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
        self.phi = self.theta[...]
        
        # parameters of q(alpha| a, b) for each (n, a, o, i) agent
        self.a = self.c[...]
        self.b = self.d[...]
        
        # initialize marginal q(z) for each (n, k) indices
        # each element = arrays of q(z) for each (n, k) agent
        # each array has dimension T_k * |Z|
        self.qz = [[] for i in range(self.N)]
        
        
    def fit(self, data, policy_list, max_iter = 10, tol = 1e-4):
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
        
        # extract action & node trans probabilities for each agent from polict list
        self.behave_act_prob = map(lambda n: policy_list[n].action_prob, range(self.N))
        self.behave_act_prob = np.array(list(self.behave_act_prob))
        self.behave_node_prob = map(lambda n: policy_list[n].node_prob, range(self.N))
        self.behave_node_prob = np.array(list(self.behave_node_prob))
        
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
        # calc initial ELBO(q)
        self.elbo_values = [-np.inf]
        # CAVI iteration
        for it in range(1, max_iter + 1):
            # CAVI update
            self.calc_phi_omega() # calc ML pi & omega
            self.update_z()     # update each q(z) distribution
            self.reweight_reward()  # compute reweighted reward
            self.update_v()     # update each q(v) distribution
            self.update_pi()    # update each q(pi) distribution
            self.update_alpha() # update each q(alpha) distribution
            # calc ELBO(q) at the end of all updates
            self.elbo_values.append(self.calc_ELBO())
            
            # if converged, stop iteration
            if np.abs(self.elbo_values[-2] - self.elbo_values[-1]) <= tol:
                # print('CAVI converged with ELBO(q) %.3f at iteration %d'%(self.elbo_values[-1], it))
                break
        
        # iteration terminates but still cannot converge
        # if it == max_iter:
        #     print('CAVI ended with ELBO(q) %.f'%(self.elbo_values[-1]))
        
    
    def calc_ELBO(self):
        # pre-compute some values since they are being used later
        d1 = digamma(self.sigma)
        d2 = digamma(self.lambda_)
        d12 = digamma(self.sigma + self.lambda_)
        d2d12 = d2 - d12
        
        dalnb = digamma(self.a) - np.log(self.b)
        dalnb = digamma(self.a) - np.log(self.b)
        
        dphi = digamma(self.phi) - digamma(np.sum(self.phi, axis = -1))[..., np.newaxis]
        
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
        qphi += loggamma(np.sum(self.phi, axis = -1))[..., np.newaxis]
        qphi -= loggamma(self.phi)
        lowerbound -= qphi.sum()
        
        # (7) ln(r) term in E[ln(q(z))]
        lnr = np.log(self.reweight_r)
        lowerbound += lnr.sum()
                
        # (8) E[ln(pi)] & E[ln(omega)] terms in E[ln(q(z))]
        # build omega
        _dphi = np.sum(dphi, axis = -2)
        _omega = np.zeros(self.sigma.shape)
        _omega[..., : -1] = (d1 - d12)[..., : -1]
        t1 = (d2 - d12)[..., 0: -1]
        _omega[..., 1:] += t1.cumsum(axis = -1)
        
        likelihood = 0
        n_k_pair = itt.product(range(self.N), range(self.ep))
        for (n, k) in n_k_pair:
            # get act #
            aa = tuple(self._action[n][k])
            # get obv #
            oo = tuple(self._obv[n][k])
            
            tt = np.ones(len(aa)).cumsum()[::-1]
            _act = _dphi[n, aa] * tt
            likelihood += _act.sum()
            
            om = _omega[n, aa, oo, ...]
            om = np.sum(om, axis = (1, 2)) * tt
            likelihood += om.sum()
            
            # for i, a_o in enumerate(zip(aa, oo)):
            #     likelihood += _omega[n, a_o[0], a_o[1], ...].sum() * (len(aa) - i + 1)
        
        lowerbound += likelihood
        
        assert np.isscalar(lowerbound)
        return lowerbound
    
    
    def calc_phi_omega(self):
        # build action prob pi
        t1 = digamma(np.sum(self.phi, axis = -1))
        self.pi = np.exp(digamma(self.phi) - t1[..., np.newaxis])
        # normalize to valid prob distribution
        self.pi /= np.sum(self.pi, axis = -1)[..., np.newaxis]
        
        # build omega
        d1 = digamma(self.sigma)
        d2 = digamma(self.lambda_)
        d12 = digamma(self.sigma + self.lambda_)
        self.omega = np.zeros(self.sigma.shape)
        # build trans prob to node 1
        self.omega[..., : -1] = (d1 - d12)[..., : -1]
        # build trans prob to node 2~|Z|-1
        t1 = (d2 - d12)[..., 0: -1]
        self.omega[..., 1:] += t1.cumsum(axis = -1)
        self.omega = np.exp(self.omega)
        self.omega /= np.sum(self.omega, axis = -1)[..., np.newaxis]
    

    def update_z(self):
        
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
                # t1 *= self.reweight_r[k, t]
                assert np.sum(t1) > 0
                qz_n_k[i, :] = t1 / np.sum(t1)
                
            
            self.qz[n].append(qz_n_k)
    
    
    def update_v(self):
        
        self.sigma = np.zeros((self.N, self.A, self.O, self.Z, self.Z))
        self.lambda_ = np.zeros((self.N, self.A, self.O, self.Z, self.Z))
        
        n_k_pair = itt.product(range(self.N), range(self.ep))
        for (n, k) in n_k_pair:
            # get act #
            aa = tuple(self._action[n][k])
            # get obv #
            oo = tuple(self._obv[n][k])
            # get nu_t^k
            v = np.where(self.action[n][k] >= 0)[0]
            v = self.nu[k][v]
            
            tt = np.ones(len(aa)).cumsum()[::-1]
            
            # update parameter sigma
            # get q(z) array for agent n, episode k
            q = self.qz[n][k]
            # times action prob
            q *= self.pi[n, :, aa]
            # times node trans prob
            q = q[..., np.newaxis] * self.omega[n, aa, oo, ...]
            q *= tt[..., None, None]
            q *= v[..., None, None]
            q = np.array(q, ndmin = 3)
            self.sigma[n, aa, oo, ...] += q
            
            #update parameter lambda
            qq = np.cumsum(q[..., -1:0:-1], axis = -1)[..., ::-1]
            self.lambda_[n, aa, oo, :, :-1] += qq
            
            
        self.sigma += 1
        
        t1 = np.zeros(self.lambda_.shape) + (self.a / self.b)[..., None]
        # self.lambda_ /= self.ep
        self.lambda_ += t1

    
    def update_pi(self):
        self.phi = np.zeros(self.theta.shape)
        
        n_k_pair = itt.product(range(self.N), range(self.ep))
        for (n, k) in n_k_pair:
            # get act #
            aa = tuple(self._action[n][k])
            # get q(z)
            q = self.qz[n][k]
            # get nu_t^k
            v = np.where(self.action[n][k] >= 0)[0]
            v = self.nu[k][v]
            
            tt = np.ones(len(aa)).cumsum()[::-1]
            q *= tt[..., np.newaxis]
            q *= v[..., np.newaxis]
            self.phi[n, :, aa] += q
            # for i, a in enumerate(aa):
            #     self.phi[n, :, a] += (q[i] * (q.shape[0] - i + 1))
                
        # self.phi /= self.ep
        self.phi += self.theta
            
    
    def update_alpha(self):
        d2d12 = digamma(self.lambda_) - digamma(self.sigma + self.lambda_)
        
        self.a = self.c + self.Z
        self.b = self.d - np.sum(d2d12, axis = -1)
                

    def reweight_reward(self):
        # extract rewards from each episode
        rewards = np.array(list(map(lambda t: t[2], self.data)))
        self.reweight_r = np.zeros(rewards.shape)
        self.nu = np.zeros(rewards.shape)
        
        # rescale rewards
        r_max = np.max(rewards)
        r_min = np.min(rewards)
        rewards = (rewards - r_min + 1) / (r_max - r_min + 1)
        
        # impose discount factor
        ga = np.ones((self.ep, self.T))
        ga[:, 1:] *= 0.9
        rewards *= np.cumprod(ga, axis = 1)
        
        # in each episode, compute reweighted rewards
        for k in range(self.ep):
            # initialize p(z) array to track p(z_{t-1}, a_{t-1}, ..., a_0)
            p_ao_p = np.empty_like(self.eta)
            p_ao_p[...] = self.eta[...]
            p_ao_q = np.empty_like(self.eta)
            p_ao_q[...] = self.eta[...]
            # action tracker array, used to track the order of action for each agent
            action_num = np.zeros(self.N, dtype = np.int)
            
            # reward array for episode k
            rwd_k = rewards[k, :]
            # get action & observation arrays for episode k
            act_array = self.data[k][0]
            obv_array = self.data[k][1]
            for t in range(self.T):
                # extract the agents who contribute reward at time k
                agent_idx = tuple(np.where(act_array[:, t] >= 0)[0])
                # extract their action #
                act_idx = tuple(act_array[agent_idx, t])
                # extract their obv #
                obv_idx = tuple(obv_array[agent_idx, t])
                # extract initial node prob for behavior policy
                p_eta = np.array(p_ao_p[agent_idx, ...], ndmin = 2)
                # extract initial node prob for optimal policy
                q_eta = np.array(p_ao_q[agent_idx, ...], ndmin = 2)
                # extract corresponding taken action prob given all nodes
                p_act_p = self.behave_act_prob[agent_idx, :, act_idx]
                p_act_q = self.pi[agent_idx, :, act_idx]
                # compute p(a,z)=p(z)p(a|z)
                p_eta *= p_act_p
                q_eta *= p_act_q
                
                p_node = np.array(self.behave_node_prob[agent_idx, act_idx, obv_idx, ...], ndmin = 3)
                q_node = np.array(self.omega[agent_idx, act_idx, obv_idx, ...], ndmin = 3)
                
                # if t > 0, need to times p(z|z, a, o) extra
                for ii, nn in enumerate(agent_idx):
                    if action_num[nn] > 0:
                        assert act_array[nn, t] >= 0
                        assert obv_array[nn, t] >= 0
                        p_eta[ii] = (p_node[ii] * p_eta[ii, :, np.newaxis]).sum(axis = -2)
                        q_eta[ii] = (q_node[ii] * q_eta[ii, :, np.newaxis]).sum(axis = -2)
                    # replace p(z)
                    p_ao_p[nn, :] = p_eta[ii, :]
                    p_ao_q[nn, :] = q_eta[ii, :]
                        
                action_num[np.array(agent_idx)] += 1
                                    
                assert np.prod(np.sum(p_eta, axis = -1)) > 0
                self.reweight_r[k, t] = rwd_k[t] / np.prod(np.sum(p_eta, axis = -1))
                self.nu[k, t] = self.reweight_r[k, t] * np.prod(np.sum(q_eta, axis = -1))



    def calc_node_number(self):
        # compute the converged node number for each agent's FSC policy
        self.node_num = np.zeros(self.N)
        
        for n in range(self.N):
            a1 = np.sum(self.phi[n] - self.theta[n], axis = -1)
            self.node_num[n] = len(a1[a1 > 0])