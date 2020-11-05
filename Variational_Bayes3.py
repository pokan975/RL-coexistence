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
        self.phi = np.empty_like(self.theta)
        self.phi[...] = self.theta
        
        # parameters of q(alpha| a, b) for each (n, a, o, i) agent
        self.a = np.empty_like(self.c)
        self.a[...] = self.c
        self.b = np.empty_like(self.d)
        self.b[...] = self.d
        
        
    def fit(self, data, policy_list, max_iter = 50, tol = 1e-2):
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
        
        # extract initial node distribution
        self.eta = np.array(list(map(lambda n: policy_list[n].eta, range(self.N))))
        
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
        # compute reweighted reward
        self.reweight_reward()
        # calc initial ELBO(q)
        self.elbo_values = [self.calc_ELBO()]
        # CAVI iteration
        for it in range(1, max_iter + 1):
            # CAVI update
            self.calc_pi_omega() # calc ML pi & omega
            self.update_z()     # update each q(z) distribution
            # self.reweight_nu()  # compute \hat{nu}
            self.update_v()     # update each q(v) distribution
            self.update_pi()    # update each q(pi) distribution
            self.update_alpha() # update each q(alpha) distribution
            # calc ELBO(q) at the end of all updates
            self.elbo_values.append(self.calc_ELBO())
            
            # if converged, stop iteration
            if np.abs(self.elbo_values[-2] - self.elbo_values[-1]) <= tol:
                # print('CAVI converged with ELBO(q) %.3f at iteration %d'%(self.elbo_values[-1], it))
                break
        
    
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
        
        # # (7) ln(r) term in E[ln(q(z))]
        # lnr = np.log(self.reweight_r)
        # lowerbound += lnr.sum()
                
        # # (8) E[ln(pi)] & E[ln(omega)] terms in E[ln(q(z))]
        # # build omega
        # _dphi = np.sum(dphi, axis = -2)
        # _omega = np.zeros(self.sigma.shape)
        # _omega[..., : -1] = (d1 - d12)[..., : -1]
        # t1 = (d2 - d12)[..., 0: -1]
        # _omega[..., 1:] += t1.cumsum(axis = -1)
        
        # likelihood = 0
        # n_k_pair = itt.product(range(self.N), range(self.ep))
        # for (n, k) in n_k_pair:
        #     # get act #
        #     aa = tuple(self._action[n][k])
        #     # get obv #
        #     oo = tuple(self._obv[n][k])
            
        #     tt = np.ones(len(aa)).cumsum()[::-1]
        #     _act = _dphi[n, aa] * tt
        #     likelihood += _act.sum()
            
        #     om = _omega[n, aa, oo, ...]
        #     om = np.sum(om, axis = (1, 2)) * tt
        #     likelihood += om.sum()
        
        # lowerbound += likelihood
        
        assert np.isscalar(lowerbound)
        return lowerbound
    
    
    def calc_pi_omega(self):
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
        self.omega[...] = (d1 - d12)#self.omega[..., : -1] = (d1 - d12)[..., : -1]
        # build trans prob to node 2~|Z|-1
        t1 = (d2 - d12)[..., 0: -1]
        self.omega[..., 1:] += t1.cumsum(axis = -1)
        self.omega = np.exp(self.omega)
        self.omega /= np.sum(self.omega, axis = -1)[..., np.newaxis]
    

    def update_z(self):
        # initialize marginal q(z) for each (n, k) indices
        # each element = arrays of q(z) for each (n, k) agent
        # each array has dimension T_k * |Z|
        self.qz = [[] for i in range(self.N)]
        
        self.v_hat = np.ones(self.reweight_r.shape)
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
            qz_n_k = np.zeros((len(index), self.Z))
            qz_n_k[0, :] = self.eta[n, :]
            
            for i, t in enumerate(index):
                # if this is the 1st action
                if i == 0:
                    t1 = np.array(qz_n_k[i, :])
                else:
                    # t1 = np.array(qz_n_k[i - 1, :])
                    t1 = t1[...,np.newaxis] * self.omega[n, act_n_k[i-1], obv_n_k[i-1], ...]
                    t1 = np.sum(t1, axis = 0)
                
                t1 *= self.pi[n, :, act_n_k[i]]
                t2 = np.sum(t1)
                assert t2 > 0
                qz_n_k[i, :] = t1 / t2
                self.v_hat[k, t] *= t2
            
            self.qz[n].append(qz_n_k)
    
    
    def update_v(self):
        
        self.sigma = np.ones((self.N, self.A, self.O, self.Z, self.Z))
        self.lambda_ = np.zeros((self.N, self.A, self.O, self.Z, self.Z))
        
        n_k_pair = itt.product(range(self.N), range(self.ep))
        for (n, k) in n_k_pair:
            # get act #
            aa = tuple(self._action[n][k])
            # get obv #
            oo = tuple(self._obv[n][k])
            # get nu_t^k
            # v = tuple(np.where(self.action[n][k] >= 0)[0])
            # v = self.nu[k, v]
            
            tt = np.ones(len(aa)).cumsum()[::-1]
            
            # update parameter sigma
            # get q(z) array for agent n, episode k
            q = np.array(self.qz[n][k])
            # times action prob
            q *= self.pi[n, :, aa]
            # times node trans prob
            q = q[..., np.newaxis] * self.omega[n, aa, oo, ...]
            q *= tt[..., None, None]
            # q *= v[..., None, None]
            q = np.array(q, ndmin = 3)
            self.sigma[n, aa, oo, ...] += q
            
            #update parameter lambda
            qq = np.cumsum(q[..., -1:0:-1], axis = -1)[..., ::-1]
            self.lambda_[n, aa, oo, :, :-1] += qq
            
        # self.lambda_ /= self.ep
        self.lambda_ += (self.a / self.b)[..., None]

    
    def update_pi(self):
        self.phi = np.empty_like(self.theta)
        self.phi[...] = self.theta
        
        n_k_pair = itt.product(range(self.N), range(self.ep))
        for (n, k) in n_k_pair:
            # get act #
            aa = tuple(self._action[n][k])
            # get q(z)
            q = np.array(self.qz[n][k])
            # get nu_t^k
            # v = tuple(np.where(self.action[n][k] >= 0)[0])
            # v = self.nu[k, v]
            
            tt = np.ones(len(aa)).cumsum()[::-1]
            q *= tt[..., np.newaxis]
            # q *= v[..., np.newaxis]
            self.phi[n, :, aa] += q
            # for i, a in enumerate(aa):
            #     self.phi[n, :, a] += (q[i] * (q.shape[0] - i + 1))
                
        # self.phi /= self.ep
            
    
    def update_alpha(self):
        d2d12 = digamma(self.lambda_) - digamma(self.sigma + self.lambda_)
        
        self.a = self.c + self.Z
        self.b = self.d - np.sum(d2d12, axis = -1)
                

    def reweight_nu(self):
        # extract rewards from each episode
        self.nu = np.zeros(self.reweight_r.shape)
                
        # in each episode, compute reweighted rewards
        for k in range(self.ep):
            # initialize p(z) array to track p(z_{t-1}, a_{t-1}, ..., a_0)
            q_ao = np.empty_like(self.eta)
            q_ao[...] = self.eta
            # action tracker array, used to track the order of action for each agent
            action_num = np.zeros(self.N, dtype = np.int)
            
            # get action & observation arrays for episode k
            act_array = self.data[k][0]
            for t in range(self.T):
                # extract the agents who contribute reward at time k
                agent_idx = tuple(np.where(act_array[:, t] >= 0)[0])
                
                # extract initial node prob for optimal policy
                q_eta = np.array(q_ao[agent_idx, ...], ndmin = 2)
                
                # if t > 0, need to times p(z|z, a, o) extra
                for ii, nn in enumerate(agent_idx):
                    if action_num[nn] > 0:
                        act_pre = self._action[nn][k][action_num[nn] - 1]
                        obv_pre = self._obv[nn][k][action_num[nn] - 1]
                        
                        q_node = np.array(self.omega[nn, act_pre, obv_pre, ...])
                        q_eta[ii] = (q_node * q_eta[ii, :, np.newaxis]).sum(axis = 0)
                    
                    # extract their action #
                    act_idx = act_array[nn, t]
                    # extract corresponding taken action prob given all nodes
                    q_act = self.pi[nn, :, act_idx]
                    # compute p(a,z)=p(z)p(a|z)
                    q_eta[ii] *= q_act
                    
                    # replace p(z)
                    q_ao[nn, :] = q_eta[ii, :]
                        
                action_num[np.array(agent_idx)] += 1
                self.nu[k, t] = self.reweight_r[k, t] * np.prod(np.sum(q_eta, axis = -1)) #/ self.v_hat[k, t]



    def reweight_reward(self):
        # extract rewards from each episode
        rewards = np.array(list(map(lambda t: t[2], self.data)))
        self.reweight_r = np.zeros(rewards.shape)
        
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
            p_ao = np.empty_like(self.eta)
            p_ao[...] = self.eta
            # action tracker array, used to track the order of action for each agent
            action_num = np.zeros(self.N, dtype = np.int)
            
            # get action & observation arrays for episode k
            act_array = self.data[k][0]
            # obv_array = self.data[k][1]
            for t in range(self.T):
                # extract agents # who contribute to reward at time t
                agent_idx = tuple(np.where(act_array[:, t] >= 0)[0])
                
                # extract p(z_t-1, a_t-1, ..., a_0)
                p_eta = np.array(p_ao[agent_idx, ...], ndmin = 2)
                
                # if t > 0, need to multiply p(z|z, a, o) extra
                for ii, nn in enumerate(agent_idx):
                    if action_num[nn] > 0:
                        act_pre = self._action[nn][k][action_num[nn] - 1]
                        obv_pre = self._obv[nn][k][action_num[nn] - 1]
                        # extract p(z_t|z_t-1, a_t-1, o_t)
                        p_node = np.array(self.behave_node_prob[nn, act_pre, obv_pre, ...])
                        # p(z_t, a_t-1, ..., a_0)
                        p_eta[ii] = (p_node * p_eta[ii, :, np.newaxis]).sum(axis = 0)
                    
                    # extract action #
                    act_idx = act_array[nn, t]
                    # extract p(a_t|z_t)
                    p_act = self.behave_act_prob[nn, :, act_idx]
                    # compute p(z_t, a_t, ..., a_0)
                    p_eta[ii] *= p_act
                    
                    # replace p(z)
                    p_ao[nn, :] = p_eta[ii, :]
                        
                action_num[np.array(agent_idx)] += 1
                                    
                assert np.prod(np.sum(p_eta, axis = -1)) > 0
                self.reweight_r[k, t] = rewards[k, t] / np.prod(np.sum(p_eta, axis = -1))


    def calc_node_number(self):
        # compute the converged node number for each agent's FSC policy
        self.node_num = np.zeros(self.N)
        
        for n in range(self.N):
            a1 = np.sum(self.phi[n] - self.theta[n], axis = -1)
            self.node_num[n] = len(a1[a1 > 0])