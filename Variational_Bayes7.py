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
    
    def __init__(self, L, W, A, O, Z):
        self.L = L  # number of LTE agents
        self.W = W  # number of WiFi agents
        self.N = self.L + self.W
        self.A = A  # size of action set
        self.O = O  # size of observation set
        self.Z = Z  # truncation level for q
        
        
    def update_prior(self):
        # parameters of p(pi|z, theta) for (n, i) LTE & WiFi agents
        self.theta = np.ones((self.N, self.Z, self.A)) / self.A
        # parameters of p(alpha|c, d) for (n, a, o) LTE & WiFi agents
        self.c = 0.1*np.ones((self.N, self.A, self.O))
        self.d = 10*np.ones((self.N, self.A, self.O))
        # parameters of p(rho|e, f) for (n) LTE & WiFi agents
        self.e = 0.1
        self.f = 10
    
    
    def init_q_param(self):
        '''
        Returns
        -------
        initialize variational distributions q.
        '''
        # parameter of q(u|delta, mu) for (n, i) agent
        self.delta = np.ones((self.N, self.Z))
        self.mu = np.ones((self.N, self.Z))
        
        # parameter of q(V|sigma, lambda) for (n, a, o, i, j) agent
        self.sigma = np.ones((self.N, self.A, self.O, self.Z, self.Z))
        self.lambda_ = np.ones((self.N, self.A, self.O, self.Z, self.Z))
        
        # parameter of q(pi|z, phi) for (n, i) agent
        self.phi = np.empty_like(self.theta)
        self.phi[...] = self.theta
        
        # parameters of q(alpha| a, b) for each (n, a, o, i) agent
        self.a = 0.1*np.ones((self.N, self.A, self.O, self.Z))
        self.b = 10*np.ones((self.N, self.A, self.O, self.Z))
        
        # parameters of q(rho|g, h) for each (n) agent
        self.g = 0.1*np.ones(self.N)
        self.h = 10*np.ones(self.N)
        
        
    def fit(self, data, policy_list, max_iter = 50, tol = 1e-8):
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
        self.init_node = np.array(list(map(lambda n: policy_list[n].eta, range(self.N))))
        
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
        self.card = [np.array([200,200])]
        # CAVI iteration
        for it in range(1, max_iter + 1):
            # CAVI update
            self.calc_eta_pi_omega() # calc EM eta & pi & omega
            self.reweight_nu()  # compute \hat{nu}
            self.update_z()     # update each q(z) distribution
            self.update_u()     # update each q(u) distribution
            self.update_v()     # update each q(v) distribution
            self.update_pi()    # update each q(pi) distribution
            self.update_rho()   # update each q(rho) distribution
            self.update_alpha() # update each q(alpha) distribution
            # calc ELBO(q) at the end of all updates
            self.elbo_values.append(self.calc_ELBO())
            self.card.append(self.calc_node_number())
            # if converged, stop iteration
            if np.abs(self.elbo_values[-2] - self.elbo_values[-1]) <= tol:
                break
        
    
    def calc_ELBO(self):
        # pre-compute some values since they are being used later
        d1_u = digamma(self.delta)
        d2_u = digamma(self.mu)
        d12_u = digamma(self.delta + self.mu)
        d2d12_u = d2_u - d12_u
        
        d1_V = digamma(self.sigma)
        d2_V = digamma(self.lambda_)
        d12_V = digamma(self.sigma + self.lambda_)
        d2d12_V = d2_V - d12_V
        
        dalnb = digamma(self.a) - np.log(self.b)
        dglnh = digamma(self.g) - np.log(self.h)
        dphi = digamma(self.phi) - digamma(np.sum(self.phi, axis = -1))[..., np.newaxis]
        
        lowerbound = 0
        
        # (1) E[lnp(alpha| c, d)]
        t1 = (self.c - 1)[..., np.newaxis] * dalnb
        t2 = self.d[..., np.newaxis] * (self.a / self.b)
        palpha = t1 - t2
        lowerbound += palpha.sum()
        
        # (2) E[lnp(rho| e, f)]
        prho = (self.e - 1) * dglnh - self.f * (self.g / self.h)
        lowerbound += prho.sum()
        
        # (3) E[lnp(u| rho)]
        pu = d2d12_u * ((self.g / self.h) - 1)[..., np.newaxis]
        pu += dglnh[..., np.newaxis]
        lowerbound += pu.sum()
        
        # (4) E[lnp(V| alpha)]
        pv = d2d12_V * ((self.a / self.b) - 1)[..., np.newaxis]
        pv += dalnb[..., np.newaxis]
        lowerbound += pv.sum()
        
        # (5) E[lnp(pi| theta)]
        pphi = (self.theta - 1) * dphi
        lowerbound += pphi.sum()
        
        # (6) E[lnq(alpha| a, b)]
        qalpha = self.a * (digamma(self.a) - 1)
        qalpha -= loggamma(self.a)
        qalpha -= dalnb
        lowerbound -= qalpha.sum()
        
        # (7) E[lnq(rho| g, h)]
        qrho = self.g * (digamma(self.g) - 1)
        qrho -= loggamma(self.g)
        qrho -= dglnh
        lowerbound -= qrho.sum()
        
        # (8) E[lnq(u| delta, mu)]
        t1 = (self.delta - 1) * (d1_u - d12_u)
        t2 = (self.mu - 1) * d2d12_u
        qu = t1 + t2 + loggamma(self.delta + self.mu)
        qu -= loggamma(self.delta)
        qu -= loggamma(self.mu)
        lowerbound -= qu.sum()

        # (9) E[lnq(V| sigma, lambda)]
        t1 = (self.sigma - 1) * (d1_V - d12_V)
        t2 = (self.lambda_ - 1) * d2d12_V
        qv = t1 + t2 + loggamma(self.sigma + self.lambda_)
        qv -= loggamma(self.sigma)
        qv -= loggamma(self.lambda_)
        lowerbound -= qv.sum()
        
        # (10) E[lnq(pi| phi)]
        t1 = (self.phi - 1) * dphi - loggamma(self.phi)
        t2 = loggamma(np.sum(self.phi, axis = -1))
        qphi = t1.sum() + t2.sum()
        lowerbound -= qphi
        
        return lowerbound
    
    
    def calc_eta_pi_omega(self):
        # build init node probability eta
        d1 = digamma(self.delta)
        d2 = digamma(self.mu)
        d12 = digamma(self.delta + self.mu)
        self.eta = np.zeros(self.delta.shape)
        # build trans prob to node 1
        self.eta[..., : -1] = (d1 - d12)[..., : -1]
        # build trans prob to node 2~|Z|-1
        self.eta[..., 1:] += (d2 - d12)[..., 0: -1].cumsum(axis = -1)
        self.eta = np.exp(self.eta)
        self.eta /= np.sum(self.eta, axis = -1)[..., np.newaxis]
        
        # build action prob pi
        t1 = digamma(self.phi) - digamma(np.sum(self.phi, axis = -1))[..., np.newaxis]
        self.pi = np.exp(t1)
        # normalize to valid prob distribution
        self.pi /= np.sum(self.pi, axis = -1)[..., np.newaxis]
        
        # build node transition prob omega
        d1 = digamma(self.sigma)
        d2 = digamma(self.lambda_)
        d12 = digamma(self.sigma + self.lambda_)
        self.omega = np.zeros(self.sigma.shape)
        # build trans prob to node 1
        self.omega[..., : -1] = (d1 - d12)[..., : -1]
        # self.omega[...] = (d1 - d12)
        # build trans prob to node 2~|Z|-1
        self.omega[..., 1:] += (d2 - d12)[..., 0: -1].cumsum(axis = -1)
        self.omega = np.exp(self.omega)
        self.omega /= np.sum(self.omega, axis = -1)[..., np.newaxis]
    

    def update_z(self):
        # compute alpha and beta (forward and backward messages) for marginal q(z)
        self.fwd_alpha = [[] for i in range(self.N)]
        self.back_beta = [[] for i in range(self.N)]
        
        n_k_pair = itt.product(range(self.N), range(self.ep))
        
        for (n, k) in n_k_pair:
            # extract action history for agent n at episode k
            act = self._action[n][k]
            ob = self._obv[n][k]
            
            alpha_t = np.zeros((len(act), self.Z))
            alpha_t[0, :] = self.eta[n] * self.pi[n, :, act[0]]
            
            beta_t = [self.calc_beta(n, k, 0)]
            
            for t in range(1, len(act)):
                beta_t.append(self.calc_beta(n, k, t))
                
                a = alpha_t[t - 1][..., np.newaxis] * self.omega[n, act[t-1], ob[t-1], ...]
                a *= self.pi[n, :, act[t]][np.newaxis, ...]
                alpha_t[t, :] = np.sum(a, axis = 0)
             
            self.fwd_alpha[n].append(alpha_t)
            self.back_beta[n].append(beta_t)
        
            
    def calc_beta(self, n, k, t):
        # backward message for agent n, episode k, up to time index t
        beta = np.zeros((t + 1, self.Z))
        beta[t, :] = np.ones(self.Z)
        
        if t > 0:
            act = self._action[n][k][: t + 1]
            ob = self._obv[n][k][: t + 1]
        
            for i in range(t - 1, -1, -1):
                b = self.omega[n, act[i], ob[i], ...] * self.pi[n, :, act[i + 1]][np.newaxis, ...]
                b *= beta[i + 1][np.newaxis, ...]
                beta[i, :] = np.sum(b, axis = 1)
            
        return beta
    
    
    def update_u(self):
        self.delta = np.zeros(self.delta.shape)
        self.mu = np.zeros(self.mu.shape)
        
        n_k_pair = itt.product(range(self.N), range(self.ep))
        for (n, k) in n_k_pair:
            alpha = self.fwd_alpha[n][k] # array
            beta = self.back_beta[n][k]  # list of arrays
            
            # get nu for (n, k)
            v = tuple(np.where(self.action[n][k] >= 0)[0])
            v = self.nu[k, v]
            
            for t in range(len(v)):
                qz = alpha[0] * beta[t][0]
                if np.sum(qz) > 0:
                    qz /= np.sum(qz)
                
                self.delta[n] += (v[t] * qz / self.v_k)
                
                qq = np.cumsum(qz[-1:0:-1])[::-1]
                self.mu[n, : -1] += (v[t] * qq / self.v_k)
                
        self.delta = (self.delta / self.ep) + 1
        self.mu = (self.mu / self.ep) + (self.g / self.h)[..., None]
        
    
    
    def update_v(self):
        self.sigma = np.zeros(self.sigma.shape)
        self.lambda_ = np.zeros(self.lambda_.shape)
        
        n_k_pair = itt.product(range(self.N), range(self.ep))
        for (n, k) in n_k_pair:
            alpha = self.fwd_alpha[n][k] # array
            beta = self.back_beta[n][k]  # list of arrays
            
            # get nu for (n, k)
            v = tuple(np.where(self.action[n][k] >= 0)[0])
            v = self.nu[k, v]
            
            # get act #
            aa = self._action[n][k]
            # get obv #
            oo = self._obv[n][k]
            
            for t in range(len(aa)):
                for tau in range(1, t + 1):
                    qz = alpha[tau-1][:, np.newaxis] * self.omega[n, aa[tau-1], oo[tau-1], ...]
                    qz *= self.pi[n, :, aa[tau]][np.newaxis, :] 
                    qz *= beta[t][tau][np.newaxis, :]
                    if np.sum(qz) > 0:
                        qz /= np.sum(qz)
                    
                    self.sigma[n, aa[tau-1], oo[tau-1], ...] += (v[t-1] * qz / self.v_k)
                    
                    qq = np.cumsum(qz[..., -1:0:-1], axis = -1)[..., ::-1]
                    self.lambda_[n, aa[tau-1], oo[tau-1], :, :-1] += (v[t-1] * qq / self.v_k)
                    
        self.sigma = (self.sigma / self.ep) + 1
        self.lambda_ = (self.lambda_ / self.ep) + (self.a / self.b)[..., None]
        
    
    def update_pi(self):
        self.phi = np.zeros(self.theta.shape)
        
        n_k_pair = itt.product(range(self.N), range(self.ep))
        for (n, k) in n_k_pair:
            alpha = self.fwd_alpha[n][k] # array
            beta = self.back_beta[n][k]  # list of arrays
            
            # get nu for (n, k)
            v = tuple(np.where(self.action[n][k] >= 0)[0])
            v = self.nu[k, v]
            
            # get act #
            aa = self._action[n][k]
            
            for t in range(len(aa)):
                for tau in range(t + 1):
                    qz = alpha[tau] * beta[t][tau]
                    if np.sum(qz) > 0:
                        qz /= np.sum(qz)
                    self.phi[n, :, aa[tau]] += (v[t] * qz / self.v_k)
                    
        self.phi = (self.phi / self.ep) + self.theta
            
    
    def update_alpha(self):
        self.a = self.c[..., None] + self.Z * np.ones(self.a.shape)
        
        d2d12 = digamma(self.lambda_) - digamma(self.sigma + self.lambda_)
        self.b = self.d[..., None] - np.sum(d2d12, axis = -1)
        
        
    def update_rho(self):
        self.g = self.e + self.Z * np.ones(self.g.shape)
        
        d2d12 = digamma(self.mu) - digamma(self.delta + self.mu)
        self.h = self.f - np.sum(d2d12, axis = -1)
        

    def reweight_nu(self):
        # extract rewards from each episode
        self.nu = np.array(self.reweight_r)
                
        # in each episode, compute reweighted rewards
        for k in range(self.ep):
            # initialize p(z) array to track p(z_{t-1}, a_{t-1}, ..., a_0)
            q_az_o = np.empty_like(self.eta)
            q_az_o[...] = self.eta
            # action tracker array, used to track the order of action for each agent
            action_num = np.zeros(self.N, dtype = np.int)
            
            # get action & observation arrays for episode k
            act_array = self.data[k][0]
            for t in range(self.T):
                # extract the agents who contribute reward at time k
                agent_idx = tuple(np.where(act_array[:, t] >= 0)[0])
                
                # extract initial node prob for optimal policy
                q_eta = np.array(q_az_o[agent_idx, ...], ndmin = 2)
                
                # if t > 0, need to times p(z|z, a, o) extra
                for ii, nn in enumerate(agent_idx):
                    if action_num[nn] > 0:
                        act_pre = self._action[nn][k][action_num[nn] - 1]
                        obv_pre = self._obv[nn][k][action_num[nn] - 1]
                        
                        q_node = np.array(self.omega[nn, act_pre, obv_pre, ...])
                        q_eta[ii] = (q_node * q_eta[ii][:, np.newaxis]).sum(axis = 0)
                    
                    # extract their action #
                    act_idx = self._action[nn][k][action_num[nn]]#act_array[nn, t]
                    # extract corresponding taken action prob given all nodes
                    q_act = self.pi[nn, :, act_idx]
                    # compute p(a,z)=p(z)p(a|z)
                    q_eta[ii] *= q_act
                    
                    # replace p(z)
                    q_az_o[nn, :] = q_eta[ii, :]
                        
                action_num[np.array(agent_idx)] += 1
                self.nu[k, t] *= np.prod(np.sum(q_eta, axis = -1))
                
        
        # self.nu += 1 # test
        self.v_k = 1#np.sum(self.nu) / self.ep


    def reweight_reward(self):
        # extract rewards from each episode
        self.reweight_r = np.array(list(map(lambda t: t[2], self.data)), dtype = np.float)
        
        # rescale rewards
        r_max = np.max(self.reweight_r)
        r_min = np.min(self.reweight_r)
        self.reweight_r = (self.reweight_r - r_min + 1) / (r_max - r_min + 1)
        # impose discount factor
        ga = np.ones((self.ep, self.T))
        ga[:, 1:] *= 0.9
        self.reweight_r *= np.cumprod(ga, axis = 1)
        
        self.rr = np.empty_like(self.reweight_r)
        self.rr[...] = self.reweight_r
        
        # in each episode, compute reweighted rewards
        for k in range(self.ep):
            # initialize p(z) array to track p(z_{t-1}, a_{t-1}, ..., a_0|o_t, ..., o_0)
            p_az_o = np.empty_like(self.init_node)
            p_az_o[...] = self.init_node
            # action tracker array, used to track the order of action for each agent
            action_num = np.zeros(self.N, dtype = np.int)
            
            # get action & observation arrays for episode k
            act_array = self.data[k][0]
            # obv_array = self.data[k][1]
            for t in range(self.T):
                # extract agents # who contribute to reward at time t
                agent_idx = tuple(np.where(act_array[:, t] >= 0)[0])
                
                # extract p(z_t-1, a_t-1, ..., a_0)
                p_eta = np.array(p_az_o[agent_idx, ...], ndmin = 2)
                
                # if t > 0, need to multiply p(z|z, a, o) extra
                for ii, nn in enumerate(agent_idx):
                    if action_num[nn] > 0:
                        act_pre = self._action[nn][k][action_num[nn] - 1]
                        obv_pre = self._obv[nn][k][action_num[nn] - 1]
                        # extract p(z_t|z_t-1, a_t-1, o_t)
                        p_node = np.array(self.behave_node_prob[nn, act_pre, obv_pre, ...])
                        # p(z_t, a_t-1, ..., a_0)
                        p_eta[ii] = (p_node * p_eta[ii][:, np.newaxis]).sum(axis = 0)
                    
                    # extract action #
                    act_idx = self._action[nn][k][action_num[nn]]#act_array[nn, t]
                    # extract p(a_t|z_t)
                    p_act = self.behave_act_prob[nn, :, act_idx]
                    # compute p(z_t, a_t, ..., a_0)
                    p_eta[ii] *= p_act
                    
                    # replace p(z)
                    p_az_o[nn, :] = p_eta[ii, :]
                        
                action_num[np.array(agent_idx)] += 1
                                    
                assert np.prod(np.sum(p_eta, axis = -1)) > 0
                self.reweight_r[k, t] /= np.prod(np.sum(p_eta, axis = -1))


    def calc_node_number(self):
        # compute the converged node number for each agent's FSC policy
        node_num = np.zeros(self.N)
        
        for n in range(self.N):
            a1 = np.sum(self.phi[n] - self.theta[n], axis = -1)
            node_num[n] = len(a1[a1 > 0])
            
        return node_num