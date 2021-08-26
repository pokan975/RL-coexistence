# -*- coding: utf-8 -*-
"""
@author: Po-Kan (William) Shih
@advisor: Dr. Bahman Moraffah
Coordinate ascent variational inference function for posterior approximation
**follow Liu's algorithm 1
"""
import numpy as np
import itertools as itt
import sys_parameter as var
import scipy.stats as st
from scipy.special import digamma, loggamma, logsumexp
from policy import initializeFSCs


class CAVI:
    gamma = var.gamma  # discount factor
    episodes = var.episode  # number of episodes
    T = var.T  # length for each episode
    
    def __init__(self):
        self.L = var.L  # number of LTE agents
        self.W = var.W  # number of WiFi agents
        self.N = var.N
        self.A = 7      # size of action set
        self.O = 20 + 1 # size of observation set
        self.Z = 50
        
        
    def init_prior(self):
        self.theta = []
        
        for n in range(self.N):
            # parameters of p(pi|z, theta) for n-th agent
            theta_n = 0.5 * np.ones((self.Z_cardinality[n], self.A)) / self.A
            self.theta.append(theta_n)
            
        # parameters of p(alpha|c, d) for n-th agents
        self.c = 0.1 * np.ones((self.N, self.A, self.O))
        self.d = 100 * np.ones((self.N, self.A, self.O))
        
        # parameters of p(rho|e, f) for all agents
        self.e = 0.1
        self.f = 100
    
    
    def init_q(self):
        # initialize lists for parameters of q distributions
        self.phi = []
        self.delta = []
        self.mu = []
        self.sigma = []
        self.lambda_ = []
        self.a = []
        self.b = []
        self.lnu = []
        self.lnV = []
        
        for n in range(self.N):
            # parameter of q(pi|z, phi) for n-th agent
            phi_n = np.array(self.theta[n])
            self.phi.append(phi_n)
            
            # parameter of q(u|delta, mu) for n-th agent
            delta_n = np.ones(self.Z_cardinality[n])
            self.delta.append(delta_n)
            mu_n = np.ones(self.Z_cardinality[n])
            self.mu.append(mu_n)
            self.lnu.append(mu_n)
            
            # parameter of q(V|sigma, lambda) for n-th agent
            sigma_n = np.ones((self.A, self.O, self.Z_cardinality[n], self.Z_cardinality[n]))
            self.sigma.append(sigma_n)
            lambda_n = np.ones((self.A, self.O, self.Z_cardinality[n], self.Z_cardinality[n]))
            self.lambda_.append(lambda_n)
            self.lnV.append(lambda_n)
            
            # parameters of q(alpha| a, b) for n-th agent
            a_n = np.ones((self.A, self.O, self.Z_cardinality[n]))
            self.a.append(a_n)
            b_n = np.ones((self.A, self.O, self.Z_cardinality[n]))
            self.b.append(b_n)
        
        # parameters of q(rho|g, h) for all agents
        self.g = np.ones(self.N)
        self.h = np.ones(self.N)
        # parameter evlution for q(rho|g, h)
        self.g_history = [self.g]
        self.h_history = [self.h]
        
        
    def fit(self, data, policy_list, max_iter = 150, tol = 1e-5):
        
        self.data = data  # list of trajectories
        
        # build initial FSC policies & get upper bounds of node number
        self.Z_cardinality = self.Z * np.ones(self.N, dtype = np.int)
        # self.FSC, self.Z_cardinality = initializeFSCs(self.N, self.episodes, self.data)
        
        # get behavior initial node distributions
        self.init_node = list(map(lambda n: n.eta, policy_list))
        # get behavior action probabilities
        self.behave_act_prob = list(map(lambda n: n.action_prob, policy_list))
        # get behavior node transition probabilities
        self.behave_node_prob = list(map(lambda n: n.node_prob, policy_list))
        
        # get action & observation histories of each agent
        self.action = []
        self._action = []
        self.obv = []
        self._obv = []
        
        for n in range(self.N):
            # get effective actions
            action_n = np.array(list(map(lambda x: x[0][n,:], self.data)))
            self.action.append(action_n)
            self._action.append(list(map(lambda x: x[x >= 0], action_n)))
            
            # get effective observations
            obv_n = np.array(list(map(lambda x: x[1][n,:], self.data)))
            self.obv.append(obv_n)
            self._obv.append(list(map(lambda x: x[x >= 0], obv_n)))
        
        # initialize prior models & posterior surrogates q
        self.init_prior()
        self.init_q()
        
        
        # reweight global rewards with behavior policies
        self.reweight_log_reward()
        # compute initial ELBO(q)
        self.elbo_values = [self.calc_ELBO()]
        # record the history of |Z|
        self.card_history = [self.Z_cardinality]
        self.value = []
        # CAVI iteration
        for it in range(1, max_iter + 1):
            # compute q(z)
            self.calc_eta_pi_omega() # compute EM eta, pi, and omega
            self.reweight_nu()  # compute \hat{nu}
            self.update_log_z() # compute each alpha and beta messages
            
            # compute q distributions
            self.update_u()     # update each q(u) distribution
            self.update_pi()    # update each q(pi) distribution
            self.update_v()     # update each q(v) distribution
            self.update_rho()   # update each q(rho) distribution
            self.update_alpha() # update each q(alpha) distribution
            # compute the cardinalities for learned policies so far
            self.card_history.append(self.calc_node_number())
            
            # compute ELBO(q) for every iteration
            self.elbo_values.append(self.calc_ELBO())
            # if converged, stop iteration
            if np.abs((self.elbo_values[-1] - self.elbo_values[-2])/self.elbo_values[-2]) <= tol:
                break
        
    
    def calc_ELBO(self):
        lowerbound = 0  # initialize ELBO value
        
        # pre-compute some values since they are being used multiple times later
        d1_u = list(map(lambda x: digamma(x), self.delta))
        d2_u = list(map(lambda x: digamma(x), self.mu))
        d12_u = list(map(lambda x, y: digamma(x + y), self.delta, self.mu))
        d2d12_u = list(map(lambda x, y: x - y, d2_u, d12_u))
        
        d1_v = list(map(lambda x: digamma(x), self.sigma))
        d2_v = list(map(lambda x: digamma(x), self.lambda_))
        d12_v = list(map(lambda x, y: digamma(x + y), self.sigma, self.lambda_))
        d2d12_v = list(map(lambda x, y: x - y, d2_v, d12_v))
        
        dalnb = list(map(lambda x, y: digamma(x) - np.log(y), self.a, self.b))
        dglnh = digamma(self.g) - np.log(self.h)
        dphi = list(map(lambda x: digamma(x) - digamma(np.sum(x, axis = -1))[..., None], self.phi))
        
        for n in range(self.N):
            # (1) E[lnp(alpha| c, d)]
            t1 = dalnb[n]
            t1 = (self.c[n] - 1) * np.sum(t1, axis = 2)
            t2 = self.a[n] / self.b[n]
            t2 = self.d[n] * np.sum(t2, axis = 2)
            palpha_n = t1 - t2
            lowerbound += palpha_n.sum()
        
            # (2) E[lnp(rho| e, f)]
            prho_n = (self.e - 1) * dglnh[n] - self.f * (self.g[n] / self.h[n])
            lowerbound += prho_n
        
            # (3) E[lnp(u| rho)]
            pu_n = d2d12_u[n] * ((self.g[n] / self.h[n]) - 1)
            pu_n += dglnh[n]
            lowerbound += pu_n.sum()
        
            # (4) E[lnp(V| alpha)]
            pv_n = d2d12_v[n] * ((self.a[n] / self.b[n]) - 1)[..., None]
            pv_n += dalnb[n][..., None]
            lowerbound += pv_n.sum()
        
            # (5) E[lnp(pi| theta)]
            pphi_n = (self.theta[n] - 1) * dphi[n]
            lowerbound += pphi_n.sum()
        
            # (6) E[lnq(alpha| a, b)]
            qalpha_n = self.a[n] * (digamma(self.a[n]) - 1)
            qalpha_n -= loggamma(self.a[n])
            qalpha_n -= dalnb[n]
            lowerbound -= qalpha_n.sum()
        
            # (7) E[lnq(rho| g, h)]
            qrho_n = self.g[n] * (digamma(self.g[n]) - 1)
            qrho_n -= loggamma(self.g[n])
            qrho_n -= dglnh[n]
            lowerbound -= qrho_n
        
            # (8) E[lnq(u| delta, mu)]
            qu_n = (self.delta[n] - 1) * (d1_u[n] - d12_u[n]) + (self.mu[n] - 1) * d2d12_u[n]
            qu_n += loggamma(self.delta[n] + self.mu[n])
            qu_n -= loggamma(self.delta[n])
            qu_n -= loggamma(self.mu[n])
            lowerbound -= qu_n.sum()

            # (9) E[lnq(V| sigma, lambda)]
            qv_n = (self.sigma[n] - 1) * (d1_v[n] - d12_v[n]) + (self.lambda_[n] - 1) * d2d12_v[n]
            qv_n += loggamma(self.sigma[n] + self.lambda_[n])
            qv_n -= loggamma(self.sigma[n])
            qv_n -= loggamma(self.lambda_[n])
            lowerbound -= qv_n.sum()
        
            # (10) E[lnq(pi| phi)]
            t1 = (self.phi[n] - 1) * dphi[n] - loggamma(self.phi[n])
            t2 = loggamma(np.sum(self.phi[n], axis = -1))
            qphi_n = t1.sum() + t2.sum()
            lowerbound -= qphi_n
        
        return lowerbound
    
    
    def calc_eta_pi_omega(self):
        # for building initial node probabilities
        d1_u = list(map(lambda x: digamma(x), self.delta))
        d2_u = list(map(lambda x: digamma(x), self.lnu))
        d12_u = list(map(lambda x, y: digamma(x + y), self.delta, self.lnu))
        
        # for building node transition probabilities
        d1_v = list(map(lambda x: digamma(x), self.sigma))
        d2_v = list(map(lambda x: digamma(x), self.lnV))
        d12_v = list(map(lambda x, y: digamma(x + y), self.sigma, self.lnV))
        
        self.eta = []
        self.pi = []
        self.omega = []
        
        for n in range(self.N):
            # initial node distribution
            eta_n = np.zeros(d1_u[n].shape)
            # transition probability to node 1~|Z_n|-1
            eta_n[: -1] = (d1_u[n] - d12_u[n])[: -1]
            # transition probability to node 2~|Z_n|
            eta_n[1: ] += (d2_u[n] - d12_u[n])[: -1].cumsum()
            eta_n = np.exp(eta_n)
            # eta_n /= np.sum(eta_n)
            self.eta.append(eta_n)
            
            # action distribution given node
            pi_n = digamma(self.phi[n]) - digamma(self.phi[n].sum(axis = 1))[..., None]
            pi_n = np.exp(pi_n)
            # pi_n /= np.sum(pi_n, axis = 1)[..., None]
            self.pi.append(pi_n)
            
            # node transition probabilities
            omega_n = np.zeros(d1_v[n].shape)
            # transition probability to node 1~|Z_n|-1
            omega_n[..., : -1] = (d1_v[n] - d12_v[n])[..., : -1]
            # transition probability to node 2~|Z_n|
            omega_n[..., 1: ] += (d2_v[n] - d12_v[n])[..., 0: -1].cumsum(axis = -1)
            omega_n = np.exp(omega_n)
            # omega_n /= np.sum(omega_n, axis = -1)[..., None]
            self.omega.append(omega_n)
        
            

    def update_log_z(self):
        # compute alpha and beta (forward and backward messages) for marginal q(z)
        self.log_alpha = [[] for i in range(self.N)]
        self.log_beta = [[] for i in range(self.N)]
        
        # process by each agent & episode
        n_k_pair = itt.product(range(self.N), range(self.episodes))
        for (n, k) in n_k_pair:
            # extract action history for agent n at episode k
            act = self._action[n][k]
            ob = self._obv[n][k]
            
            T_nk = len(act)  # effective length of episode k for agent n
            
            alpha_t = np.zeros((T_nk, self.Z_cardinality[n]))
            # compute ln(p(z_0, a_0))
            alpha_t[0, :] = np.log(self.eta[n] + 1e-200) + np.log(self.pi[n][:, act[0]])
            # compute ln(p(a_0|z_0))
            beta_t = [self.calc_log_beta(n, k, 0, act, ob)]
            
            for t in range(1, T_nk):
                
                # get ln(alpha) at time tau-1
                a = alpha_t[t - 1]
                # compute ln(p(z_{tau}, z_{tau-1}|a_{0:tau-1}, o_{1:tau}))
                # a = a[..., None] + np.log(self.omega[n][act[t-1], ob[t-1], ...])
                b = np.log(self.omega[n][act[t-1], ob[t-1], ...] + 1e-200)
                a = a[..., None] + b
                # get ln(p(a_{tau}|z_{tau}))
                p_a = np.log(self.pi[n][:, act[t]])
                # compute ln(p(z_{tau}, z_{tau-1}, a_{tau}|a_{0:tau-1}, o_{1:tau}))
                a += p_a[None, ...]
                # sum over tau-1 axis, get ln(p(z_{tau}, a_{tau}|a_{0:tau-1}, o_{1:tau}))
                alpha_t[t, :] = logsumexp(a, axis = 0)
                
                # compute beta_{tau} for history 0~t
                beta_t.append(self.calc_log_beta(n, k, t, act, ob))
             
            self.log_alpha[n].append(alpha_t)
            self.log_beta[n].append(beta_t)
        
            
    def calc_log_beta(self, n, k, t, act, ob):
        # backward message for agent n, episode k, up to time index t
        beta = np.zeros((t + 1, self.Z_cardinality[n]))
        
        if t > 0:
            for i in range(t - 1, -1, -1):
                # get ln(p(a_{tau+1}|z_{tau+1}))
                p_a = np.log(self.pi[n][:, act[i + 1]])
                # compute ln(p(z_{tau+1}, a_{tau+1}|z_{tau}, a_{tau}, o_{tau+1}))
                b = np.log(self.omega[n][act[i], ob[i], ...] + 1e-200) + p_a[None, ...]
                # * ln(beta) at time tau+1
                b += beta[i + 1][None, ...]
                # get ln(p(z_{tau+1}, a_{tau+1}|z_{tau}, a_{tau}, o_{tau+1}))
                beta[i, :] = logsumexp(b, axis = 1)
            
        return beta
    
    
    
    def update_u(self):
        # initialize delta & mu matrices for each agent
        delta = list(map(lambda d: np.zeros(d.shape), self.delta))
        mu = list(map(lambda m: np.zeros(m.shape), self.mu))
        
        # process by each agent & episode
        n_k_pair = itt.product(range(self.N), range(self.episodes))
        for (n, k) in n_k_pair:
            # get log alpha from time 0 (array)
            alpha = self.log_alpha[n][k]
            # get log beta to time 0~T (list of arrays)
            beta = self.log_beta[n][k]
            
            # find the indices where agent n contributes to global rewards in episode k
            v = tuple(np.where(self.action[n][k] >= 0)[0])
            # get nu values for above indices
            v = self.nu[k, v]
            
            for t in range(len(v)):
                # compute maringal q(z) at time 0 in history 0~t
                log_qz = alpha[0] + beta[t][0]
                # normalize q(z) to valid probability distribution
                # qz = log_qz - logsumexp(log_qz)
                # convert back to normal space
                qz = np.exp(log_qz)
                # qz = np.exp(qz)
                # update parameter delta
                delta[n][:] += (v[t] * qz)
                # update parameter mu
                qq = np.cumsum(qz[-1:0:-1])[::-1]
                mu[n][: -1] += (v[t] * qq)
                
        
        # add prior values
        self.delta = list(map(lambda d: d + 1, delta))
        self.lnu = list(map(lambda m, g, h: m + (g / h), mu, self.g, self.h))
        
        for n in range(self.N):
            rho_n = st.gamma(self.g[n], scale =  1/self.h[n]).rvs(size = self.Z_cardinality[n])
            mu[n] += rho_n
        
        self.mu = mu.copy()
        # self.mu = list(map(lambda m, g, h: m + (g / h), mu, self.g, self.h))
        
    
    
    def update_v(self):
        # initialize sigma & lambda matrices for each agent
        sigma = list(map(lambda s: np.zeros(s.shape), self.sigma))
        lambda_ = list(map(lambda l: np.zeros(l.shape), self.lambda_))
        
        # process by each agent & episode
        n_k_pair = itt.product(range(self.N), range(self.episodes))
        for (n, k) in n_k_pair:
            # get forward messages (array)
            alpha = self.log_alpha[n][k]
            # get backward messages to time 0~T (list of arrays)
            beta = self.log_beta[n][k]
            
            # find the indices where agent n contributes to global rewards in episode k
            v = tuple(np.where(self.action[n][k] >= 0)[0])
            # get nu values for above indices
            v = self.nu[k, v]
            
            # get action history for agent n in episode k
            eff_act = self._action[n][k]
            # get observation history for agent n in episode k
            eff_obv = self._obv[n][k]
            
            for t in range(len(eff_act)):
                for tau in range(1, t + 1):
                    # compute maringal q(z) at time tau in history 0~t
                    log_qz = alpha[tau-1][:, None] + np.log(self.omega[n][eff_act[tau-1], eff_obv[tau-1]] + 1e-200)
                    log_qz += np.log(self.pi[n][:, eff_act[tau]][None, :]) 
                    log_qz += beta[t][tau][None, :]
                    # normalize q(z) to valid probability distribution
                    # qz = log_qz - logsumexp(log_qz)
                    # convert back to normal space
                    qz = np.exp(log_qz)
                    # qz = np.exp(qz)
                    # update parameter sigma
                    sigma[n][eff_act[tau-1], eff_obv[tau-1], ...] += (v[t-1] * qz)
                    # update parameter lambda
                    qq = np.cumsum(qz[..., -1:0:-1], axis = -1)[..., ::-1]
                    lambda_[n][eff_act[tau-1], eff_obv[tau-1], :, :-1] += (v[t-1] * qq)
                    
        # add prior values
        self.sigma = list(map(lambda s: s + 1, sigma))
        self.lnV = list(map(lambda l, a, b: l + (a / b)[..., None], lambda_, self.a, self.b))
        
        for n in range(self.N):
            # aa = np.repeat(self.a[n][...,None], self.Z_cardinality[n], -1)
            # bb = np.repeat(self.b[n][...,None], self.Z_cardinality[n], -1)
            aa, bb = self.a[n].T, self.b[n].T
            dim = tuple([self.Z_cardinality[n]]) + aa.shape
            alpha_n = st.gamma(aa, scale = 1/bb).rvs(dim)
            lambda_[n] += alpha_n.T
        
        self.lambda_ = lambda_.copy()
        # self.lambda_ = list(map(lambda l, a, b: l + (a / b)[..., None], lambda_, self.a, self.b))
        
    
    def update_pi(self):
        # initialize phi matrix for each agent
        phi = list(map(lambda n: np.zeros(n.shape), self.theta))
        
        # process by each agent & episode
        n_k_pair = itt.product(range(self.N), range(self.episodes))
        for (n, k) in n_k_pair:
            # get forward messages (array)
            alpha = self.log_alpha[n][k]
            # get backward messages to time 0~T (list of arrays)
            beta = self.log_beta[n][k]
            
            # find the indices where agent n contributes to global rewards in episode k
            v = tuple(np.where(self.action[n][k] >= 0)[0])
            # get nu values for above indices
            v = self.nu[k, v]
            
            # get action history for agent n in episode k
            eff_act = self._action[n][k]
            
            for t in range(len(eff_act)):
                for tau in range(t + 1):
                    # compute maringal q(z) at time tau in history 0~t
                    log_qz = alpha[tau] + beta[t][tau]
                    # normalize q(z) to valid probability distribution
                    # qz = log_qz - logsumexp(log_qz)
                    # convert back to normal space
                    qz = np.exp(log_qz)
                    # qz = np.exp(qz)
                    # update parameter phi
                    phi[n][:, eff_act[tau]] += (v[t] * qz)
                    
        # add prior values
        self.phi = list(map(lambda p, t: p + t, phi, self.theta))

            
    
    def update_alpha(self):
        for n in range(self.N):
            c_ao = self.c[n]
            self.a[n] = c_ao[..., None] + self.Z_cardinality[n]* np.ones(self.a[n].shape)
            
            d_ao = self.d[n]
            ln1_V = st.beta(self.sigma[n], self.lambda_[n]).rvs()
            ln1_V = np.log(1 - ln1_V + 1e-200)
            self.b[n] = d_ao[..., None] - np.sum(ln1_V, axis = -1)
        
        
    def update_rho(self):
        self.g = self.e + self.Z_cardinality
        
        h = np.empty_like(self.h)
        for n in range(self.N):
            ln1_u = st.beta(self.delta[n], self.mu[n]).rvs()
            ln1_u = np.log(1 - ln1_u + 1e-200)
            h[n] = self.f - np.sum(ln1_u)
            
        self.h = np.array(h)
        # record parameter evolution for q(rho|g, h)
        self.g_history.append(self.g)
        self.h_history.append(self.h)
        

    def reweight_nu(self):
        # initialize reweighted nu (computation in log space)
        self.nu = np.array(self.reweight_log_r)
                
        # for each episode, compute rewighted rewards weighted by sample policies
        for k in range(self.episodes):
            # initialize all agents' ln(p(a_{0:t-1}, z_{t-1}|o_{0:t})) arrays,
            # initial value is ln(p(z_0))
            ln_q_az_o = list(map(lambda p: np.log(p + 1e-200), self.eta))
            # tracker for tracking the latest indices of actions for each agent
            index_tracker = np.zeros(self.N, dtype = np.int)
            
            # get joint action history of episode k
            joint_action = self.data[k][0]
            
            for t in range(self.T):
                # extract agent indices which contribute to reward r_t^k
                contributors = tuple(np.where(joint_action[:, t] >= 0)[0])
                
                # get all contributors' ln(p(a_{0:t-1}, z_{t-1}|o_{0:t}))
                eff_ln_q_az_o = list(map(lambda a: ln_q_az_o[a], contributors))
                
                # for each contributor, compute its ln(p(a_{0:t-1}|o_{0:t}))
                for i, n in enumerate(contributors):
                    # if t > 0, need to extra multiply ln(p(z_t|z_{t-1}, a_{t-1}, o_t))
                    if index_tracker[n] > 0:
                        # get action & observation at time t-1
                        act_pre = self._action[n][k][index_tracker[n] - 1]
                        obv_pre = self._obv[n][k][index_tracker[n] - 1]
                        # get p(z_t|z_{t-1}, a_{t-1}, o_t)
                        ln_q_z_zao = np.log(self.omega[n][act_pre, obv_pre, ...] + 1e-200)
                        # compute p(z_t, z_{t-1}, a_{0:t-1}|o_{1:t}) then
                        # marginalize z_{t-1} out
                        # add a scalar b/c we use log-sum-exp trick
                        ln_q_z_zao += eff_ln_q_az_o[i][:, None]
                        eff_ln_q_az_o[i] = logsumexp(ln_q_z_zao, axis = 0)
                    
                    # get action index at time t
                    act_cur = self._action[n][k][index_tracker[n]]
                    # get ln(p(a_t|z_t))
                    ln_q_a_z = np.log(self.pi[n][:, act_cur])
                    # compute ln(p(z_t, a_{0:t}|o_{1:t}))
                    eff_ln_q_az_o[i] += ln_q_a_z
                    
                    # overwrite ln(p(a_{0:t-1}, z_{t-1}|o_{0:t})) for agent n
                    ln_q_az_o[n][:] = eff_ln_q_az_o[i]
                        
                # tracker + 1 for all effective agents at time t
                index_tracker[np.array(contributors)] += 1
                
                # marginalize p(a_{0:t-1}, z_{t-1}|o_{0:t}) to get p(a_{0:t-1}|o_{0:t})
                # for all effective agents
                joint_q_a_o = np.fromiter(map(logsumexp, eff_ln_q_az_o), dtype = np.float)
                # reweight reward with Prod_{n}p(a_{0:t-1}|o_{0:t})
                self.nu[k, t] += np.sum(joint_q_a_o)
                
        # nu cimputation finished, convert back to normal space
        self.value.append(np.exp(self.nu).sum())
        self.nu -= logsumexp(self.nu)
        self.nu = np.exp(self.nu)
    
    
    
    def reweight_log_reward(self):
        # extract rewards from trajectories
        self.reweight_log_r = np.array(list(map(lambda r: r[2], self.data)), dtype = np.float)
        
        # shift & rescale rewards, then convert to log space
        self.reweight_log_r = np.log(self.reweight_log_r + 1)
        # r_max = np.max(self.reweight_log_r)
        # r_min = np.min(self.reweight_log_r)
        # self.reweight_log_r = np.log((self.reweight_log_r - r_min + 1) / (r_max - r_min + 1))
        
        # impose discount factor (log space)
        discount = np.ones(self.reweight_log_r.shape)
        discount = (np.cumsum(discount, axis = 1) - 1) * np.log(self.gamma)
        self.reweight_log_r += discount
        

        # for each episode, compute global rewards reweighted by behavior policies
        for k in range(self.episodes):
            # initialize all agents' ln(p(a_{0:t-1}, z_{t-1}|o_{0:t})) arrays,
            # initial value is ln(p(z_0))
            ln_p_az_o = list(map(lambda p: np.log(p), self.init_node))
            # tracker for tracking the latest indices of actions for each agent
            index_tracker = np.zeros(self.N, dtype = np.int)
            
            # get joint action history of episode k
            joint_action = self.data[k][0]
            
            for t in range(self.T):
                # extract agent indices which contribute to reward r_t^k
                contributors = tuple(np.where(joint_action[:, t] >= 0)[0])
                # get all contributors' ln(p(a_{0:t-1}, z_{t-1}|o_{0:t}))
                eff_ln_p_az_o = list(map(lambda a: ln_p_az_o[a], contributors))
                
                # for each contributor, compute its ln(p(a_{0:t-1}|o_{0:t}))
                for i, n in enumerate(contributors):
                    # if t > 0, need to extra add ln(p(z_t|z_{t-1}, a_{t-1}, o_t))
                    if index_tracker[n] > 0:
                        # get action & observation at time t-1
                        act_pre = self._action[n][k][index_tracker[n] - 1]
                        obv_pre = self._obv[n][k][index_tracker[n] - 1]
                        # get ln(p(z_t|z_{t-1}, a_{t-1}, o_t))
                        ln_p_z_zao = np.log(self.behave_node_prob[n][act_pre, obv_pre, ...])
                        # compute p(z_t, z_{t-1}, a_{0:t-1}|o_{1:t}) then
                        # marginalize z_{t-1} out
                        # add a scalar b/c we use log-sum-exp trick
                        ln_p_z_zao += eff_ln_p_az_o[i][:, None]
                        eff_ln_p_az_o[i] = logsumexp(ln_p_z_zao, axis = 0)
                    
                    # get action index at time t
                    act_cur = self._action[n][k][index_tracker[n]]
                    # get ln(p(a_t|z_t))
                    ln_p_a_z = np.log(self.behave_act_prob[n][:, act_cur])
                    # compute ln(p(z_t, a_{0:t}|o_{1:t}))
                    eff_ln_p_az_o[i] += ln_p_a_z
                    
                    # overwrite ln(p(a_{0:t-1}, z_{t-1}|o_{0:t})) for agent n
                    ln_p_az_o[n][:] = eff_ln_p_az_o[i]
                
                # tracker + 1 for all effective agents at time t
                index_tracker[np.array(contributors)] += 1
                # marginalize p(a_{0:t-1}, z_{t-1}|o_{0:t}) to get p(a_{0:t-1}|o_{0:t})
                # for all effective agents
                joint_p_a_o = np.fromiter(map(logsumexp, eff_ln_p_az_o), dtype = np.float)
                
                # reweight reward with Sum_{n}ln(p(a_{0:t-1}|o_{0:t}))
                self.reweight_log_r[k, t] -= np.sum(joint_p_a_o)


    def calc_node_number(self):
        # compute the converged node number for each agent's FSC policy
        node_num = np.zeros(self.N)
        # record the node indices for positive reward after each VI iteration
        self.remainingNodes = []
        
        for n in range(self.N):
            a1 = np.sum(self.phi[n] - self.theta[n], axis = 1)
            # find the indices of nodes with positive reward
            nodes_pos_rwd = tuple(np.where(a1 > 0)[0])
            self.remainingNodes.append(nodes_pos_rwd)
            # compute the number of nodes with positive reward assigned to
            node_num[n] = len(nodes_pos_rwd)
            assert node_num[n]> 0
            
        return node_num
    
    
    def logSumExp(self, arr, axis):
        sumOfExp = np.exp(arr).sum(axis = axis)
        
        return np.log(sumOfExp)
            
