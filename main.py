# -*- coding: utf-8 -*-
"""
@author: Po-Kan (William) Shih
@advisor: Dr. Bahman Moraffah
Main code for simulation
"""

import Variables as var
import numpy as np
import matplotlib.pyplot as plt

from POMDP_6 import POMDP
from Variational_Bayes3 import CAVI

plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10

###############################################################################
# for learning phase
# num of episodes at each collection
episode = 2
# number of learning iterations
max_iterations = 2
# exploration factor
epsilon = np.linspace(0.9, 0.8, max_iterations)
# truncated number of nodes
Z = 200

obv_setsize = 20

###############################################################################
# for evaluation phase
eval_episode = 100
eval_T = 500

###############################################################################
# initialize POMDP & CAVI models
model = POMDP()
model.initialize()
# get action set size
act_setsize = len(model.action_set)
# initialize CAVI object
vi = CAVI(var.L, var.W, act_setsize, obv_setsize, Z)
# initialize prior models
# input observation & node sets cardinality
# vi.prior_param()


# collecting initial data to generate initial behavior policy
# model.init_behavior_policy(500)
# obv_setsize = len(model.observation)

# initialize learned parameter array
phi = np.array([None] * var.N)

z_cardinality = []

# learning loop
for m in range(max_iterations):
    # perform data collection
    model.collect_episode(episode, var.T, Z, epsilon[m], phi)

    # retrieve data
    # obv_setsize = model.observation.loc[model.observation['value'] >= 0].size
    # get behavior policy for computing rewards
    policy_list = model.policies
    data = model.episodes
    # compute truncation level for q
    # for code simplicity, use max truncation level for all agents
    # Z = max(map(lambda x: len(x), model.unique_a_o_r))

    # update prior models
    vi.update_prior()
    # perform VI
    vi.fit(data, policy_list)
    
    # compute node set cardinality
    z_cardinality.append(vi.calc_node_number())
    
    # retrieve learned parameters for next iteration & performance evaluation
    theta = vi.theta
    phi = vi.phi
    sigma = vi.sigma
    lambda_ = vi.lambda_


# plot ELBO curve
plt.plot(vi.elbo_values, marker = "o")
plt.xlabel("iteration")
plt.ylabel("ELBO value")
plt.grid()
plt.show()