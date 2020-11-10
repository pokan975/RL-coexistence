# -*- coding: utf-8 -*-
"""
@author: Po-Kan (William) Shih
@advisor: Dr. Bahman Moraffah
Main code for simulation
"""

import Variables as var
import numpy as np
import matplotlib.pyplot as plt

from POMDP_5 import POMDP
from Variational_Bayes3 import CAVI

# num of episodes at each collection
episode = 2
# number of learning iterations
max_iterations = 500  
# exploration factor
epsilon = 0.5


# initialize POMDP & CAVI models
model = POMDP()
model.initialize()
# get action set size
act_setsize = len(model.action_set)
# do CAVI for q
vi = CAVI(var.L, var.W, act_setsize)


# collecting initial data to generate initial behavior policy
# model.init_behavior_policy(500)
# obv_setsize = len(model.observation)
# truncated number of nodes
Z = 200

# learning loop
for m in range(1):
    # perform data collection
    model.collect_episode(episode, var.T, Z, 0)

    #obv_setsize = len(model.observation)
    obv_setsize = model.observation.loc[model.observation['value'] >= 0].size
    # get behavior policy for computing rewards
    policy_list = model.policies
    # retrieve data
    data = model.episodes
    # compute truncation level for q
    # for code simplicity, use max truncation level for all agents
    # Z = max(map(lambda x: len(x), model.unique_a_o_r))

    # initialize prior models
    # input observation set size & node number
    vi.prior_param(obv_setsize, Z)
    # perform VI
    vi.fit(data, policy_list)


# RL learning main loop
# for itation in range(max_iterations):

plt.plot(vi.elbo_values, marker = "o")
plt.xlabel("iteration")
plt.ylabel("ELBO value")
plt.grid()
plt.show()
