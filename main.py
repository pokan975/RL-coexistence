# -*- coding: utf-8 -*-
"""
@author: Po-Kan (William) Shih
@advisor: Dr. Bahman Moraffah
Main code for simulation
"""

import Variables as var
import numpy as np
import matplotlib.pyplot as plt

from POMDP_4 import POMDP
from Variational_Bayes import CAVI

# num of episodes at each collection
episode = 2

# number of RL iterations
max_iterations = 500  


# generate POMDP world model
model = POMDP()
model.initialize()
# get action set sizes for each agent
act_setsize = len(model.action_set)

# collecting initial data
model.init_behavior_policy(50)

# find initial number of nodes & observation set
Z = max(map(lambda x: len(x), model.unique_a_o_r))
obv_setsize = len(model.observation)

# perform data collection
model.collect_episode(episode, var.T, 50, 0)

obv_setsize = len(model.observation)
policy_list = model.policies
data = model.episodes
history = model.histories


# compute truncation level for q
# for code simplicity, use max truncation level for all agents

# do CAVI for q
vi = CAVI(var.L, var.W, 51, act_setsize, obv_setsize)
vi.fit(data, policy_list)

# generate initial prior models

# generate initial behavior policy


# RL learning main loop
# for itation in range(max_iterations):
    