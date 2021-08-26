# -*- coding: utf-8 -*-
"""
@author: Po-Kan (William) Shih
@advisor: Dr. Bahman Moraffah
Main code for simulation
main file for exploration rate (epsilon 1) 
for running on Agave node
"""
import numpy as np
import threading
import scipy.io as io
import matplotlib.pyplot as plt
import sys_parameter as var
from itertools import cycle
from environment import Spectrum
from var_inference import CAVI

###############################################################################
# program settings
###############################################################################
# plot setting
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10
# mark = cycle(["o","v","^","<",">","1","2","3","4","8","s","p","*","h","H","+","x","d","D","|","P","X"])
mark = ["o","v","^","<",">","1","2","3","4","8","s","p","*","h","H","+","x","d","D","|","P","X"]
color = 'rgbcmykw'
linecycler = cycle(["-","--","-.",":"])

# create a file to record learning procedure
with open("epsilon1.txt", "w+") as file:
    pass

iterations = 40
    
# build exploration rates for learning process
eps_step = np.arange(1, iterations + 1)
epsilon1 = eps_step**5
epsilon2 = eps_step**2
epsilon1 = (epsilon1 / np.max(epsilon1)) * (0.9 - 0.2) + 0.2
epsilon2 = (epsilon2 / np.max(epsilon2)) * (0.9 - 0.2) + 0.2
epsilon1 = np.round(epsilon1[::-1], 2)
epsilon2 = np.round(epsilon2[::-1], 2)


###############################################################################
# thread body for learning process 1 (exploration rate = epsilon1)
###############################################################################
def learningProcess_1():
    # dec-POMDP environment
    dec_pomdp = Spectrum()
    
    # variational inference object
    cavi = CAVI()

    # initial policy parameters
    delta = np.array([None] * var.N)
    mu = np.array([None] * var.N)
    phi = np.array([None] * var.N)
    sigma = np.array([None] * var.N)
    lambda_ = np.array([None] * var.N)

    # history of node cardinality for all agents throughout learning process
    z_cardinality = []
    # histories of mean & total rewards throughout learning process 
    mean_rewards = np.zeros(iterations + 1)
    max_rewards = np.zeros(iterations + 1)
    
    # learning iteration
    for it in range(iterations):
        # record current iteration #
        with open("epsilon1.txt", "a+") as file:
            file.write("iteration {}\n".format(it + 1))
        
        # print("learning process 1 iteration:", it + 1)
        
        # collect trajectories
        dec_pomdp.collect_episodes(var.episode, var.T, epsilon1[it], delta, mu, phi, sigma, lambda_)

        # retrieve trajectories & behavior policies
        policy_list = dec_pomdp.policies
        data = dec_pomdp.episodes

        # approximate posterior distributions
        cavi.fit(data, policy_list)
    
        # compute cardinality of controller nodes
        z_cardinality.append(cavi.calc_node_number())
    
        # retrieve learned parameters for next iteration & performance evaluation
        delta = cavi.delta
        mu = cavi.mu
        phi = cavi.phi
        sigma = cavi.sigma
        lambda_ = cavi.lambda_
        theta = cavi.theta
    
        # get (mean + max) rewards for evaluation
        mean_reward, max_reward, mean_jain = dec_pomdp.evaluate_policy(var.eval_episode, var.eval_T, 
                                                        delta, mu, theta, phi, sigma, lambda_)
        # record mean & max rewards
        mean_rewards[it + 1] = mean_reward
        max_rewards[it + 1] = max_reward
        
        # save mean & max reward arrays to file
        values = {"mean": mean_rewards, "max": max_rewards}
        io.savemat("learning_1.mat", values)


###############################################################################
# run threads for different exploration rate evolution curves
###############################################################################
# create threads
RL_thread_1 = threading.Thread(target = learningProcess_1)
    
# starting threads
RL_thread_1.start()
  
# wait until threads are completely executed
RL_thread_1.join()

