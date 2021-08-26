# -*- coding: utf-8 -*-
"""
@author: Po-Kan (William) Shih
@advisor: Dr. Bahman Moraffah
Main code for simulation
"""
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from itertools import cycle

import sys_parameter as var
from environment import Spectrum
from var_inference import CAVI

plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10
# mark = cycle(["o","v","^","<",">","1","2","3","4","8","s","p","*","h","H","+","x","d","D","|","P","X"])
mark = ["o","v","^","<",">","1","2","3","4","8","s","p","*","h","H","+","x","d","D","|","P","X"]
color = 'rgbcmykw'
linecycler = cycle(["-","--","-.",":"])


iterations = 41  # number of learning iterations
# exploration factor
epsilon1 = np.linspace(0.9, 0.5, iterations)
epsilon2 = np.linspace(0.9, 0.2, iterations)


###############################################################################
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

# history of node cardinality for all agents through learning process
z_cardinality = []
# histories of mean & total rewards through learning process 
mean_rewards = np.zeros(iterations + 1)
max_rewards = np.zeros(iterations + 1)

# learning iteration
for it in range(2):
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
    
    # # get mean+max rewards for evaluation
    mean_reward, max_reward, mean_jain = dec_pomdp.evaluate_policy(var.eval_episode, var.eval_T, 
                                                        delta, mu, theta, phi, sigma, lambda_)
    mean_rewards[it] = mean_reward
    max_rewards[it] = max_reward


###############################################################################
# Plots for single VI iteration
###############################################################################
# plot ELBO curve
plt.figure()
# x = np.array(range(len(cavi.elbo_values[1:])))[::5]
# y = cavi.elbo_values[1:][::5]
plt.plot(cavi.elbo_values[1:], marker = "s", linewidth = 3)
plt.xlabel("Iteration")
plt.ylabel("Evidence lowerbound")
plt.show()


# # plot |Z| evolution
# plt.figure()
# z = np.array(cavi.card_history)
# for n in range(var.N):
#     # color = tuple(np.random.random_sample(3))
#     name = dec_pomdp.agents[n].agent_type
#     plt.plot(z[:,n], color = color[n], linewidth = 3 , linestyle = next(linecycler),
#               marker = mark[n], markersize = 7, label = name + r"$_{}$".format(n))

# plt.xlabel("Iteration")
# plt.ylabel("|Z|")
# plt.ylim(0, 10)
# plt.legend()
# # plt.legend(loc='center left', bbox_to_anchor = [1, 0], prop={'size': 8})
# plt.show()


# # plot discounted value evolution
# x = np.array(range(len(cavi.value)))[::5]
# plt.plot(x, cavi.value[::5], "-o", color = "r", markersize = 8, linewidth = 4)
# plt.xlabel("Iteration")
# plt.ylabel("Discounted value")
# plt.show()


# # parameter evolution
# x = np.array(range(len(cavi.g_history)))[::5]
# g_history = np.array(cavi.g_history)[::5, ...]
# for n in range(var.N):
#     # color = tuple(np.random.random_sample(3))
#     name = dec_pomdp.agents[n].agent_type
#     plt.plot(x, g_history[:,n], color = color[n], linewidth = 3 , linestyle = next(linecycler),
#               marker = mark[n], markersize = 7, label = name + r"$_{}$".format(n))

# plt.xlabel("Iteration")
# plt.ylabel(r"g evolution in Gamma($\rho$|g, h)")
# plt.legend()
# plt.show()

# h_history = np.array(cavi.h_history)[::5, ...]
# for n in range(var.N):
#     # color = tuple(np.random.random_sample(3))
#     name = dec_pomdp.agents[n].agent_type
#     plt.plot(x, h_history[:,n], color = color[n], linewidth = 3 , linestyle = next(linecycler),
#               marker = mark[n], markersize = 7, label = name + r"$_{}$".format(n))

# plt.xlabel("Iteration")
# plt.ylabel(r"h evolution in Gamma($\rho$|g, h)")
# plt.legend(bbox_to_anchor = [1, 0.6])
# # plt.legend(loc='center left', bbox_to_anchor = [1, 0], prop={'size': 8})
# plt.show()
    

###############################################################################
# Plots for entire learning process
###############################################################################
# # plot CDF of throughput
# for n, thr in enumerate(dec_pomdp.throughputs):
#     # get throughput histories & normalize them 
#     normed_thr = np.sort(thr[1: ] - np.min(thr[1: ]))
#     # compute CDF over exponential distribution with lambda = 0
#     cdf = st.expon(0).cdf(normed_thr)
#     color = tuple(np.random.random_sample(3))
#     name = dec_pomdp.agents[n].agent_type
#     plt.plot(normed_thr, cdf, "-o", color = color, label = name+r"$_{}$".format(n))


# plt.legend()
# plt.xlabel("Throughput (Mbps)")
# plt.ylabel("Probability")
# plt.ylim(0, 1)
# plt.grid()
# plt.show()


# # print mean Jain's fairness indicator
# print("Mean Jain's fairness indicator in final evaluation is", mean_jain)


# plt.plot(epsilon1, label = r"$\epsilon_1$")
# plt.plot(epsilon2, label = r"$\epsilon_2$")
# plt.xlabel("Learning iteration")
# plt.ylabel(r"$\epsilon$ value")
# plt.xlim(0, 50)
# plt.ylim(0, 1)
# plt.legend()
# plt.show()