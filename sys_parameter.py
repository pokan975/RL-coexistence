# -*- coding: utf-8 -*-
"""
@author: Po-Kan (William) Shih
@advisor: Dr. Bahman Moraffah
put all predefined parameters for the problem setup here
"""

# =============================================================================
# global variables
# =============================================================================
# for learning phase
episode = 2
T = 50
# episode = 200  # number of episodes
# T = 150  # length of each episode

# for evaluation phase
eval_episode = 2
eval_T = 50
# eval_episode = 500
# eval_T = 1000

gamma = 0.9  # discount factor

L = 2   # number of LTE nodes
W = 2   # number of Wi-Fi nodes
N = L + W  # number of total agents


# =============================================================================
# predefined parameters
# =============================================================================
# duration of spectrum sensing techniques
ICCA = 5   # clocks for LTE ICCA duration
DIFS = 4   # clocks for Wi-Fi DIFS duration
ECCA = 1   # clocks for each backoff slot (minimum clock unit = 10 us)
# action set for LTE agents
actions_LTE = {0: (1, 3), 
               1: (3, 6), 
               2: (6, 6), 
               3: (12, 8), 
               4: (25, 8), 
               5: (51, 10), 
               6: (102, 10)}
# action set for Wi-Fi agents
actions_Wifi = {0: (1, 4), 
               1: (3, 4), 
               2: (6, 4), 
               3: (12, 4), 
               4: (25, 4), 
               5: (51, 4), 
               6: (102, 4)}
