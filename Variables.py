# -*- coding: utf-8 -*-
"""
@author: Po-Kan (William) Shih
@advisor: Dr. Bahman Moraffah
put all initial variables & predefined parameters here
"""

# =============================================================================
# global variables
# =============================================================================
K = 10  # number of episodes
L = 2   # number of LTE nodes
W = 2   # number of Wi-Fi nodes
N = L + W  # number of total agents
channels = [1] # number of total channels

# length of each episode
T = 100

# =============================================================================
# predefined parameters
# =============================================================================
# duration of spectrum sensing techniques (unit: ms)
# for POMDP_2
CCA = 20*1e-3    # LTE CCA
DIFS = 34*1e-3   # Wi-Fi DIFS
Wifi_BO = 9*1e-3 # Wi-Fi backoff sensing
# contention window size
CW = [2, 4, 8]
# duration of channel occupancy time (unit: ms)
burst = [1, 3]
# duration of idle time (unit: ms)
idle_duration = [1]

C1 = 1   # weight of backoff duration
C2 = 2   # penalty if backoff time > Th
C3 = 5   # weight of penalty for collision
Th = 8   # threshold of "too long" backoff duration

# discount factor
gamma = 0.9