# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 11:33:25 2020

@author: William
"""
# -*- coding: utf-8 -*-
"""
@author: Po-Kan (William) Shih
@advisor: Dr. Bahman Moraffah
In this file we define the infinite-horizon POMDP model
"""
import Variables as var
import itertools as it
import numpy as np
from SB_prior import FSC_policy, uniform_policy


class POMDP(object):
    
    def __init__(self):
        self.ch = var.channels  # channel list
        self.agentL = list(range(var.L))  # LTE agent list
        self.agentW = list(range(var.W))  # WiFi agent list
        
        # build actions for each type of agent
        self.initialize()
        
    def initialize(self):
        '''
        Returns
        -------
        Initialize action lists for LTE and WiFi agents respectively, each action
        set is a dict, in which each action is an entry {key: value}, key = index
        and value = action.
        '''
        # build each action tuple: (ch#, slots, transmission time)
        # if ch# not 0 -> agent in active, slots = # of backoff slots
        # if ch# = 0 -> agent in idle, slots = 0, transmission time = idle time (ms)
        idle_time = list(it.product([0], [0], var.idle_duration))
        
        # generate action tuples for LTE
        active_lte = list(it.product(self.ch, var.CW_L, var.LTE_burst))
        action = idle_time + active_lte
        self.action_lte = dict(zip(range(len(action)), action))
        
        # generate action tuples for WiFi
        active_wifi = list(it.product(self.ch, var.CW_W, var.Wifi_packet))
        action = idle_time + active_wifi
        self.action_wifi = dict(zip(range(len(action)), action))
        
        # policy list for each agent
        self.policies = []
        
        # generate initial policies for LTE agents
        for i in self.agentL:
            policy = uniform_policy(self.action_lte)
            self.policies.append(policy)
            
        # generate initial policies for WiFi agents
        for i in self.agentW:
            policy = uniform_policy(self.action_wifi)
            self.policies.append(policy)
            
        # list of unique (act, obv, reward) tuple for each agent
        # used to compute truncation level
        self.unique_a_o_r = [set() for i in range(var.N)]
        # initialize global observation set
        self.observation = set()
        

    def interaction(self, epi, T):
        '''
        Parameters
        ----------
        epi : int
            number of episodes.
        T : int
            length of each episode.
        Returns
        -------
        perform agent-env interactions, collect trajectories and histories.
        '''
        
        # initialize list of histories
        self.histories = []
        # initialize list of episodes
        self.episodes = []
        
        
        for ep in range(epi):
            # initialize local reward tracker for each agent
            self.local_rwd = np.zeros(var.N)
            # initialize state indicator for each agent
            # in-action bit | ch# | sensing time (us) | transmission time (us) |
            # transmission time (fixed) | action index
            self.agents = np.zeros((var.N, 6), dtype = np.int)
            # initialize timing recorder for computing actual sensing duration
            # sensing start | sensing end | sensing start bit | transmission start bit
            # transmission start bit used as flag for computating local reward
            self.agentTimer = np.zeros((var.N, 4), dtype = np.int)
            # initialize channel occupancy indicator for each channel
            self.ch_occupancy = np.zeros(len(self.ch) + 1, dtype = np.int)
            # initialize action history array (-1 means that entry is none)
            act_history = -1 * np.ones((var.N, T))
            # initialize obsveration history array (-1 means that entry is none)
            obv_history = -1 * np.ones((var.N, T))
            # global reward array
            global_reward = np.zeros(T)
        
            # global timer for all channels
            self.timer = 0
            # episode length counter
            self.t = 0
            
            # assign initial action for each agent
            for n in range(var.N):
                # pick an action from policy
                act_index = self.policies[n].select_action()
                self.assign_action(n, act_index)
            
            # episode collection loop
            while self.t < T:
                # if agent has no running action, pick a new one
                for n in np.where(self.agents[:, 0] == 0)[0]:
                    act_index = self.policies[n].select_action()
                    # fill in action info
                    self.assign_action(n, act_index)
                
                # agents run actions
                for n in range(var.N):
                    self.action(n)

                
                # extract indices of agents that need to compute rewards
                comp_rwd = np.where(self.agentTimer[:, 3] == 1)[0]
                # if some agents compute their rewards, episode proceeds
                if comp_rwd.size != 0:
                    # compute golbal rewards
                    global_reward[self.t] = sum(map(self.reward, comp_rwd))
                    
                    for nn in comp_rwd:
                        # add action index to action history
                        act_history[nn, self.t] = self.agents[nn, -1]
                        # add observation to action history
                        obv_history[nn, self.t] = self.agentTimer[nn, 1] - self.agentTimer[nn, 0]
                            
                        # if it is new observation, add to observation set
                        # used to compute size of observation set
                        self.observation.add(obv_history[nn, self.t])
                        
                        # check (act, obv, reward) is unique tuple for agent n,
                        # add it to unique set, this set will be used to compute
                        # the truncation level for agent n
                        a_o_r = (act_history[nn, self.t], obv_history[nn, self.t], global_reward[self.t])
                        self.unique_a_o_r[nn].add(a_o_r)
                        
                    # episode proceeds
                    self.t += 1
                    
                
                # update global timer (unit: us)
                self.timer += 1
            
            
            self.episodes.append((act_history, obv_history, global_reward))
            self.histories.append((act_history, obv_history))
    
    
    def action(self, index):
        '''
        Parameters
        ----------
        index : int
            agent index.
        Returns
        -------
        None.
        '''
        # extract accessing channel #
        ch_access = self.agents[index, 1]
        
        # channel # not 0, agent in active mode
        if self.agents[index, 1] != 0:
            
            assert self.agents[index, 2] >= 0
            # in sensing phase 
            if self.agents[index, 2] > 0:
                
                # check if it is the 1st sec in sensing phase
                if self.agentTimer[index, 2] == 1:
                    # record the start time of sensing phase
                    self.agentTimer[index, 0] = self.timer
                
                # clear sensing start bit
                self.agentTimer[index, 2] = 0
                
                assert self.ch_occupancy[ch_access] >= 0
                # if channel is sensed idle, down count sensing duration
                if self.ch_occupancy[ch_access] == 0:
                    self.agents[index, 2] -= 1
            
            # in transmission phase
            else:
                assert self.agents[index, 3] >= 0
                if self.agents[index, 3] > 0:
                    # check if it is the 1st sec in transmission phase
                    if self.agents[index, 3] == self.agents[index, 4]:
                        # record the end time of sensing phase
                        self.agentTimer[index, 1] = self.timer
                        # start using channel
                        self.ch_occupancy[ch_access] += 1
                        # set transmission start bit
                        self.agentTimer[index, 3] = 1
                        
                    else:
                        # clear transmission start bit
                        self.agentTimer[index, 3] = 0
                
                    # down count transmission duration
                    self.agents[index, 3] -= 1
                # transmission end
                else:
                    assert self.ch_occupancy[ch_access] >= 0
                    # leave channel
                    self.ch_occupancy[ch_access] -= 1
                    # reset in-action bit as finished
                    self.agents[index, 0] = 0
                
                
        # agent in idle mode
        else:
            
            assert self.agents[index, 3] >= 0
            if self.agents[index, 3] > 0:
                # check if it is the 1st sec of idle duration
                if self.agents[index, 3] == self.agents[index, 4]:
                    # set idle start bit
                    self.agentTimer[index, 3] = 1
                else:
                    # clear idle start bit
                    self.agentTimer[index, 3] = 0
                    
                # down count idle duration
                self.agents[index, 3] -= 1
                    
            else:
                # reset in-action bit as finished
                self.agents[index, 0] = 0
            
    
    def reward(self, n):
        '''
        Parameters
        ----------
        n : int
            agent index.
        Returns
        -------
        int
            conclude the local reward for agent n.
        '''
        #initialize reward value
        rwd = 0
        
        # extract channel # of user
        ch_access = self.agents[n, 1]
        assert self.ch_occupancy[ch_access] >= 0
        
        # compute backoff duration
        t_bo = self.agentTimer[n, 1] - self.agentTimer[n, 0]
        t_bo *= 1e-3
        
        # agent in active
        if ch_access != 0:
            # constant penalty if backoff time > threshold
            rwd -= var.C2 * int(t_bo >= var.Th)
            
            # no collision
            if self.ch_occupancy[ch_access] == 1:
                t_diff = self.agents[n, 4] * 1e-3 - (var.C1 * t_bo)
                rwd += np.sign(t_diff) * np.log(abs(t_diff) + 1)
            
            # collision happens
            else:
                # extract colliding agents indices
                col_agents = np.where(self.agents[:, 1] == ch_access)[0]
                # extract their transmission durations
                t_col = sum(self.agents[:, 4][col_agents])
                t_col *= 1e-3
                rwd -= var.C3 * np.log(t_col + 1)
                
                
        # agent in idle
        else:
            # get idle duration length
            t_off = self.agents[n, 4] * 1e-3
            rwd -= np.log(t_off + 1)
        
        # accumulate reward
        self.local_rwd[n] = round(self.local_rwd[n] + rwd, 2)
        return self.local_rwd[n]


    def assign_action(self, n, action_index):
        '''
        Parameters
        ----------
        n : int
            agent index.
        action_index : int
            action index picked by FSC policy.
        Returns
        -------
        fill in action info for agent n by action index.
        '''
        # store action index
        self.agents[n, -1] = action_index
        
        # if it is LTE agent
        if n < var.L:
            # extract action tuple by index
            act = self.action_lte[action_index]
        # if it is WiFi agent
        else:
            act = self.action_wifi[action_index]
        
        # set in-action bit
        self.agents[n, 0] = 1
        # set channel #
        self.agents[n, 1] = act[0]
        
        # action sets agent active
        if act[0] != 0:
            # for LTE agants
            if n < var.L:
                # compute required sensing time (ms)
                self.agents[n, 2] = var.CCA * (act[1] + 1)
            # for WiFi agants
            else:
                # compute required sensing time (ms)
                self.agents[n, 2] = var.DIFS + (var.Wifi_BO * act[1])
                
        # action sets agent idle, no sensing time (= 0)
        else:
            self.agents[n, 2] = 0
                    
        # set transmission/idle time (ms)
        self.agents[n, 3] = act[2] * 1e3
        # copy transmission time for start indicator
        self.agents[n, 4] = act[2] * 1e3
        
        # set sensing start bit
        self.agentTimer[n, 2] = 1
        # clear transmission start bit
        self.agentTimer[n, 3] = 0
