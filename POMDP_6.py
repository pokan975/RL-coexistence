# -*- coding: utf-8 -*-
"""
@author: Po-Kan (William) Shih
@advisor: Dr. Bahman Moraffah
In this file we define the infinite-horizon POMDP model
"""
import Variables as var
import itertools as it
import numpy as np
import pandas as pd
from SB_prior import behavior_policy2, eval_policy



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
        self.O = 20
        # build each action tuple: (ch#, slots, transmission time)
        # if ch# not 0 -> agent in active, slots = # of backoff slots
        # if ch# = 0 -> agent in idle, slots = 0, transmission time = idle time (ms)
        idle_time = list(it.product([0], [0], var.idle_duration))
        
        # generate action tuples
        active_tup = list(it.product(self.ch, var.CW, var.burst))
        action = idle_time + active_tup
        # key: agent index, value: action tuple
        self.action_set = dict(zip(range(len(action)), action))
                
        # list of unique (act, obv, reward) tuple for each agent
        # used to compute truncation level
        self.unique_a_o_r = [set() for i in range(var.N)]
        # initialize obv dataframe 
        c = {'value': -1 * np.ones(self.O)}
        self.observation = pd.DataFrame(c)
        
           
    def collect_episode(self, epi, T, Z, epsilon, phi, sigma, lambda_):
        '''
        Parameters
        ----------
        epi : int
            number of episodes.
        T : int
            length of each episode.
        Z : int
            truncated number of nodes.
        epsilon : float
            greedy exploration factor.
        phi : array of arrays or None
            previous learned phi parameters for action distribution.
        Returns
        -------
        None.
        '''
        self.Z = Z
        # policy list for each agent
        self.policies = []
        # generate initial policies for LTE agents
        for i in range(var.N):
            policy = behavior_policy2(len(self.action_set), self.O, Z, epsilon, phi[i], sigma[i], lambda_[i])
            # policy = behavior_policy(len(self.action_set), len(self.observation), Z, epsilon)
            self.policies.append(policy)
                    
        # collect
        self.interaction(epi, T)
        

    def evaluate_policy(self, epi, T, theta, phi, sigma, lambda_):
        self.policies = []
        # generate initial policies for LTE agents
        for n in range(var.N):
            policy = eval_policy(theta[n], phi[n], sigma[n], lambda_[n])
            self.policies.append(policy)
            
        rewards = np.zeros(epi)
        for ep in range(epi):
            rewards[ep] = np.mean(self.eval_interact(T))
        
        return np.mean(rewards)


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
            self.agents = np.zeros((var.N, 6), dtype = object)
            # initialize timing recorder for computing actual sensing duration
            # sensing start | sensing end | sensing start bit | transmission start bit
            # transmission start bit used as flag for computating local reward
            self.agentTimer = np.zeros((var.N, 4))
            # initialize channel occupancy indicator for each channel
            self.ch_occupancy = np.zeros(len(self.ch) + 1, dtype = np.int)
            # initialize action history array (-1 means that entry is none)
            act_history = -1000 * np.ones((var.N, T), dtype = np.int)
            # initialize obsveration history array (-1 means that entry is none)
            obv_history = -1000 * np.ones((var.N, T), dtype = np.int)
            # global reward array
            global_reward = np.zeros(T, dtype = np.int)
        
            # global timer for all channels
            self.timer = 0
            # episode length counter
            counter = 0
            
            # assign initial action for each agent
            for n in range(var.N):
                self.policies[n].refresh_prob()
                # pick random action from policy
                act_index = self.policies[n].select_action()
                self.assign_action(n, act_index)
            
            # episode collection loop
            while counter < T:
                # if agent has no running action, pick a new one
                for n in np.where(self.agents[:, 0] == 0)[0]:
                    # find previous action
                    temp_a = act_history[n, :counter]
                    pre_act = temp_a[temp_a >= 0][-1]
                    # find previous observation
                    temp_o = obv_history[n, :counter]
                    pre_obv = temp_o[temp_o >= 0][-1]
                    
                    # find current node given previous node, action, observation
                    act_index = self.policies[n].select_action(pre_act, pre_obv)
                    # fill in action info
                    self.assign_action(n, act_index)
                
                # agents run actions
                for n in range(var.N):
                    self.action(n)

                
                # extract indices of agents that need to compute rewards
                comp_rwd = np.where(self.agentTimer[:, 3] == 1)[0]
                # if some agents compute their rewards, episode proceeds
                if comp_rwd.size != 0:
                    # compute golbal rewards (round to reduce possible number of values)
                    rr = round(sum(map(self.reward, comp_rwd)))
                    # bound reward value
                    global_reward[counter] = rr#max(min(40, rr), -40)
                    
                    
                    for nn in comp_rwd:
                        # add action index to action history
                        act_history[nn, counter] = self.agents[nn, -1]
                        # add observation to action history
                        obv_ = round(self.agentTimer[nn, 1] - self.agentTimer[nn, 0])
                        # bound observation value
                        obv_ = min(obv_, 20)
                        
                        # if it is new observation, add to observation set
                        # used to compute size of observation set
                        if self.observation.loc[self.observation['value'] == obv_].empty:
                            l = self.observation.loc[self.observation['value'] != -1]
                            self.observation.loc[l.size, 'value'] = obv_
                        
                        obv_history[nn, counter] = self.observation.index[self.observation['value'] == obv_][0]
                        
                        # check (act, obv, reward) is unique tuple for agent n,
                        # add it to unique set, this set will be used to compute
                        # the truncation level for agent n
                        a_o_r = (obv_history[nn, counter], global_reward[counter])
                        self.unique_a_o_r[nn].add(a_o_r)
                        assert len(self.unique_a_o_r[nn]) < 200
                        
                    # episode proceeds
                    counter += 1
                    
                
                # update global timer (unit: us)
                self.timer = round(self.timer + 1e-3, 3)
                
            
            self.episodes.append((act_history, obv_history, global_reward))
            self.histories.append((act_history, obv_history))
    
    
    def eval_interact(self, T):
        '''
        Parameters
        ----------
        T : int
            length of each episode.
        Returns
        -------
        interactions for performance evaluation, using learned policy.
        '''
        
        # initialize local reward tracker for each agent
        self.local_rwd = np.zeros(var.N)
        # initialize state indicator for each agent
        # in-action bit | ch# | sensing time (us) | transmission time (us) |
        # transmission time (fixed) | action index
        self.agents = np.zeros((var.N, 6), dtype = object)
        # initialize timing recorder for computing actual sensing duration
        # sensing start | sensing end | sensing start bit | transmission start bit
        # transmission start bit used as flag for computating local reward
        self.agentTimer = np.zeros((var.N, 4))
        # initialize channel occupancy indicator for each channel
        self.ch_occupancy = np.zeros(len(self.ch) + 1, dtype = np.int)
        # initialize action history array (-1 means that entry is none)
        act_history = np.zeros(var.N, dtype = np.int)
        # initialize obsveration history array (-1 means that entry is none)
        obv_history = np.zeros(var.N, dtype = np.int)
        # initialize node history list
        node_history = np.zeros(var.N, dtype = np.int)
        # global reward array
        global_reward = np.zeros(T, dtype = np.int)
        
        # global timer for all channels
        self.timer = 0
        # episode length counter
        counter = 0
        
        # assign initial action for each agent
        for n in range(var.N):
            # initial node at 0
            node_history[n] = 0
            # pick random action from policy
            act_index = self.policies[n].select_action(0)
            self.assign_action(n, act_index)
            
        # episode collection loop
        while counter < T:
            # if agent has no running action, pick a new one
            for n in np.where(self.agents[:, 0] == 0)[0]:
                # find previous node
                pre_node = node_history[n]
                # find previous action
                pre_act = act_history[n]
                # find previous observation
                pre_obv = obv_history[n]
                    
                # find current node given previous node, action, observation
                node_index = self.policies[n].next_node(pre_node, pre_act, pre_obv)
                node_history[n] = node_index
                act_index = self.policies[n].select_action(node_index)
                # fill in action info
                self.assign_action(n, act_index)
            
            # agents run actions
            for n in range(var.N):
                self.action(n)
                
            # extract indices of agents that need to compute rewards
            comp_rwd = np.where(self.agentTimer[:, 3] == 1)[0]
            # if some agents compute their rewards, episode proceeds
            if comp_rwd.size != 0:
                # compute golbal rewards (round to reduce possible number of values)
                rr = round(sum(map(self.reward, comp_rwd)))
                # bound reward value
                global_reward[counter] = rr#max(min(40, rr), -40)
                
                for nn in comp_rwd:
                    # add action index to action history
                    act_history[nn] = self.agents[nn, -1]
                    # add observation to action history
                    obv_ = round(self.agentTimer[nn, 1] - self.agentTimer[nn, 0])
                    # bound observation value
                    obv_ = min(obv_, 20)
                    obv_history[nn] = self.observation.index[self.observation['value'] == obv_][0]
                    
                
                # episode proceeds
                counter += 1
                
            # update global timer (unit: us)
            self.timer = round(self.timer + 1e-3, 3)
            
        return global_reward
    
    
    
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
        assert self.ch_occupancy[ch_access] >= 0
        
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
                
                # if channel is sensed idle, down count sensing duration
                if self.ch_occupancy[ch_access] == 0:
                    self.agents[index, 2] -= 1e-3
                    self.agents[index, 2] = round(self.agents[index, 2], 3)
            
            # in transmission phase
            else:
                assert self.agents[index, 3] >= 0
                # transmission not finished yet
                if self.agents[index, 3] > 0:
                    # check if it is the 1st sec in transmission phase
                    if self.agents[index, 3] >= self.agents[index, 4]:
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
                    self.agents[index, 3] -= 1e-3
                    self.agents[index, 3] = round(self.agents[index, 3], 3)
                # transmission end
                else:
                    # leave channel
                    self.ch_occupancy[ch_access] -= 1
                    assert self.ch_occupancy[ch_access] >= 0
                    # reset in-action bit as finished
                    self.agents[index, 0] = 0
                
                
        # agent in idle mode
        else:
            
            assert self.agents[index, 3] >= 0
            if self.agents[index, 3] > 0:
                # check if it is the 1st sec of idle duration
                if self.agents[index, 3] >= self.agents[index, 4]:
                    # set idle start bit
                    self.agentTimer[index, 3] = 1
                else:
                    # clear idle start bit
                    self.agentTimer[index, 3] = 0
                    
                # down count idle duration
                self.agents[index, 3] -= 1e-3
                self.agents[index, 3] = round(self.agents[index, 3], 3)
                    
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
        
        # agent in active
        if ch_access != 0:
            # constant penalty if backoff time > threshold
            rwd -= var.C2 * int(t_bo >= var.Th)
            
            # no collision
            if self.ch_occupancy[ch_access] == 1:
                act_agents = np.where(self.agents[:, 1] == ch_access)[0]
                t_diff = var.N * self.agents[n, 4] - (var.C1 * t_bo)
                rwd += np.sign(t_diff) * np.log(abs(t_diff) + 1)
            
            # collision happens
            else:
                # extract colliding agents indices
                col_agents = np.where((self.agents[:, 1] == ch_access) & (self.agents[:, 2] <= 0))[0]
                # extract their transmission durations
                t_col = sum(self.agents[:, 4][col_agents])
                rwd -= var.C3 * np.log(t_col + 1)
                
                
        # agent in idle
        else:
            # get idle duration length
            t_off = self.agents[n, 4]
            rwd -= 0#np.log(t_off + 1)
        
        # accumulate reward
        self.local_rwd[n] += rwd
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
        
        # find action tuple
        act = self.action_set[action_index]
        
        # set in-action bit
        self.agents[n, 0] = 1
        # set channel #
        self.agents[n, 1] = int(act[0])
        
        # action sets agent active
        if act[0] != 0:
            # for LTE agants
            if n < var.L:
                # compute required sensing time (ms)
                sensing_time = var.CCA * (act[1] + 1)
            # for WiFi agants
            else:
                # compute required sensing time (ms)
                sensing_time = var.DIFS + (var.Wifi_BO * act[1])
                
        # action sets agent idle, no sensing time (= 0)
        else:
            sensing_time = 0
                    
        self.agents[n, 2] = round(sensing_time, 3)
        
        # set transmission/idle time (ms)
        self.agents[n, 3] = act[2]
        # copy transmission time for start indicator
        self.agents[n, 4] = act[2]
        
        # set sensing start bit
        self.agentTimer[n, 2] = 1
        # clear transmission start bit
        self.agentTimer[n, 3] = 0
        