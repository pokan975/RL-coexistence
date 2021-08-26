# -*- coding: utf-8 -*-
"""
@author: Po-Kan (William) Shih
@advisor: Dr. Bahman Moraffah
Define the infinite-horizon dec-POMDP system
"""
import sys_parameter as var
import itertools as it
import numpy as np
from policy import behavior_policy2, learned_policy
from agent import LTE_agent, WiFi_agent



# =============================================================================
# declare unlicensed spectrum environment (POMDP)
# =============================================================================
class Spectrum:
    L = var.L
    W = var.W
    N = L + W
    
    def __init__(self):
        self.Z = 50   # size for initial policy
        self.O = 20   # max observation value
        self.transmission_rate = 30  # transmission of the spectrum
        
        # observation dict, each entry is {value: index}
        self.observation = dict(zip(range(self.O + 1), range(self.O + 1)))
        # create LTE & Wifi agent objects
        self.agents = []
        for l in range(self.L):
            self.agents.append(LTE_agent())
            
        for w in range(self.W):
            self.agents.append(WiFi_agent())
        
        # theoretical fair throughput
        self.fair_throughput = self.transmission_rate / self.N
        
        
    def reward_function(self, n):
        # get effective transmissin duration
        PL = self.agents[n].transmission_duration
        # if collision happens, subtract collision time
        if self.agents[n].collision_start > 0 and self.agents[n].collision_end > 0:
            # for LTE agent, comuted its clear duration
            if self.agents[n].agent_type == "LTE":
                PL -= (self.agents[n].collision_end - self.agents[n].collision_start)
            
            # for Wifi agent, the packet is lost if it is affected
            else:
                PL = 0
                
        # take floor to get the integer effective number of sub-frames
        PL = np.floor(PL)
        # turn effective duration into amount of data (kbits)
        PL *= self.transmission_rate
        
        # compute total duration used
        # D = self.agents[n].action_end - self.agents[n].action_start        
        D = self.agents[n].CCA_end - self.agents[n].action_start
        D = min(D, self.O)
        D += self.agents[n].transmission_duration
        
        # compute actual throughput & store to immediate throughput
        Thr = PL / D
        self.immediate_thr[n] = Thr
        
        # get latest throughputs of all agents
        last_throughputs = np.fromiter(map(lambda x: x[-1], self.throughputs), dtype = np.float)
        # replace n's throughput with the computed one
        last_throughputs[n] = Thr
        
        # compute the fair ratio for all agents
        x = last_throughputs / self.fair_throughput
        
        # compute the Jain's fairness indicator
        if np.sum(x) <= 0:
            jain = 0
        else:
            jain = np.sum(x)**2 / (self.N * np.sum(x**2))
            assert jain <= 1
            
        # record computed fairness value
        if self.phase == "evaluation":
            self.Jain_history.append(jain)
        
        # reweight actual throughput by Jain's fairness indicator
        reweighted_thr = Thr * jain
        #local immediate reward value
        immediate_reward = np.log(abs(reweighted_thr) + 1)
                
        # accumulate reward
        self.local_reward[n] += immediate_reward
        return self.local_reward[n]
    
    
    def collect_episodes(self, episode, T, epsilon, delta, mu, phi, sigma, lambda_):
        '''
        Parameters
        ----------
        episode : int
            number of episodes.
        T : int
            length of each episode.
        epsilon : float
            greedy exploration factor.
        phi : array of arrays or None
            previous learned phi parameters for action distribution.
        Returns
        -------
        None.
        '''
        # list of data collecting policies
        self.policies = []
        
        # generate policies for agents
        for i in range(self.N):
            # extract parameters learned in previous iteration
            # initial value is None
            policy_parameters = (delta[i], mu[i], phi[i], sigma[i], lambda_[i])
            # generate policy
            policy = behavior_policy2(self.O, self.Z, epsilon, policy_parameters)
            self.policies.append(policy)
                    
        self.phase = "learning"
        
        # collect trajectories
        self.interaction(episode, T)
        
        
    def evaluate_policy(self, episode, T, delta, mu, theta, phi, sigma, lambda_):
        # list of evaluation policies
        self.policies = []
        
        # generate evaluation policies from learned parameters in previous learning iteration
        for i in range(self.N):
            # extract parameters learned in previous iteration
            # initial value is None
            policy_parameters = (delta[i], mu[i], phi[i], theta[i], sigma[i], lambda_[i])
            # generate policy
            policy = learned_policy(policy_parameters)
            self.policies.append(policy)
            
        self.phase = "evaluation"
        
        # collect trajectories
        self.interaction(episode, T)
        # extract mean reward of each trajectory
        mean_rewards = np.fromiter(map(lambda r: r[0], self.episodes), dtype = np.float)
        # extract max reward of all trajectories
        max_rewards = np.fromiter(map(lambda r: r[1], self.episodes), dtype = np.float)
        # extract mean fairness vlue of each trajectory
        Jains = np.fromiter(map(lambda r: r[2], self.episodes), dtype = np.float)
       
        # compute & return the max of max rewards, means of mean rewards, and
        # mean fairness values in all trajectories
        return np.mean(mean_rewards), np.max(max_rewards), np.mean(Jains)
    
    
    
    def interaction(self, episode, T):
                
        # list of episodes
        self.episodes = []
                
        for ep in range(episode):
            # local rewards for all agent
            self.local_reward = np.zeros(self.N)
            # local throughput history (initial value = 0)
            self.throughputs = [np.zeros(1) for n in range(self.N)]
            self.immediate_thr = np.zeros(self.N)
            
            if self.phase == "evaluation":
                # Jain's indicator history
                self.Jain_history = []
            
            # action history array (-1: entry is none)
            act_history = -1 * np.ones((self.N, T), dtype = int)
            # obsveration history array (-1: entry is none)
            obv_history = -1 * np.ones((self.N, T), dtype = int)
            # global reward array
            global_reward = np.zeros(T, dtype = int)
            
            
            # system time
            sys_time = 0
            # episode length counter
            t = 0
            
            # initially pick action for each agent
            for n in range(self.N):
                # reset policy
                self.policies[n].reset_policy()
                # pick random action from policy
                act_index = self.policies[n].select_action()
                self.agents[n].assign_action(act_index, sys_time)
            
            # episode loop
            while t < T:
                
                # update spectrum usage by checking on spectrum flag of each agent
                occupants = map(lambda n: 1 if self.agents[n].onSpectrum_flag else 0, range(self.N))
                # spectrum occupancy (0: offline, 1: online)
                self.spectrum_usage = np.fromiter(occupants, dtype = int)
                
                # checl each agent's action status
                for n in range(self.N):
                    # agent has no active action, select a new one
                    if not self.agents[n].acting_flag:
                        # get effective previous action (index)
                        pre_act = self.agents[n].action_idx
                        # extract observation history up to t
                        obv_t = obv_history[n, :t]
                        # get effective previous observation (index)
                        pre_obv = obv_t[obv_t >= 0][-1]
                        # sample new action given previous action & observation
                        act_index = self.policies[n].select_action(pre_act, pre_obv)
                        
                        # assign action to agent
                        self.agents[n].assign_action(act_index, sys_time)
                        assert self.agents[n].acting_flag
                    
                    # perform action
                    self.agents[n].perform_action(self.spectrum_usage, sys_time)
                

                
                # extract indices of agents that need to compute rewards
                r_contributor = filter(lambda n: 0 if self.agents[n].acting_flag else 1, range(self.N))
                r_contributor = np.fromiter(r_contributor, dtype = int)
                
                # compute global reward & proceed episode
                if r_contributor.size > 0:
                    # compute local reward for each agent
                    reward = np.fromiter(map(self.reward_function, r_contributor), dtype = float)
                    # sum all local rewards to get 1 global reward & round it
                    reward = sum(reward) #round(sum(reward))
                    # bound reward value if needed
                    global_reward[t] = reward #max(min(40, reward), -40)
                    
                    
                    # record action index & observation
                    for n in r_contributor:
                        # append immediate throughputs to throughput history
                        self.throughputs[n] = np.append(self.throughputs[n], self.immediate_thr[n])
                        
                        # add action index to action history
                        act_history[n, t] = self.agents[n].action_idx
                        
                        # compute the backoff sensing time as observation value
                        # the actual sensing time = CCA_end - action_start + 1e-3
                        # (need to +1 clock back to get the most accurate value)
                        # but we will truncate D so it almost makes no difference
                        obv = self.agents[n].CCA_end - self.agents[n].action_start
                        assert obv > 0
                        
                        # truncate & bound observation value
                        obv = min(round(obv), self.O)
                        # add observation index to history given its dict value
                        obv_history[n, t] = self.observation[obv]
                        
                        
                    # episode proceeds whenever a global reward is computed
                    t += 1
                    
                
                # update system time (unit: minisec)
                # notice that the minimum time unit is 10 us
                sys_time = round(sys_time + 1e-2, 2)
                
            # for learning phase, record action, observation, reward histories
            if self.phase == "learning":
                self.episodes.append((act_history, obv_history, global_reward))
            
            # for evaluation phase, only record reward histories & fairness values
            else:
                # compute the mean of the fairness values
                mean_jain = np.mean(np.array(self.Jain_history))
                # get the max of global rewards of each episode
                max_reward = np.max(global_reward)
                # compute the mean of global rewards of each episode
                mean_reward = np.mean(global_reward)
                # append mean+max rewards & mean fairness value of each episode
                self.episodes.append((mean_reward, max_reward, mean_jain))
                
        
        
        