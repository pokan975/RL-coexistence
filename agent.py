# -*- coding: utf-8 -*-
"""
@author: Po-Kan (William) Shih
@advisor: Dr. Bahman Moraffah
Agent objects are defined here
"""
import sys_parameter as var
import numpy as np
from random import randint


class WirelessNode:
    def __init__(self):
        # set the duration of 1 back-off sensing slot as ECCA
        self.backoff_slot = var.ECCA
        
        # clear in-action flag
        self.acting_flag = False
        # clear on-spectrum flag
        self.onSpectrum_flag = False
        # clear transmission downcunters
        self.transmission_timer = 0
        self.transmission_duration = 0
        # clear action index
        self.action_idx = -1
        
        # clear sensing downcounters
        self.ICCA_timer = 0
        self.ECCA_slots = 0
        # clear sensing phase flags
        self.ICCA_flag = False
        self.ECCA_flag = False
        # clear system time recordings
        self.CCA_end = -1
        self.collision_start = -1
        self.collision_end = -1
        
        # clear time records for duration computation
        self.action_start = -1
        self.action_end = -1
        
        
    
    def assign_action(self, index, sys_time):
        ## for internal use ###################################################
        # extract action tuple from action set
        # each action tuple: (CW, transmission time)
        act = self.action_set[index]
        
        # set ICCA timer to max value
        self.ICCA_timer = self.ICCA_duration
        
        # sample a random integer ~ [0, CW] for backoff slot counter
        # (now a multiple of back-off slot)
        self.ECCA_slots = randint(0, act[0])
        
        # set transmission timer (in minimum unit)
        self.transmission_timer = act[1] * 100
        
        # set ICCA sensing phase at start
        self.ICCA_flag = True
        # clear ECCA sensing flag
        self.ECCA_flag = False
        
        
        ## for system information #############################################
        # record action index
        self.action_idx = index
        # set in-action flag to indicate agent is in action
        # when action is finished, clear this flag to signal the system to 
        # record action index & reward
        self.acting_flag = True
        # not using spectrum at start
        self.onSpectrum_flag = False
        # backup the total transmission time for computing reward
        self.transmission_duration = act[1]
        # record the system time of spectrum sensing end
        self.CCA_end = -1
        # clear the record for collision start & end time
        self.collision_start = -1
        self.collision_end = -1
        # record the start time of action duration & spectrum sensing
        self.action_start = sys_time
        # clear action end time record
        self.action_end = -1
        
    
    
    def perform_action(self, spectrum_usage, sys_time):
        # spectrum_usage is a numpy array where each element is 0 or 1, indicating 
        # if agent n is using spectrum or not
        # sys_time is the current system clock for agent to compute its access duration
        if self.ICCA_flag:
            self.ICCA_sensing(spectrum_usage)
        elif self.ECCA_flag:
            self.ECCA_sensing(spectrum_usage, sys_time)
        else:
            self.transmitting(spectrum_usage, sys_time)
        
                
                
    def ICCA_sensing(self, spectrum_usage):
        # if spectrum clear & ICCA not finished, down count ICCA timer
        if np.sum(spectrum_usage) == 0 and self.ICCA_timer > 0:
            self.ICCA_timer -= 1
            
        # finish ICCA sensing and go to ECCA sensing, clear & set flags 
        if self.ICCA_timer == 0:
            self.ICCA_flag = False
            self.ECCA_flag = True
    
    
    def ECCA_sensing(self, spectrum_usage, sys_time):
        
        # ECCA slots > 0, count down backoff slot
        if self.ECCA_slots > 0:
            # for each 1-clock slot, check if there are spectrum occupants
            # if spectrum is clear, slot counter - 1
            if np.sum(spectrum_usage) == 0:
                self.ECCA_slots -= 1
            else:
                # for every occupant, flip a coin to see if it can be detected
                detected_users = np.random.default_rng().binomial(1, 0.995, np.sum(spectrum_usage))
            
                # if no occupant detected, slot counter - 1
                if np.sum(detected_users) == 0:
                    self.ECCA_slots -= 1
                
                
        # ECCA slots reach 0, record system time for computing sensing duration
        # clear flag, set transmission flag
        if self.ECCA_slots == 0:
            self.CCA_end = sys_time
            self.ECCA_flag = False
            self.onSpectrum_flag = True
        
    
    def transmitting(self, spectrum_usage, sys_time):
        
        # if transmission is not finished yet, count down timer
        if self.transmission_timer > 0:
            self.onSpectrum_flag = True
            self.transmission_timer -= 1
            
            # if collision appears & the start time yet to be recorded, record it
            if np.sum(spectrum_usage) > 1 and self.collision_start < 0:
                self.collision_start = sys_time
                
            # if collision disappears during transmission, record it
            elif np.sum(spectrum_usage) < 2 and self.collision_start > 0:
                self.collision_end = sys_time
        
        # transmission ends, leave spectrum
        if self.transmission_timer == 0:
            # if collision stays until transmission ends, set the collision end
            # time as transmission end time
            if np.sum(spectrum_usage) > 1 and self.collision_start > 0:
                self.collision_end = sys_time
            
            # clear on-spectrum flag
            self.onSpectrum_flag = False
            # clear action flag
            self.acting_flag = False
            # record action ending time
            self.action_end = sys_time
        
        
# =============================================================================
# declare LTE agent
# =============================================================================
class LTE_agent(WirelessNode):
    def __init__(self):
        WirelessNode.__init__(self)
        self.ICCA_duration = var.ICCA        
        self.action_set = var.actions_LTE
        self.agent_type = "LTE"
        


# =============================================================================
# declare Wi-Fi agent
# =============================================================================
class WiFi_agent(WirelessNode):
    def __init__(self):
        WirelessNode.__init__(self)
        self.ICCA_duration = var.DIFS
        self.action_set = var.actions_Wifi
        self.agent_type = "Wi-Fi"