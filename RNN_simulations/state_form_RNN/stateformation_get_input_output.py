#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Loads one participant's trial sequence and turns it into network inputs and targets.
# Used by RNN_simulation.py.
#
# Author: Shany Grossman

import numpy as np
import pandas as pd
#%%
def get_input_output_stateformation(sub,csvpath,isRecurrent,withPassive):
    # Loads one participant's day-1 trial sequence (active + passive/estimation trials) from
    # State_Formation_behaviour.csv and returns it in a form ready to feed an RNN.
    #
    # INPUTS:
    #   sub:          a real participant, e.g. 'sub508' (see the full subject list below), or
    #                 'randomized_episode' to instead return a shuffled sequence of all 192
    #                 unique task inputs (used for probing/analysis, not real subject data).
    #   csvpath:      path to the directory containing State_Formation_behaviour.csv.
    #   isRecurrent:  True  -> each input is a 2x16 array (row 0 = client/context nodes,
    #                          row 1 = house nodes), fed to the network across 2 time steps.
    #                 False -> each input is a single 1x16 array (client + house pair presented
    #                          simultaneously).
    #   withPassive:  True  -> also include passive/estimation trials (single house, inserted
    #                          before/after each active learning segment).
    #                 False -> active (learning) trials only.
    #                 For passive trials, the single house is encoded in the LEFT house position
    #                 (inputs[i,1,4:10]); the right house position is all zeros. outputs[i,0]
    #                 holds that house's value, outputs[i,1] is NaN.
    #
    # OUTPUTS (9 values):
    #   inputs:             [n_trials x 2 x 16] (or [n_trials x 16] if isRecurrent=False)
    #   outputs:            [n_trials x 2] reward/value of the left [:,0] and right [:,1] house
    #   isPassive:          [n_trials] 0 = active trial, 1 = passive/estimation trial
    #   states:             [n_trials] state code string per trial, e.g. 'a1a2' (active: context a,
    #                       house type 1 on left / house type 2 on right) or 'b2_l' (passive:
    #                       context b, house type 2, presented on the left by convention).
    #   block:              [n_trials] block_type_num (0-11 for day 1). 11 blocks, 42 trials each,
    #                       EXCEPT sub501/sub502 which have 40 trials/block (different
    #                       pre/post-passive windowing, handled internally below) and sub520,
    #                       which is genuinely missing block 6 in the source data (this function
    #                       does not special-case it; it just returns however many trials
    #                       exist; the downstream co-update analysis skips the missing window
    #                       for sub520, which has one fewer block).
    #   passive_type:       [n_trials] which segment a passive trial belongs to: pre_early /
    #                       post_early / pre_late / post_late. A block is structured as:
    #                       4 passive (pre_early) -> 4 active -> 4 passive (post_early) ->
    #                       16 active -> 5 passive (pre_late) -> 4 active -> 5 passive (post_late).
    #                       pre_late/post_late have 5 (not 4) passive trials because one of the 5
    #                       states also appears in the active trials in between (added as of
    #                       sub503, to get a within-block ceiling measure of representational
    #                       change).
    #   responses:          [n_trials] 'left'/'right' (active), 'more'/'less' (passive), or
    #                       'none' if no response was made in time.
    #   correct_responses:  [n_trials] the correct response, same coding as responses.
    #   probed_thresholds:  [n_trials] the % threshold value presented on passive/estimation
    #                       trials (e.g. "is this house worth more or less than 50%?").
    #
    # Data-quality notes (sub542, sub544, sub553, sub554 needed a partial+full block manually
    # stitched together; already done in the current CSV):
    #   sub542: stitch 2_2 (active8) partial with the one that fully ran
    #   sub544: stitch 1_3 (active2) partial with full
    #   sub553: stitch 1_5 (active4) partial with full (take the pre_early from the partial)
    #   sub554: stitch 1_7 (active6) partial with full (take pre_early + post_early from partial)
    #
    # Full day-1 subject list:
    #   sub501-sub564 (excluding sub-numbers not used in the study), 64 subjects total.
    #
    # Related functions in this module:
    #   get_active_trials_input_comb: builds the 192 unique inputs for sub='randomized_episode'
    #   rewardfun:                    computes the reward/value for a given input + choice
    #   input2state / state code dict: converts between a 10-bit input array and its state
    #                                    code string (e.g. 'a1a2'), and back

    # Some hardcode params of the task:
    n_feat_itemA=2 # 2 binary client features (male/female, glasses/no glasses)
    n_feat_itemB=3 # 3 binary house features (forest/mountain, swing/pool, )
    n_actions=2 # 2 choice alternative  - left house / right house
    n_input_nodes = 16 # 16 inputs node - each 2 nodes code a single binary featue, 8 features altogether (client + 2 houses)
    reward_weights = np.array([-1,2,0,0,-1,1,-3,3,0,0]) #weights of each feature - client f1, client f2, house f1, house f2, house f3   
    if sub=='randomized_episode':
        inputs = get_active_trials_input_comb(n_input_nodes,n_actions,n_feat_itemA) #inputs: 192X16 array (need to break into 2 time points when passed forward)
        maxtrials = inputs.shape[0]
        assert(maxtrials==192)
        
    else:
        # Load CSV summerizing behaviour of all participants
        log_df = pd.read_csv(csvpath + 'State_Formation_behaviour.csv')


        if withPassive:
            # Filter active and assive trials in day 1 of selected subject 
            csub_log = log_df[(log_df['sub']==sub) & (log_df['block_type_num']<12)]
            
        else:
            # Filter active trials in day 1 of selected subject
            csub_log = log_df[(log_df['sub']==sub) & (log_df['is_passive']==0) & (log_df['block_type_num']<12)]
            
        # Convert inputs from strings to numpy array
        tmp = np.array(csub_log.input)
        inputs = np.array([np.fromstring(tmp[i][1:-1], dtype=int, sep=' ') for i in range(len(tmp))])
    
    # Break input into 2 time point (2X16 instead of 1X16), and compute reward (i.e. target output):
    n_trials = inputs.shape[0]
    outputs = np.zeros((n_trials,2)) #trials X outputnodes
    states = np.empty(n_trials, dtype=object)
    passive_type = np.empty(n_trials, dtype=object)
    responses = np.empty(n_trials, dtype=object)
    correct_responses = np.empty(n_trials, dtype=object)
    probed_thresholds = np.empty(n_trials, dtype=object)
    block = np.zeros((n_trials))
    block[:] = np.nan
    recurrent_inputs = np.zeros((n_trials,2,16)) # trials X time points X input nodes 
    outputs[:],recurrent_inputs[:], states[:] = np.nan, np.nan, np.nan
    isPassive = np.zeros(n_trials)
    for itrial in range(n_trials):
        recurrent_inputs[itrial,0,:]=np.hstack((inputs[itrial][:4],np.zeros(12)))
        recurrent_inputs[itrial,1,:]=np.hstack((np.zeros(4),inputs[itrial][4:]))
        cinputs = recurrent_inputs[itrial,:,:]
        block[itrial] = csub_log.iloc[itrial].block_type_num
        responses[itrial] = csub_log.iloc[itrial].response
        correct_responses[itrial] = csub_log.iloc[itrial].correct_response
        probed_thresholds[itrial] = csub_log.iloc[itrial].single_house_prob_value


        itrial_inblock = np.sum(block==block[itrial])-1


        # Compute reward (i.e, Netwokks tragt output for left and right output nodes)
        if np.all(cinputs[1,10:]==0): # passive trial - single house is presented (coded in the left house position at cinputs[1,4:10])
            outputs[itrial,0] = rewardfun(cinputs, 0, reward_weights, n_feat_itemA, n_feat_itemB, n_actions, n_input_nodes) 
            outputs[itrial,1],isPassive[itrial]  = np.nan, 1 #insert nan in output for right house
            
            if sub in ['sub501', 'sub502']:
                if itrial_inblock<5:
                    passive_type[itrial] = 'pre_early'
                    
                elif itrial_inblock>7 and itrial_inblock<12:
                    passive_type[itrial] = 'post_early'
                    
                elif itrial_inblock>27 and itrial_inblock<32:
                    passive_type[itrial] = 'pre_late'
                    
                elif itrial_inblock>35 and itrial_inblock<40:
                    passive_type[itrial] = 'post_late'
                
            else:
                if itrial_inblock<5:
                    passive_type[itrial] = 'pre_early'
                    
                elif itrial_inblock>7 and itrial_inblock<12:
                    passive_type[itrial] = 'post_early'
                    
                elif itrial_inblock>27 and itrial_inblock<33:
                    passive_type[itrial] = 'pre_late'
                    
                elif itrial_inblock>36 and itrial_inblock<42:
                    passive_type[itrial] = 'post_late'

        else: # active trial - 2 possible outputs
            outputs[itrial,:] = [rewardfun(cinputs, x, reward_weights, n_feat_itemA, n_feat_itemB, n_actions, n_input_nodes) for x in range(n_actions)]
            
        # Log state code
        states[itrial] = input2state(inputs[itrial][np.array([0,1,4,5,6,7,10,11,12,13])])

    if isRecurrent:
        inputs = recurrent_inputs.copy()
 
    return inputs, outputs, isPassive, states, block, passive_type,  responses, correct_responses, probed_thresholds

#%% Define a function that generates a shuffled episode of 192 trials (all possible inputs)
def get_active_trials_input_comb(n_input_nodes,n_actions,n_feat_itemA):       

    # make inputs by combining all choice options
    itemA_options = np.array([[0, 1, 0, 1],
                           [0, 1, 1, 0],
                           [1, 0, 0, 1],
                           [1, 0, 1, 0]])
    itemB_options = np.array([[0,1,0,1,0,1],[0,1,0,1,1,0],
                             [0,1,1,0,0,1],[0,1,1,0,1,0],
                             [1,0,0,1,0,1],[1,0,0,1,1,0],
                             [1,0,1,0,0,1],[1,0,1,0,1,0]])
                             #all possible options for a single itemB
    inputs = np.zeros((itemA_options.shape[0]*itemB_options.shape[0]*(itemB_options.shape[0]-1), n_input_nodes)) # set array with all possible inputs (each row is an input for a single decision stage (i.e. itemA/itemB); entries 0-3 represent compA, entries 4-8 represent compB )
    inputs_itemA = np.zeros(itemA_options.shape[0]*itemB_options.shape[0]*(itemB_options.shape[0]-1))
    inputs_itemBacomp = np.zeros(itemA_options.shape[0]*itemB_options.shape[0]*(itemB_options.shape[0]-1))
    inputs_itemBbcomp = np.zeros(itemA_options.shape[0]*itemB_options.shape[0]*(itemB_options.shape[0]-1))
    trial_idx = -1
    
    for itemA in np.arange(0,itemA_options.shape[0]):
        for itemBacomp in np.arange(0, itemB_options.shape[0]):  
            for idx, itemBbcomp in enumerate(np.setdiff1d(np.arange(0, itemB_options.shape[0]), itemBacomp)):
                if ~(itemB_options[itemBacomp,0:4]==itemB_options[itemBbcomp,0:4]).all():
                    trial_idx = trial_idx+1
                    inputs[trial_idx, :] = np.concatenate([itemA_options[itemA,:], itemB_options[itemBacomp,:], itemB_options[itemBbcomp]])
                    inputs_itemA[trial_idx]=itemA
                    inputs_itemBacomp[trial_idx]=itemBacomp
                    inputs_itemBbcomp[trial_idx]=itemBbcomp
    
    inputs=inputs[:trial_idx+1,:]
    inputs_itemA=inputs_itemA[:trial_idx+1]
    inputs_itemBacomp=inputs_itemBacomp[:trial_idx+1]
    inputs_itemBbcomp=inputs_itemBbcomp[:trial_idx+1]
 
    return inputs 

#%% Define a function that gets an input and returns reward for selected action (0--> left house, 1--> right house)
def rewardfun(cinputs, selected_action, reward_weights, n_feat_itemA, n_feat_itemB,n_actions, n_input_nodes): 
    # geberate choicemap: mask inputs based on each of 2 possible choices
    choicemap = np.empty((n_actions, n_input_nodes))
    choicemap[0, :] = np.concatenate((np.repeat(1, n_feat_itemA*2), np.repeat(1, n_feat_itemB*2), np.repeat(0, n_feat_itemB*2)))# itemB:1
    choicemap[1, :] = np.concatenate((np.repeat(1, n_feat_itemA*2), np.repeat(0, n_feat_itemB*2), np.repeat(1, n_feat_itemB*2)))# itemB:2
    
    cidx = np.flatnonzero(choicemap[selected_action, :])
    reward_weights_tmp = reward_weights.copy()
    R = np.dot(cinputs[0,cidx[np.arange(0, n_feat_itemA*2)]], reward_weights_tmp[0:n_feat_itemA*2]) \
        * np.dot(cinputs[1,cidx[np.arange(n_feat_itemA*2,n_feat_itemA*2+n_feat_itemB*2)]], reward_weights_tmp[n_feat_itemA*2:n_feat_itemA*2+n_feat_itemB*2] ) 
    R=(R*6.25+50)/100 
    return(R)


#%% Define a function that recieves input array/state codes and returns state code/input array, respectively   
def input2state(x):
    # a1/b1 = 0,1,0,1
    # a2/b2 = 0,1,1,0
    # a3/b3 = 1,0,0,1
    # a4/b4 = 1,0,1,0

    state_2_input_dict = {  'b4b2': np.array([1,0,0,1,0,1,0,1,1,0]),
                            'b2b4': np.array([1,0,0,1,1,0,0,1,0,1]),
                            'b4b3': np.array([1,0,0,1,0,1,1,0,0,1]),
                            'b3b4': np.array([1,0,1,0,0,1,0,1,0,1]),
                            'b4b1': np.array([1,0,0,1,0,1,1,0,1,0]),
                            'b1b4': np.array([1,0,1,0,1,0,0,1,0,1]),
                            'b2b3': np.array([1,0,0,1,1,0,1,0,0,1]),
                            'b3b2': np.array([1,0,1,0,0,1,0,1,1,0]),
                            'b2b1': np.array([1,0,0,1,1,0,1,0,1,0]),
                            'b1b2': np.array([1,0,1,0,1,0,0,1,1,0]),
                            'b3b1': np.array([1,0,1,0,0,1,1,0,1,0]),
                            'b1b3': np.array([1,0,1,0,1,0,1,0,0,1]),
                        
                            'a4a2': np.array([0,1,0,1,0,1,0,1,1,0]),
                            'a2a4': np.array([0,1,0,1,1,0,0,1,0,1]),
                            'a4a3': np.array([0,1,0,1,0,1,1,0,0,1]),
                            'a3a4': np.array([0,1,1,0,0,1,0,1,0,1]),
                            'a4a1': np.array([0,1,0,1,0,1,1,0,1,0]),
                            'a1a4': np.array([0,1,1,0,1,0,0,1,0,1]),
                            'a2a3': np.array([0,1,0,1,1,0,1,0,0,1]),
                            'a3a2': np.array([0,1,1,0,0,1,0,1,1,0]),
                            'a2a1': np.array([0,1,0,1,1,0,1,0,1,0]),
                            'a1a2': np.array([0,1,1,0,1,0,0,1,1,0]),
                            'a3a1': np.array([0,1,1,0,0,1,1,0,1,0]),
                            'a1a3': np.array([0,1,1,0,1,0,1,0,0,1]),
                        
                            'b4_l': np.array([1,0,0,1,0,1,0,0,0,0]),
                            'b2_l': np.array([1,0,0,1,1,0,0,0,0,0]),
                            'b3_l': np.array([1,0,1,0,0,1,0,0,0,0]),
                            'b1_l': np.array([1,0,1,0,1,0,0,0,0,0]),
                        
                            'a4_l': np.array([0,1,0,1,0,1,0,0,0,0]),
                            'a2_l': np.array([0,1,0,1,1,0,0,0,0,0]),
                            'a3_l': np.array([0,1,1,0,0,1,0,0,0,0]),
                            'a1_l': np.array([0,1,1,0,1,0,0,0,0,0]),
                        
                            'b4_r': np.array([1,0,0,0,0,0,0,1,0,1]),
                            'b2_r': np.array([1,0,0,0,0,0,0,1,1,0]),
                            'b3_r': np.array([1,0,0,0,0,0,1,0,0,1]),
                            'b1_r': np.array([1,0,0,0,0,0,1,0,1,0]),
                        
                            'a4_r': np.array([0,1,0,0,0,0,0,1,0,1]),
                            'a2_r': np.array([0,1,0,0,0,0,0,1,1,0]),
                            'a3_r': np.array([0,1,0,0,0,0,1,0,0,1]),
                            'a1_r': np.array([0,1,0,0,0,0,1,0,1,0])}#for the sake of simplicity we take only those for the left hand side


    # a1 --> b4 (25)
    # a2 --> b2 (62)
    # a3 --> b3 (37)
    # a4 --> b1 (75)
    
    # b1 --> a4 (100)
    # b2 --> a2 (75)
    # b3 --> a3 (25)
    # b4 --> a1 (0)
    
    # Reverse dictionary function
    def reverse_dict(d):
        return {tuple(v): k for k, v in d.items()}
    
    # Create reversed dictionaries
    input_2_state_dict = reverse_dict(state_2_input_dict)
    #example ussage: input_2_state_dict[tuple(input_array)]
    
    if isinstance(x, str): #convert state to array
        y = state_2_input_dict[x]
    elif isinstance(x, np.ndarray): #convert array to state
        y = input_2_state_dict[tuple(x)]
    
    return(y)