#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Helper functions for the network simulations (softmax action selection, reward, ...).
# Used by RNN_simulation.py.
#
# Author: Shany Grossman

import numpy as np

#%% Sampling from a soft max probablities vector with a certain temprature 
def sample_from_soft_max(p, temprature):
    # helper function to sample an index from a probability array 
    p = np.log(p) /(1/temprature)
    p = np.exp(p) / np.sum(np.exp(p))
    return (np.argmax(np.random.multinomial(1, p, 1)),np.argmax(p),p)


def softmax_action_selection(predicted_vals, temprature):
    if len(predicted_vals)==1:  # single output node --> convert to two values
        t = predicted_vals[0].copy()
        predicted_vals = np.transpose(np.array([1-t, t]))

    # convert + guard
    x = np.asarray(predicted_vals, dtype=np.float64) * temprature

    # ---- NEW: numerical stability + NaN guard ----
    if not np.all(np.isfinite(x)):
        p = np.array([0.5, 0.5])
    else:
        b = np.max(x)
        z = x - b
        expz = np.exp(z)
        denom = np.sum(expz)

        if denom == 0 or not np.isfinite(denom):
            p = np.array([0.5, 0.5])
        else:
            p = expz / denom
    # --------------------------------------------

    # keep your greedy policy exactly as before
    selected_action = np.argmax(predicted_vals)
    selected_action_wo_exploration = np.argmax(predicted_vals)

    return selected_action, selected_action_wo_exploration, p


def softmax_action_selection_substraction(predicted_vals,temprature):
    assert(len(predicted_vals)==1) #single output node --> convert to two values (0--> choose left, 1--> choose right)
    t = np.array([predicted_vals[0].copy(),0])
    x = t*temprature
    b = x.max()
    p = np.exp(x-b)/sum(np.exp(x-b))
    selected_action = np.random.choice(np.arange(len(t)),p=p)
    selected_action_wo_exploration = np.argmax(t)
    return(selected_action,selected_action_wo_exploration,p) 

#%%
def rewardfun(cinputs, selected_action, reward_weights, n_feat_itemA, n_feat_itemB,n_actions, n_input_nodes): 
    # make choicemap: mask inputs based on each of 2 possible choices
    choicemap = np.empty((n_actions, n_input_nodes))
    choicemap[0, :] = np.concatenate((np.repeat(1, n_feat_itemA*2), np.repeat(1, n_feat_itemB*2), np.repeat(0, n_feat_itemB*2)))# itemB:1
    choicemap[1, :] = np.concatenate((np.repeat(1, n_feat_itemA*2), np.repeat(0, n_feat_itemB*2), np.repeat(1, n_feat_itemB*2)))# itemB:2
    
    cidx = np.flatnonzero(choicemap[selected_action, :])
    reward_weights_tmp = reward_weights.copy()
    R = np.dot(cinputs[0,cidx[np.arange(0, n_feat_itemA*2)]], reward_weights_tmp[0:n_feat_itemA*2]) \
        * np.dot(cinputs[1,cidx[np.arange(n_feat_itemA*2,n_feat_itemA*2+n_feat_itemB*2)]], reward_weights_tmp[n_feat_itemA*2:n_feat_itemA*2+n_feat_itemB*2] ) 
    R=(R*6.25+50)/100 #make sure to add this to all other!
    return(R)

#%%
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

#%%
def active_trial_mapiulation(manipulation, cinputs):
    manip_input = cinputs.copy()
    if manipulation == 'flip side':
        manip_input[1,4:10] = cinputs[1,10:]
        manip_input[1,10:] = cinputs[1,4:10]
                
    elif manipulation == 'change_irrl_client':
        manip_input[0,2:4] = np.flip(cinputs[0,2:4])
                
    elif manipulation == 'change_irrl_lhouse':
        manip_input[1,8:10] = np.flip(cinputs[1,8:10])
                
    elif manipulation == 'change_irrl_rhouse':
        manip_input[1,14:16] = np.flip(cinputs[1,14:16])
                
    elif manipulation == 'change_irrl_lhouse_and_rhouse':
        manip_input[1,8:10] = np.flip(cinputs[1,8:10])
        manip_input[1,14:16] = np.flip(cinputs[1,14:16])
                
    elif manipulation == 'change_all_irrel':
        manip_input[1,8:10] = np.flip(cinputs[1,8:10])
        manip_input[1,14:16] = np.flip(cinputs[1,14:16])
                
    elif manipulation == 'change_client_rel_feature':  
        manip_input[0,0:2] = np.flip(cinputs[0,0:2])
                
    elif manipulation == 'change_client': 
        manip_input[0,0:2] = np.flip(cinputs[0,0:2])
        manip_input[0,2:4] = np.flip(cinputs[0,2:4])
                
    elif manipulation == 'change_lhouse_primary_feature': 
        inputs_change_primary_lhouse = cinputs.copy()
        inputs_change_primary_lhouse[1,6:8] = np.flip(cinputs[1,6:8])
                
    elif manipulation == 'change_lhouse_secondary_feature': 
        manip_input[1,4:6] = np.flip(cinputs[1,4:6])
                
    elif manipulation == 'change_rhouse_primary_feature': 
        manip_input[1,12:14] = np.flip(cinputs[1,12:14])
                
    elif manipulation == 'change_rhouse_secondary_feature': 
        manip_input[1,10:12] = np.flip(cinputs[1,10:12])
                
    elif manipulation == 'change_houses_primary_feature': 
            manip_input[1,12:14] = np.flip(cinputs[1,12:14])
            manip_input[1,6:8] = np.flip(cinputs[1,6:8])
                
    elif manipulation == 'change_houses_secondary_feature': 
        manip_input[1,4:6] = np.flip(cinputs[1,4:6])
        manip_input[1,10:12] = np.flip(cinputs[1,10:12])  
                
    elif manipulation == 'change_client_lhouse_primary_feature': 
        manip_input[1,6:8] = np.flip(cinputs[1,6:8])
        manip_input[0,0:2] = np.flip(cinputs[0,0:2])
                
    elif manipulation == 'change_client_lhouse_secondary_feature': 
        manip_input[1,4:6] = np.flip(cinputs[1,4:6])
        manip_input[0,0:2] = np.flip(cinputs[0,0:2])
        
    elif manipulation == 'change_client_rhouse_primary_feature': 
        manip_input[1,12:14] = np.flip(cinputs[1,12:14])
        manip_input[0,0:2] = np.flip(cinputs[0,0:2])
                
    elif manipulation == 'change_client_rhouse_secondary_feature': 
        manip_input[1,10:12] = np.flip(cinputs[1,10:12])
        manip_input[0,0:2] = np.flip(cinputs[0,0:2])
                    
    elif manipulation == 'change_client_houses_primary_feature': 
        manip_input[1,12:14] = np.flip(cinputs[1,12:14])
        manip_input[1,6:8] = np.flip(cinputs[1,6:8])
        manip_input[0,0:2] = np.flip(cinputs[0,0:2])
                    
    elif manipulation == 'change_client_houses_secondary_feature': 
        manip_input[1,4:6] = np.flip(cinputs[1,4:6])
        manip_input[1,10:12] = np.flip(cinputs[1,10:12])  
        manip_input[0,0:2] = np.flip(cinputs[0,0:2])
        
    return manip_input