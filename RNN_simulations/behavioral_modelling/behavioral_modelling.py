#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Linear function-approximation models of choice (Fig. 2).
#
# behavioral_modelling() runs one model on one participant's learning trials: on every trial the
# model estimates the value of both houses from its state representation, chooses with a
# softmax, and updates its weights from the reward. The models differ only in their state space:
#   'full_face'     - perceptual model: every combination of the visual features
#   'context+house' - compressed model: context x house (the task's latent states)
# Called by fit_lin_func_approx_models_grid_search.py.
# Input:  summary_files_behaviour/State_Formation_behaviour.csv
#
# Author: Shany Grossman

import os

# Folder with the participant-level behavioural summary files, found by searching upwards
# from this script (works on the cluster, on a mounted drive and in the code repository).
def _find_summary_files_behaviour():
    d = os.path.dirname(os.path.abspath(__file__))
    while not os.path.isdir(os.path.join(d, 'summary_files_behaviour')):
        if os.path.dirname(d) == d:
            raise FileNotFoundError('summary_files_behaviour folder not found above ' + os.path.abspath(__file__))
        d = os.path.dirname(d)
    return os.path.join(d, 'summary_files_behaviour') + os.sep
SUMMARY_DIR = _find_summary_files_behaviour()

def behavioral_modelling(input_model_name,
                         subj,
                         lr,
                         temp,
                         w_init,
                         update_rule='model',
                         greedy_choice=False):
    # update_rule : 'participant': update weights based on the participant's actual choice
    #               'model': update weights based on the model's own softmax choice
   
    #import packages:    
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import os
    
    #set default plotting settings:
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial"]
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['figure.dpi'] = 600
    plt.rcParams['savefig.dpi'] = 600
    plt.rcParams['axes.linewidth'] = 1
    
    
    #set RL params:
    alpha = lr #learning rate
    temprature = temp #15#for softmax action slection - smaller is more exploration
    #less exploration - better gradual fit-- why? 
    
    #set fitting params:
    
    #set models to run:
    models = [input_model_name]
    n_models = len(models)
    
    
    #what learner class to run on?
            
    #load behavioral log
    logs_df = pd.read_csv(SUMMARY_DIR + 'State_Formation_behaviour.csv')
    
    #prepare some useful functions
    def update_weigths(model,c_input,reward):
        #comute pe
        pe = reward - np.dot(c_input,model['w'])
        #update weights:
        w_new, w_old = model['w'].copy(), model['w'].copy()
        for i in range(len(c_input)):
            w_new[i] = w_old[i] + alpha*pe*c_input[i]
            
        #store new weigths in model:
        model['w'] = w_new
        return model
    
    #initialize model 
    def init_model(model_name):
        # w_init must be a pre-generated weight vector of the correct size
        model = {"w": np.array(w_init, dtype=float).copy(), "name": model_name}
        return model
    
    #convert raw input to state number (per house!):
    def convert_input_to_state(model_name, raw_inp, face=None):
        if model_name == 'full':  
            input2state_dict = {'1010101010':0,
                                '0110101010':1,
                                '1001101010':2,
                                '0101101010':3,
                                    
                                '1010011010':4,
                                '0110011010':5,
                                '1001011010':6,
                                '0101011010':7,
                                    
                                '1010010110':8,
                                '0110010110':9,
                                '1001010110':10,
                                '0101010110':11,
                                    
                                '1010010101':12,
                                '0110010101':13,
                                '1001010101':14,
                                '0101010101':15,
                                
                                '1010100110':16,
                                '0110100110':17,
                                '1001100110':18,
                                '0101100110':19,
                                
                                '1010100101':20,
                                '0110100101':21,
                                '1001100101':22,
                                '0101100101':23,
                                
                                '1010011001':24,
                                '0110011001':25,
                                '1001011001':26,
                                '0101011001':27,
                                
                                '1010101001':28,
                                '0110101001':29,
                                '1001101001':30,
                                '0101101001':31
                                }
            output = input2state_dict[raw_inp]
        elif model_name == 'context':
            input2state_dict = {'1010101010':0,
                                '0110101010':1,
                                '1001101010':0,
                                '0101101010':1,
                                    
                                '1010011010':2,
                                '0110011010':3,
                                '1001011010':2,
                                '0101011010':3,
                                    
                                '1010010110':4,
                                '0110010110':5,
                                '1001010110':4,
                                '0101010110':5,
                                    
                                '1010010101':6,
                                '0110010101':7,
                                '1001010101':6,
                                '0101010101':7,
                                
                                '1010100110':8,
                                '0110100110':9,
                                '1001100110':8,
                                '0101100110':9,
                                
                                '1010100101':10,
                                '0110100101':11,
                                '1001100101':10,
                                '0101100101':11,
                            
                                '1010011001':12,
                                '0110011001':13,
                                '1001011001':12,
                                '0101011001':13,
                                
                                '1010101001':14,
                                '0110101001':15,
                                '1001101001':14,
                                '0101101001':15
                                }
            output = input2state_dict[raw_inp]
        elif model_name =='context+house':
            input2state_dict = {'101010':0,
                                '101001':1,
                                '100110':2,
                                '100101':3,

                                '011010':4,
                                '011001':5,
                                '010110':6,
                                '010101':7
                                }
            output = input2state_dict[raw_inp[:2] + raw_inp[4:-2]]

        elif model_name in ('full_with_face', 'full_face'):
            # 64 states = 4 ctx_bit_patterns × 2 face identities × 8 houses
            # Reuse full-model state index (0-31), then offset by 32 for face='b'
            full_state = convert_input_to_state('full', raw_inp)
            face_offset = 32 if face == 'b' else 0
            output = full_state + face_offset

        return output
    
    #softmax action selectio rule    
    def softmax_action_selection(predicted_vals): #left, right options
        if len(predicted_vals)==1: #single output node --> convert to two values (0--> choose left, 1--> choose right)
            t = predicted_vals[0].copy()
            predicted_vals = np.transpose(np.array([1-t,t]) )     
        x = predicted_vals*temprature
        b = x.max()
        p = np.exp(x-b)/sum(np.exp(x-b))
        if greedy_choice:
            selected_action = np.argmax(predicted_vals)
        else:
            selected_action = np.random.choice(np.arange(2),p=p)
        return(selected_action,p) 
    
    #make choicemap: mask inputs based on each of 2 possible choices
    choicemap = np.empty((2, 16))
    n_feat_itemA, n_feat_itemB = 2,3
    choicemap[0, :] = np.concatenate((np.repeat(1, n_feat_itemA*2), np.repeat(1, n_feat_itemB*2), np.repeat(0, n_feat_itemB*2)))# itemB:1
    choicemap[1, :] = np.concatenate((np.repeat(1, n_feat_itemA*2), np.repeat(0, n_feat_itemB*2), np.repeat(1, n_feat_itemB*2)))# itemB:2
    
    #reward function
    def rewardfun(int_inp, selected_action): 
        cidx = np.flatnonzero(choicemap[selected_action, :])
        reward_weights = np.array([-1,2,0,0,-1,1,-3,3,0, 0])
        R = np.dot(int_inp[cidx[np.arange(0, n_feat_itemA*2)]], reward_weights[0:n_feat_itemA*2]) \
            * np.dot(int_inp[cidx[np.arange(n_feat_itemA*2,n_feat_itemA*2+n_feat_itemB*2)]], reward_weights[n_feat_itemA*2:n_feat_itemA*2+n_feat_itemB*2] ) 
        #rescale rewards to be in the range 0f 0-100
        R=(R*6.25+50)/100
        return(R)
    
    #%% Begin fitting loop over participants 
    
        
    for imodel, model_name in enumerate(models):
        
                
        for isub, sub in enumerate([subj]): #learners_dict[learner_type]):
            
            #filter to get only learning trials in day 1 of current subject:
            sub_log = logs_df[(logs_df['sub']==sub) & (logs_df['is_passive']==0) & (logs_df['block_type_num']<12) & (logs_df['response']!='none')] #12
            n_trials_ppt = sub_log.shape[0]
            print(sub + ' ' + str(n_trials_ppt) + ' trials model ' + model_name + ' lr=' +  str(lr) + ' temp=' + str(temp) + ' w init=' + str(w_init) + '\n')
        
            #extract target responses and actual responses::
            responses_ppt = np.array(sub_log.response)
            responses_ppt_int = np.zeros(len(responses_ppt))
            responses_ppt_int[:] = np.nan
            for j in range(len(responses_ppt)):
                if responses_ppt[j]=='left':
                    responses_ppt_int[j]=0
                elif responses_ppt[j]=='right':
                    responses_ppt_int[j]=1
                else: 
                    responses_ppt_int[j]=1000
                    
            correct_responses_ppt = np.empty(n_trials_ppt).astype('int32')
            correct_responses_ppt[sub_log.value_left>sub_log.value_right] = 0
            correct_responses_ppt[sub_log.value_left<sub_log.value_right] = 1


            #init model (for now, we start with the full model only and will add others later)
            model = init_model(model_name)
            
            #####start linear approximation loop over single trials######
            cwin=-1
            choices_p = np.zeros((n_trials_ppt,2))
            choices_p[:] = np.nan
            output_vals = np.zeros((n_trials_ppt,2))
            output_vals[:] = np.nan
            selected_action = np.zeros(n_trials_ppt)
            selected_action[:] = np.nan
            correct_responses = np.zeros(n_trials_ppt)
            correct_responses[:] = np.nan

            for itrial in range(n_trials_ppt):
                raw_inp = sub_log['input'].iloc[itrial][1:-1].replace(' ', '')
                face_trial = sub_log['face'].iloc[itrial] if 'face' in sub_log.columns else None

                #convert input to current model's state for left and right house
                lstate = convert_input_to_state(model_name, raw_inp[:10], face=face_trial)
                rstate = convert_input_to_state(model_name, raw_inp[:4] + raw_inp[10:], face=face_trial)
                left_input, right_input = np.zeros(model['w'].shape[0]), np.zeros(model['w'].shape[0])
                left_input[lstate] = 1
                right_input[rstate] = 1
                
                #compute estimated reward for each house
                l_reward_est, r_reward_est = np.dot(model['w'],left_input), np.dot(model['w'],right_input)
                output_vals[itrial, 0] = l_reward_est
                output_vals[itrial, 1] = r_reward_est

                #sofmax selection of house and recieve reward
                selected_action[itrial],choices_p[itrial,:]= softmax_action_selection(np.array([l_reward_est,r_reward_est]))
                int_inp = list()
                for x in raw_inp:
                    int_inp.append(np.fromstring(x, dtype=np.int32, sep=' '))
                int_inp = np.array(int_inp).T[0]

                #log correct response
                reward_l, reward_r = rewardfun(int_inp, 0), rewardfun(int_inp, 1)
                if reward_l>reward_r: correct_response = 0
                else: correct_response = 1
                assert(correct_responses_ppt[itrial]==correct_response)
                correct_responses[itrial] = correct_response

                #update weights based on update_rule flag
                if update_rule == 'participant':
                    action_for_update = int(responses_ppt_int[itrial])  # participant's actual choice
                else:  # 'model'
                    action_for_update = int(selected_action[itrial])    # model's own choice

                reward = rewardfun(int_inp, action_for_update)
                if action_for_update == 0:
                    c_input = left_input
                    model = update_weigths(model, c_input, reward)
                elif action_for_update == 1:
                    c_input = right_input
                    model = update_weigths(model, c_input, reward)
                #else: no response (1000), skip update


    #%%
    ppt_resp = responses_ppt_int
    model_resp = selected_action
    return (model_resp, ppt_resp, choices_p, correct_responses, output_vals)
        
            