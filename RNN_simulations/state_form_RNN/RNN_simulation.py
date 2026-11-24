#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This script builds a single recurrent network, trains it using RL with inputs matched to a specific particpants curriculum.
# Also records the hidden-layer activity on every trial, which is used for the co-update analysis.
# Used by fit_RNN_to_participants.py.
#
# Author: Shany Grossman
#
# Inputs:
#   -activation_dir_name:    path to folder where simulation outputs will be saved, in a subploder named after the participant from whom protocol was taken (e.g. activation_dir_name/sub-501/)
#   -alternate_rmap:         if true, run the etire task prtocol including the change in reward function in block 15.
#                            if false, no reward remapping, train netwrok with the single reward function until  the end of day 1 (11 learning block)
#   -storePassiveActivation: if true, feedforwad all paasive trial before each learning trial and store passive activation for offline analyses of coupdate patterns  
#   -storePassiveActivation_empiricalSampling: if true, feedfarwrd passive trials exactly as presented in exepriment 
#   -subj:                   protocol of which particiapnt to use ('sub-5XX')
#   -lr:                     learning rate (0-1)
#   -temp:                   INVERSE temprature value for the softmax action selection rule, samllar --> greater exploration
#   -is_fitting_opt:         if true, returns rnn_resp, ppt_resp, choices_p) and save participants responses in activation_dir_name/sub5XX/
#   -input_client_type_only: if true, input only relevant client feature on each trial and set all the rest to zero
#   -doActiveTrialManipulationMass: if true, present all 192 possible input variation before and after each trial and log the netwrok's choice to later analyses behavioral impact of single trial feedback
#   -same_reward_range:      if true, reward weight are changed from original experiemnt such that reward in both context range between 0 and 1
#   -fc_gain:                factor by which the sd of the RNN weight is multiplied upon intialization 
#   -rnn_gain:               factor by which the sd of the FC layers' weights is multiplied upon intialization 
#   -seed:
#   -add_final_passive_sweep: if true, run one extra passive sweep after the final learning trial
#                            (saved at index i_learning_trial+1), so that the last empirical segment
#                            (trials 260-264) has activations after its last learning trial


# %%
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

def RNN_simulation_v3_one_hot_coupled_weights_single_item(activation_dir_name,
                                                          alternate_rmap, 
                                                          storePassiveActivation, 
                                                          storePassiveActivation_empiricalSampling, 
                                                          subj, lr, temp, 
                                                          is_fitting_opt, 
                                                          input_client_type_only,
                                                          doActiveTrialManipulationMass,
                                                          storeActivations,
                                                          same_reward_range,
                                                          fc_gain,
                                                          rnn_gain,
                                                          seed,
                                                          add_final_passive_sweep=False):


    # %% General
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    import pandas as pd
    import numpy as np
    import random
    import gc

    # Neural network
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.layers import Input, Dense, SimpleRNN
    from tensorflow.keras.models import Model
    if storePassiveActivation or storePassiveActivation_empiricalSampling:
        from keract import get_activations

    # Helper function for training network
    from run_RNN_helper_funcs import get_active_trials_input_comb
    from run_RNN_helper_funcs import softmax_action_selection
    from run_RNN_helper_funcs import rewardfun
    
    # Function that brings back subjects specific protocols:
    from stateformation_get_input_output import get_input_output_stateformation

    random.seed(int(seed))
    np.random.seed(int(seed))
    os.environ['PYTHONHASHSEED'] = str(int(seed))
    tf.random.set_seed(int(seed))

    #%%  Make sure output folder for storing activations exists 
    if not(os.path.isdir(activation_dir_name)): os.mkdir(activation_dir_name)
    
    # Create a subfolder with the subject name (from whom we took the experimental protocol)
    folder_name = activation_dir_name + '/' + subj
    if not(os.path.isdir(folder_name)): os.mkdir(folder_name)


    #%% Set parameters 

    # Protocol related parametrs 
    remapping_block = 14 # at what learning block number should we remap? - counting starts from zero!
 
    # Task related parameters 
    n_feat_itemA = 2 #2 client features
    n_feat_itemB = 3 #3 house features 

    n_itemA = 1 # the agent has to choose one out of n_itemA
    n_itemB = 2 # next, the agent has to choose one out of n_itemB
    n_input_nodes = n_feat_itemA*n_itemA*2 + n_feat_itemB*n_itemB*2 #check if can get rid of 

    n_actions=2
    
    # Get all possible input variations in the task
    active_inputs = get_active_trials_input_comb(n_input_nodes,n_actions,n_feat_itemA) #inputs: 192X16 array (need to break into 2 time points when passed forward)
    maxtrials = active_inputs.shape[0]

    # RNN related params
    remapping_type = 3 #swap preference for less improtant house feature
    grand_log_csv_path = SUMMARY_DIR + "State_Formation_behaviour.csv" #csv with logs for all participants
    
    #%% Build RNN that approximates [state,action] value and implements shared input weights for houses on both sides 
    
    tf.keras.backend.clear_session()
    
    # Define weights distributions at initialization
    fan_in_hidden1, fan_out_hidden1 = 10, 32
    fan_in_RNN, fan_out_RNN = fan_out_hidden1,fan_out_hidden1 
    fan_in_hidden2, fan_out_hidden2 = 16, 1
    
    initializer_hidden1 = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev = np.sqrt(2/ (fan_in_hidden1 + fan_out_hidden1)) * fc_gain) # seed=seed)
    initializer_hidden1RNN = tf.keras.initializers.TruncatedNormal(mean=0.0,   stddev = np.sqrt(2 / (fan_in_RNN + fan_out_RNN)) * rnn_gain) #, seed=seed)
    initializer_hidden2 = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev = np.sqrt(2 / (fan_in_hidden2 + fan_out_hidden2)) * fc_gain) #, seed=seed)
    initializer_output = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev = np.sqrt(2 / (fan_out_hidden2 + 1))* fc_gain)

    # Define multiple input layers (e.g., two input layers for simplicity)
    input_item = Input(shape=(2,10)) #4 client nodes (2 features, 1 is relevant) + 6 house nodes (3 features, 2 are relevant)

    # Create a shared Dense layer (this will use the same weights for both inputs)
    shared_simpleRNN_layer = SimpleRNN(fan_out_hidden1,activation='tanh', 
              input_shape=(2, fan_in_hidden1),
              kernel_initializer=initializer_hidden1,
              recurrent_initializer=initializer_hidden1RNN,
              stateful=False,
              use_bias=False,
              return_sequences=True,
              name="RNN_hidden1")

    # Apply the shared layer to both inputs
    simpleRNN_output = shared_simpleRNN_layer(input_item)

    # Further processing after combining 
    hidden2 = Dense(fan_in_hidden2, activation='relu',
                    kernel_initializer=initializer_hidden2, use_bias=False, name="hidden2")(simpleRNN_output)

    # Final output layer - value prediction for current clinet+house pair
    output = Dense(fan_out_hidden2, activation='linear',
                   kernel_initializer=initializer_output, use_bias=False, name="output")(hidden2)

    # Build the model
    model = Model(inputs=input_item, outputs=output)

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr), #clipnorm=1.0, momentum=0.75
                  loss=tf.keras.losses.MeanSquaredError())


    if storeActivations:
        activation_models = {
            layer.name: tf.keras.Model(
                inputs=model.input,
                outputs=layer.output)
            for layer in model.layers
            if "input" not in layer.name}
    
    # Helper funs:
    def get_layer_output_dim(layer):

        # Return the output dimensionality of a Keras layer
        # in a way that works across TF/Keras versions.

        # Best case: Dense layer
        if hasattr(layer, "units"):
            return layer.units
    
        # Older TF / Keras
        if hasattr(layer, "output_shape"):
            return layer.output_shape[-1]
    
        # Newer TF / Keras (model must be built)
        if hasattr(layer, "output") and layer.output is not None:
            return int(layer.output.shape[-1])
    
        raise RuntimeError(
            f"Cannot infer output dimension for layer {layer.name} "
            f"(type {type(layer)})")

    #%% Generate passive inputs (only 16 states, irrel feats set to zero)
    if storePassiveActivation: 
        #First pass only passive inputs that include only relevant features
        a = np.array([[0,1,0,0],[1,0,0,0]])
        b = np.array([[0,1,0,1,0,0],[0,1,1,0,0,0],
                      [1,0,0,1,0,0],[1,0,1,0,0,0]])

        passive_inputs = np.zeros((2,n_input_nodes,16))
        count=-1
        for ia in range(a.shape[0]):
            for ib in range(b.shape[0]):
                for side in range(2): #0--> left, 1-->right
                    count=count+1
                    passive_inputs[0,0:4,count] = a[ia]
                    if side==0:
                        passive_inputs[1,4:10,count] = b[ib]
                    elif side==1:
                        passive_inputs[1,10:16,count] = b[ib]
                        
        #Now pass all passive inputs that include also irrelvant variations (4 times bigger) 
        a = np.array([[0,1,0,1],[0,1,1,0],[1,0,0,1],[1,0,1,0]])
        b = np.array([[0,1,0,1,0,1],[0,1,0,1,1,0],[0,1,1,0,0,1],[0,1,1,0,1,0],
                      [1,0,0,1,0,1],[1,0,0,1,1,0],[1,0,1,0,0,1],[1,0,1,0,1,0]])

        passive_inputs_full = np.zeros((2,n_input_nodes,16*4))
        count=-1
        for ia in range(a.shape[0]):
            for ib in range(b.shape[0]):
                for side in range(2): #0--> left, 1-->right
                    count=count+1
                    passive_inputs_full[0,0:4,count] = a[ia]
                    if side==0:
                        passive_inputs_full[1,4:10,count] = b[ib]
                    elif side==1:
                        passive_inputs_full[1,10:16,count] = b[ib]
                        
        # Merge input passive full to compute input correlations
        merged = np.concatenate([
            passive_inputs_full[0, :4],     # shape (4, 64)
            passive_inputs_full[1, 4:]      # shape (12, 64)
        ], axis=0)
        
        # Plot passive inputs correlations:
        corr_matrix = np.corrcoef(merged.T)
        # Top-left 32x32 block (excluding diagonal)
        block1 = corr_matrix[0:32, 0:32]
        mean_block1 = (np.sum(block1) - np.trace(block1)) / (block1.size - block1.shape[0])
        
        # Bottom-right 32x32 block (excluding diagonal)
        block2 = corr_matrix[0:32, 32:64]
        mean_block2 = (np.sum(block2) - np.trace(block2)) / (block2.size - block2.shape[0])


    #%% Define reward function - this defines how the reward depends on the features of the 1st and 2nd item
    if same_reward_range:
        reward_weights = np.array([-1,1,0,0,-1,1,-3,3,0, 0])
    else:
        reward_weights = np.array([-1,2,0,0,-1,1,-3,3,0, 0])
        
    if alternate_rmap and same_reward_range:
        if remapping_type == 0:
            reward_weights_b = np.array([1,-1,0,0,-1,1,-3,3,0, 0]) #swaps client feature weights
        elif remapping_type==1:
            reward_weights_b = np.array([-1,1,0,0,-3,3,-1,1,0, 0]) #swaps house feature weights
        elif remapping_type==2:
            reward_weights_b = np.array([-1,1,0,0,-1,-3,1,3,0, 0]) #swaps f1_1 and f2_-3 (swap weight of house f1 and house f2)
        elif remapping_type==3:
            reward_weights_b = np.array([-1,1,0,0,1,-1,-3,3,0, 0]) #swaps signs of 2nd house feature 
        else :
            print('invalid input in remapping type (should be 0/1/2/3') 
        
    if alternate_rmap and not same_reward_range:
        # ignore if alternate_rmap==False
        if remapping_type == 0:
            reward_weights_b = np.array([2,-1,0,0,-1,1,-3,3,0, 0]) #swaps client feature weights
        elif remapping_type==1:
            reward_weights_b = np.array([-1,2,0,0,-3,3,-1,1,0, 0]) #swaps house feature weights
        elif remapping_type==2:
            reward_weights_b = np.array([-1,2,0,0,-1,-3,1,3,0, 0]) #swaps f1_1 and f2_-3 (swap weight of house f1 and house f2)
        elif remapping_type==3:
            reward_weights_b = np.array([-1,2,0,0,1,-1,-3,3,0, 0]) #swaps signs of 2nd house feature 
        else :
            print('invalid input in remapping type (should be 0/1/2/3') 

    #%% Set empty arrays for storing hidden activations, performance, predeicted reward and more
         
    R_all = np.full((300,), np.nan) # store actual trial by trial reward
    R_all_options = np.full((600, 2), np.nan) # store trial by trial reward for all presented options
    output_all = np.full((300,), np.nan) # store trial by trial predicted reward (for selected item)
    R_predicted_all_options = np.full((300, 2), np.nan) # store trial by trial predicted reward for all presented options
    pe_all = np.full((300,), np.nan) # store trial by trial prediction error (actual rewrd - predicted reward for slected action)
    max_reward_all = np.full((300,), np.nan) # store trial by trial maximally avialable reward
    correct = np.full((300,), np.nan) # store correct choice per trial
    correct_wo_exploration = np.full((300,), np.nan) # store correct choice per trial if no softmax delpoyed for choice
    selected_action_all = np.full((300,), np.nan) # store trial by trial action
    selected_action_wo_exploration_all = np.full((300,), np.nan) # store trial by trial action that woudl have beeon slected if there was no softmax deployed
    choices_p = np.full((300, 2), np.nan) # store choice probablities (for computing log likelihood trying to fit participant's learning curve)
    losses = np.full((300,), np.nan) #for storing trial by trial loss

    # Init arrays for storing acitvations (2 time points per trial) 
    activation_names =  [layer.name for layer in model.layers]
    activations = [[] for _ in range(len(activation_names))] #input, RNN_hidden1, hidden2, output
    for i in range(len(activations)):
        if i==0:
            activations[i] =  np.zeros((300,2,16), dtype = 'float32') #input 
        else:
            output_shape = get_layer_output_dim(model.layers[i])
            activations[i] =  np.zeros((300,2,output_shape*2), dtype = 'float32') #will concat left and right reps  

    if storePassiveActivation:
        activations_passive = [[] for _ in range(len(activation_names))] #input, RNN_hidden1, hidden2, output \  
        activations_passive_full = [[] for _ in range(len(activation_names))] #input, RNN_hidden1, hidden2, output \  
        for i in range(len(activations)):
            if i==0: #artificialy set it to 16 inputs nodes instead of 10
                activations_passive[i] =  np.zeros((300,2,16,passive_inputs.shape[-1]), dtype = 'float32')
                activations_passive_full[i] =  np.zeros((300,2,16,passive_inputs_full.shape[-1]), dtype = 'float32')
            else:
                output_shape = get_layer_output_dim(model.layers[i])
                activations_passive[i] =  np.zeros((300,2,output_shape*2,passive_inputs.shape[-1]), dtype = 'float32') #input X 16 passive inputs states 
                activations_passive_full[i] =  np.zeros((300,2,output_shape*2,passive_inputs_full.shape[-1]), dtype = 'float32')
    
    if storePassiveActivation_empiricalSampling:
        # Initialize empty arrays to store passive activations, and a df which describes what happened in each trial
        activations_passive_empricialSampling = [[] for _ in range(len(activation_names))] #input, RNN_hidden1, hidden2, output \  
        for i in range(len(activations)):
            if i==0: #artificialy set it to 16 inputs nodes instead of 10
                activations_passive_empricialSampling[i] =  np.zeros((300,2,16), dtype = 'float32')
            else:
                output_shape = get_layer_output_dim(model.layers[i])
                activations_passive_empricialSampling[i] =  np.zeros((300,2,output_shape*2), dtype = 'float32') #input X 16 passive inputs states 
        
        dfPassive_empricalSampling = pd.DataFrame(columns = ['sub', 'block', 'trial_type', 'input', 'value','state'])
        
    if doActiveTrialManipulationMass: 
        # Use the above generateed 192(X16) input variations of active trials to present before and after each uptate:
        n_manipulations = active_inputs.shape[0]
        
        # Create an Empty DataFrame object with column names only to store correlation vals
        dfMass = pd.DataFrame(columns = ['sub', 'trial', 'original_input', 'input', 'manipulation',
                                     'left_value','right_value','orig_response', 'response',
                                     'correct_response_orig', 'correct_response_manip',
                                     'pe_left', 'pe_right', 'pe_choice', 'is_correct',
                                     'probL_pre', 'probL_post', 'probR_pre', 'probR_post'])
        
        activations_active_mass_manipulations = [[] for _ in range(len(activation_names))] #input, RNN_hidden1, hidden2, output 
        for i in range(len(activations)):
            if i==0:
                activations_active_mass_manipulations[0] =  np.zeros((300*maxtrials,2,16,n_manipulations+1), dtype = 'float32')
            else:
                output_shape = get_layer_output_dim(model.layers[i])
                activations_active_mass_manipulations[0] =  np.zeros((300*maxtrials,2,output_shape*2,n_manipulations+1), dtype = 'float32')
  
    #%% Begin RNN training

    inputs, outputs, isPassive, states, block_num, passive_type, responses_ppt, correct_responses, probed_thresholds = \
        get_input_output_stateformation(sub=subj, csvpath = grand_log_csv_path.split('State_Formation_behaviour.csv')[0], isRecurrent=True, withPassive=True)
    
    rnn_resp = np.array(['none']*len(responses_ppt)) #if rnn chooses 0, we overwrite trial as "left"
    n_trials = len(isPassive)
    i_learning_trial=-1
    i_passive_trial_empricalSampling=-1

    # FIX: wrap entire training loop + save block in try/finally so model and
    # activation_models are always deleted even if an exception is raised mid-run
    try:
        
        for i_trial in range(n_trials):
            if isPassive[i_trial]==0: #run a learning trial as usual
                i_learning_trial = i_learning_trial+1
                
                cinputs = inputs[i_trial,:,:]
                        
                if alternate_rmap and block_num[i_learning_trial]>remapping_block:
                    reward_weights=reward_weights_b.copy()
                    print('altered reward weight for remapping...')
                    
                if input_client_type_only:
                    cinputs[1,:]=0
                    cinputs[0,2:]=0
                
                # Construct inputs to fit the current model's serial architecture
                left_input  = np.array([np.vstack((cinputs[0,:10], cinputs[1,:10]))],)
                right_input = np.array([np.vstack((cinputs[0,:10], np.hstack((np.zeros(4), cinputs[1,10:]))))],)
                
                # Log layer-specific activations for current trial:
                if storeActivations:
                    activations[0][i_learning_trial,:,:] = cinputs # log inputs (in orginal 2X16 format)
                    
                    c=0
                    for layer_idx, layer in enumerate(model.layers): # log subsequent layers' activations:
                        if (not 'input' in model.layers[layer_idx].name):
                            c+=1
                            l_cactivations = activation_models[layer.name](left_input, training=False).numpy()
                            r_cactivations = activation_models[layer.name](right_input, training=False).numpy()
        
                            activations[c][i_learning_trial,:,:] = np.array(tf.concat([l_cactivations, r_cactivations],axis=2))
                            
                            if storePassiveActivation: # feedforward all 24 passive states in the network after and log activations:
                                # First for passive without irrelevant features:
                                for i_passive_input in range(passive_inputs.shape[-1]):
                                    
                                    c_passive_input = passive_inputs[:,:,i_passive_input].copy()
            
                                    activations_passive[0][i_learning_trial,:,:,i_passive_input] = c_passive_input.copy()
                                    
                                    left_passive_input, right_passive_input = np.array([np.vstack((c_passive_input[0,:10],c_passive_input[1,:10]))],),  np.array([np.vstack((c_passive_input[0,:10],np.hstack((np.zeros(4),c_passive_input[1,10:]))))],)

                                    l_cactivations_passive = get_activations(model, left_passive_input,layer_names=layer.name, auto_compile=False)
                                    r_cactivations_passive = get_activations(model, right_passive_input, layer_names=layer.name, auto_compile=False)

                                    # concatantate inner reps for both houses and save in activations:
                                    activations_passive[c][i_learning_trial,:,:,i_passive_input] = np.array(tf.concat([l_cactivations_passive[layer.name], r_cactivations_passive[layer.name]],axis=2))

                                # Now, for passive_full - WITH irrelevant features variations (4 times more variations):
                                for i_passive_input in range(passive_inputs_full.shape[-1]):

                                    c_passive_input = passive_inputs_full[:,:,i_passive_input].copy()

                                    activations_passive_full[0][i_learning_trial,:,:,i_passive_input] = c_passive_input.copy()

                                    left_passive_input, right_passive_input = np.array([np.vstack((c_passive_input[0,:10],c_passive_input[1,:10]))],),  np.array([np.vstack((c_passive_input[0,:10],np.hstack((np.zeros(4),c_passive_input[1,10:]))))],)

                                    l_cactivations_passive = get_activations(model, left_passive_input,layer_names=layer.name, auto_compile=False)
                                    r_cactivations_passive = get_activations(model, right_passive_input, layer_names=layer.name, auto_compile=False)

                                    #concatantate inner rep for both houses and save in activations:
                                    activations_passive_full[c][i_learning_trial,:,:,i_passive_input] = np.array(tf.concat([l_cactivations_passive[layer.name], r_cactivations_passive[layer.name]],axis=2))
                            
                # Pass forward inputs and log predicted values 
                preds = model(np.concatenate([left_input, right_input], axis=0), training=False).numpy()
            
                l_predicted_val = preds[0][-1]
                r_predicted_val = preds[1][-1]

                predicted_vals = np.array([l_predicted_val[0],r_predicted_val[0]])  #concatante predicted values for both left and right house in this trial
        
                # Choose action according to a soft-max rule
                selected_action,selected_action_wo_exploration,p = softmax_action_selection(predicted_vals,temp)
                choices_p[i_learning_trial,:]=p # log action probabilities to later compute log likelihood if this is part of a fitting procedure 
               
                # Log predicted value of chosen house
                output_all[i_learning_trial] = predicted_vals[selected_action]
                
                # Recieve reward, compute PE and update action value
                R_all_ctrial = [rewardfun(cinputs, x, reward_weights,  n_feat_itemA, n_feat_itemB, n_actions, n_input_nodes) for x in range(n_actions)]
                R_all_options[i_learning_trial,:] = R_all_ctrial
                R_predicted_all_options[i_learning_trial*n_actions:i_learning_trial*n_actions+n_actions] = predicted_vals
                
                R = R_all_ctrial[selected_action]
                correct_response_orig=0 if R_all_ctrial[0]>R_all_ctrial[1] else 1
        
                pe_left = R_all_ctrial[0]-predicted_vals[0]
                pe_right = R_all_ctrial[1]-predicted_vals[1]
                pe = R-predicted_vals[selected_action]
        
                # Log some trial info
                selected_action_all[i_learning_trial] = selected_action
                if selected_action==0: rnn_resp[i_learning_trial]='left'
                elif selected_action==1: rnn_resp[i_learning_trial]='right'
                selected_action_wo_exploration_all[i_learning_trial] = selected_action_wo_exploration
                max_reward_all[i_learning_trial] = np.argmax(R_all)
                pe_all[i_learning_trial] = pe
                R_all[i_learning_trial] = R
                
                correct[i_learning_trial] =  int(np.isin(selected_action,np.argwhere(R_all_ctrial == np.amax(R_all_ctrial))))
                correct_wo_exploration[i_learning_trial] =int(np.isin(selected_action_wo_exploration,np.argwhere(R_all_ctrial == np.amax(R_all_ctrial))))
                
                if doActiveTrialManipulationMass:
                    model_pre_manip = keras.models.clone_model(model)
                    model_pre_manip.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr,momentum=0.0),
                                            loss=tf.keras.losses.MeanSquaredError())
                    model_pre_manip.set_weights(model.get_weights())
                
                # Train the net that approximates v(s,a) on the chosen action
                if selected_action == 0:
                    selected_input = left_input.copy()
                else:
                    selected_input = right_input.copy()

                target = R_all_ctrial[selected_action]
                history_callback = model.fit(selected_input, np.array([target]), epochs=1, batch_size=1, verbose=0)
                losses[i_learning_trial] = np.array(history_callback.history["loss"])
                
                # Run active trials manipulations (to search for behavioral effects of co-updates)
                if doActiveTrialManipulationMass:
                    # First, add df row for original input
                    new_row = {'sub': subj, 'trial':i_learning_trial, 'original_input' : cinputs.copy(), 'input' : cinputs.copy(), 'manipulation' : 'none', 
                                'left_value': R_all_ctrial[0] ,'right_value': R_all_ctrial[1] ,
                                'orig_response': selected_action, 'response': selected_action,
                                'correct_response_orig': correct_response_orig, 
                                'correct_response_manip': correct_response_orig, 
                                'pe_left' : pe_left, 'pe_right' : pe_right , 'pe_choice' : pe, 
                                'is_correct' : correct[i_learning_trial],'is_correct_pre_feedback' : correct[i_learning_trial]}
        
                    dfMass = pd.concat([dfMass, pd.DataFrame([new_row])], ignore_index=True)
                    
                    # Present the same trial just presented but with all possible changes and measure PE and accuracy:
                    for i_manipulation in range(active_inputs.shape[0]):
                        manipulation = 'TBD'
                        c_manip_input =  np.vstack((np.hstack((active_inputs[i_manipulation,:4], np.zeros(12))),np.hstack((np.zeros(4),active_inputs[i_manipulation,4:])))) 
                        c_manip_left_input = np.array([np.vstack((c_manip_input[0,:10],c_manip_input[1,:10]))],)
                        c_manip_right_input = np.array([np.vstack((c_manip_input[0,:10],np.hstack((np.zeros(4),c_manip_input[1,10:]))))],)
                        
                        # set left and right rewards for manip input
                        tmp = [rewardfun(c_manip_input, x, reward_weights,  n_feat_itemA, n_feat_itemB, n_actions, n_input_nodes) for x in range(n_actions)]
                        l_val,r_val = tmp[0],tmp[1]
                       
                        # feed manipulated input and get PE, accuracy
                        l_predicted_val_manip = model(c_manip_left_input, training=False).numpy()[0][-1]
                        r_predicted_val_manip = model(c_manip_right_input, training=False).numpy()[0][-1]
                        predicted_vals_manip = np.array([l_predicted_val_manip[0],r_predicted_val_manip[0]])
                        
                        # FIX: also convert model_pre_manip predict calls
                        l_predicted_val_pre_manip = model_pre_manip(c_manip_left_input, training=False).numpy()[0][-1]
                        r_predicted_val_pre_manip = model_pre_manip(c_manip_right_input, training=False).numpy()[0][-1]
                        predicted_vals_pre_manip = np.array([l_predicted_val_pre_manip[0],r_predicted_val_pre_manip[0]])
                        
                        # choose action according to a soft-max rule
                        selected_action_manip,selected_action_wo_exploration_manip,p_manip = softmax_action_selection(predicted_vals_manip,temp)
                        selected_action_pre_manip,selected_action_wo_exploration_pre_manip,p_pre_manip = softmax_action_selection(predicted_vals_pre_manip,temp)
                        
                        # log PE for left, right (classify also as chosen and unchosen):
                        l_predicted_val, r_predicted_val = predicted_vals_manip[0], predicted_vals_manip[1]
                        
                        # compute PE and co.
                        orig_response = selected_action
                        response = selected_action_wo_exploration_manip
                        response_pre_manip = selected_action_wo_exploration_pre_manip
                        pe_left = l_val - l_predicted_val
                        pe_right = r_val - r_predicted_val
                        
                        if l_val>r_val:
                            correct_resp_manip=0
                        else:
                            correct_resp_manip=1
                        
                        if ((l_val>r_val) and (response==0)) or ((l_val<r_val) and (response==1)):
                            is_correct_manip = 1
                        else: 
                            is_correct_manip = 0
                            
                        if ((l_val>r_val) and (response_pre_manip==0)) or ((l_val<r_val) and (response_pre_manip==1)):
                            is_correct_pre_manip = 1
                        else: 
                            is_correct_pre_manip = 0
                         
                        if response==0: pe_choice = pe_left.copy()
                        else: pe_choice = pe_right.copy()
                        
                        # append rows to an empty DataFrame
                        new_row = {'sub': subj, 'trial': i_learning_trial,'original_input' : cinputs.copy(), 'input' : c_manip_input, 'manipulation' : manipulation, 
                                    'left_value': l_val,'right_value': r_val,
                                    'orig_response': orig_response, 'response': response,
                                    'correct_response_orig': correct_response_orig, 
                                    'correct_response_manip': correct_resp_manip, 
                                    'pe_left' : pe_left, 'pe_right' : pe_right, 'pe_choice' : pe_choice, 
                                    'is_correct' : is_correct_manip, 'is_correct_pre_feedback' : is_correct_pre_manip}
                        dfMass = pd.concat([dfMass, pd.DataFrame([new_row])], ignore_index=True)
                    
                # Print mean loss every 50 training trials:                       
                if (i_learning_trial > 0) and ((i_learning_trial % 50) == 0):  
                    print("Loss:", np.mean(losses[i_learning_trial-50:i_learning_trial]))
                    
            else: # a passive trial in empirical protocol - feedforward inputs and log activations
            
                if storePassiveActivation_empiricalSampling: 
                    i_passive_trial_empricalSampling = i_passive_trial_empricalSampling+1
                    
                    cinputs = inputs[i_trial,:,:]
                    left_passive_input, right_passive_input = np.array([np.vstack((cinputs[0,:10],cinputs[1,:10]))],),  np.array([np.vstack((cinputs[0,:10],np.hstack((np.zeros(4),cinputs[1,10:]))))],)
                    
                    # Log layer-specific activations for current passive trial:
                    activations_passive_empricialSampling[0][i_passive_trial_empricalSampling,:,:] = cinputs # log inputs (in orginal 2X16 format)
                    
                    c=0
                    for layer_idx, layer in enumerate(model.layers): # log subsequent layers' activations:
                        if (not 'input' in model.layers[layer_idx].name):
                            c+=1
                            l_cactivations_passive = get_activations(model, left_passive_input,layer_names=layer.name, auto_compile=False)
                            r_cactivations_passive = get_activations(model, right_passive_input, layer_names=layer.name, auto_compile=False)
                            activations_passive_empricialSampling[c][i_passive_trial_empricalSampling,:,:] = np.array(tf.concat([l_cactivations_passive[layer.name], r_cactivations_passive[layer.name]],axis=2))
                    
                    if passive_type[i_trial]==None:
                        print('hi')
                    new_row = {'sub': subj, 'block': block_num[i_trial],
                               'trial_type': passive_type[i_trial],
                               'input': cinputs.copy(), 'value': outputs[i_trial,0],
                               'state': states[i_trial]}

                    dfPassive_empricalSampling = pd.concat([dfPassive_empricalSampling, pd.DataFrame([new_row])], ignore_index=True)

        # Extra passive sweep after the final learning trial (see add_final_passive_sweep in the header).
        # Same as the passive_full sweep in the loop, saved at index i_learning_trial+1, i.e. after the
        # last weight update.
        if storePassiveActivation and add_final_passive_sweep:
            _final_idx = i_learning_trial + 1
            if _final_idx < activations_passive_full[0].shape[0]:
                c = 0
                for layer_idx, layer in enumerate(model.layers):
                    if (not 'input' in model.layers[layer_idx].name):
                        c += 1
                        for i_passive_input in range(passive_inputs_full.shape[-1]):
                            c_passive_input = passive_inputs_full[:,:,i_passive_input].copy()
                            activations_passive_full[0][_final_idx,:,:,i_passive_input] = c_passive_input.copy()
                            left_passive_input, right_passive_input = np.array([np.vstack((c_passive_input[0,:10],c_passive_input[1,:10]))],),  np.array([np.vstack((c_passive_input[0,:10],np.hstack((np.zeros(4),c_passive_input[1,10:]))))],)
                            l_cactivations_passive = get_activations(model, left_passive_input, layer_names=layer.name, auto_compile=False)
                            r_cactivations_passive = get_activations(model, right_passive_input, layer_names=layer.name, auto_compile=False)
                            activations_passive_full[c][_final_idx,:,:,i_passive_input] = np.array(tf.concat([l_cactivations_passive[layer.name], r_cactivations_passive[layer.name]],axis=2))
                print(f'  [final passive sweep] logged at index {_final_idx} '
                      f'(unlocks the last empirical miniblock)')

        #%% Save data for offline analysis   
        
        np.save(folder_name + '/reward.npy' , R_all, allow_pickle=True, fix_imports=True)  
        np.save(folder_name + '/pe.npy' , pe_all, allow_pickle=True, fix_imports=True) 
        np.save(folder_name + '/correct.npy' , correct, allow_pickle=True, fix_imports=True) 
        np.save(folder_name + '/correct_wo_exploration.npy' , correct_wo_exploration, allow_pickle=True, fix_imports=True) 
        np.save(folder_name + '/selected_action_wo_exploration_all.npy' , selected_action_wo_exploration_all, allow_pickle=True, fix_imports=True) 
        np.save(folder_name + '/selected_action_all.npy' , selected_action_all, allow_pickle=True, fix_imports=True) 
        np.save(folder_name + '/R_all_options.npy' , R_all_options, allow_pickle=True, fix_imports=True)
        np.save(folder_name + '/losses.npy' , losses, allow_pickle=True, fix_imports=True)
        for activation_idx, activation_name in enumerate(activation_names):
            np.save(folder_name + '/activations_' + activation_name + '.npy' , activations[activation_idx], allow_pickle=True, fix_imports=True)  
        
        if doActiveTrialManipulationMass: 
            dfMass.to_csv(folder_name + '/active_trials_mass_manipulations'  +  '.csv', index=False)

        if storePassiveActivation:
            for activation_idx, activation_name in enumerate(activation_names):  
                np.save(folder_name + '/activations_passive_' + activation_name + '.npy' , activations_passive[activation_idx], allow_pickle=True, fix_imports=True)
                np.save(folder_name + '/activations_passive_full_' + activation_name + '.npy' , activations_passive_full[activation_idx], allow_pickle=True, fix_imports=True)
                
        if storePassiveActivation_empiricalSampling: 
            for activation_idx, activation_name in enumerate(activation_names):  
                np.save(folder_name + '/activations_passive_empiricalSampling_' + activation_name + '.npy' , activations_passive_empricialSampling[activation_idx], allow_pickle=True, fix_imports=True)
            dfPassive_empricalSampling.to_csv(folder_name + '/passiveTrials_empiricalSampling_metaData.csv', index=False)
            
        if is_fitting_opt:
            np.save(folder_name + '/responses.npy' , responses_ppt, allow_pickle=True, fix_imports=True) 
            ppt_resp = responses_ppt[isPassive==0]
            rnn_resp = rnn_resp[:len(ppt_resp)]
            choices_p = choices_p[:len(ppt_resp),:]
            return (rnn_resp, ppt_resp, choices_p)

    finally:
        # Always runs, also when an exception is raised.
        # Deletes the model and all sub-models so TF releases memory before the
        # subprocess exits (and definitely before any gc.collect call).
        try:
            if 'activation_models' in locals():
                for m in activation_models.values():
                    del m
                del activation_models
        except Exception:
            pass
        try:
            del model
        except Exception:
            pass
        tf.keras.backend.clear_session()
        gc.collect()
