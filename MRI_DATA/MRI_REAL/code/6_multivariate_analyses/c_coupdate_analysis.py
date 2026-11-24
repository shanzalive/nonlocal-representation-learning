#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Co-update analysis for one participant (first level).
#
# For every learning segment (mini-block), each state's change vector is computed as
# the post-learning minus the pre-learning pattern (estimation trials before and after).
# The co-update of two states is the Pearson correlation between their change vectors.
# Voxels are z-scored across trials within each run first (zscore_per_run = True).
# Input:  ROI patterns from a_extract_ROI_signals.py
# Output: <ROI folder>/coupdate_LSS/<sub>_coupdate_LSS_zscored.csv (one row per state pair and mini-block)
# doMultiverse = True runs the same analysis on the robustness variants.
# Run: c_coupdate_analysis_Tardis.sh (one job per participant)
#
# Author: Shany Grossman

import os
import glob
import numpy as np
import nilearn
from nilearn import image
from nilearn.interfaces.fmriprep import load_confounds
import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd
import sys
import scipy.stats as stats
from scipy.stats import zscore
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine
from sklearn.linear_model import LinearRegression
from scipy.linalg import orthogonal_procrustes


#%% Define analysis type
analysis = 'coupdate_LSS' #
zscore_per_run = True  # if True: z-score each voxel across trials within its run before pattern extraction
doMultiverse = False  # False = main pipeline (main text). True = robustness variants: loop over the folders below, OFC only
# Robustness variants of Fig. S3; each folder is an output of a_extract_ROI_signals.py with runMultiverse = True.
MULTIVERSE_UNIVERSES = [ ('ROI_timeseries_LSS_5mm_smoothing_dropout_30_compcorr', 'coupdate_LSS'),                # aCompCor instead of physio regressors
                        ('ROI_timeseries_LSS_5mm_smoothing_dropout_50_physio_reg_out_updated', 'coupdate_LSS'),   # dropout threshold 50%
                        ('ROI_timeseries_LSS_5mm_smoothing_dropout_80_physio_reg_out_updated', 'coupdate_LSS'),   # dropout threshold 80%
                        ('ROI_timeseries_LSS_5mm_smoothing_physio_reg_out_updated', 'coupdate_LSS'),              # no dropout threshold
                        ('ROI_timeseries_LSS_3mm_smoothing_dropout_30_physio_reg_out_updated', 'coupdate_LSS')]   # 3 mm smoothing


#%% Paths
base = '/home/mpib/grossman/'
if doMultiverse:
    base_dirs = [(base + 'STATE_FORMATION/MRI_DATA/MRI_REAL/' + d + '/', a) for d, a in MULTIVERSE_UNIVERSES]
elif 'LSS' in analysis:
    base_dirs = [(base + 'STATE_FORMATION/MRI_DATA/MRI_REAL/ROI_timeseries_LSS_5mm_smoothing_dropout_30_physio_reg_out_updated/', analysis)]
else:
    base_dirs = [(base + 'STATE_FORMATION/MRI_DATA/MRI_REAL/ROI_timeseries_zscore_5mm_smoothing_dropout_30/', analysis)]


#%% Get subject input
sub = sys.argv[1]
sub = sub.replace('sub5','sub-5')
print(sub)

#%% Loop over the output folder(s): the main pipeline, or the robustness variants if doMultiverse = True
for base_dir, analysis in base_dirs:
    print(f'--- universe: {base_dir} (analysis={analysis}) ---')
    #%% Load data for current subject 

    # Load ROI time series
    save_dir = base_dir + analysis + '/'
    os.makedirs(save_dir, exist_ok=True)

    npz_path = base_dir + sub + ('_LSS.npz' if 'LSS' in analysis else '.npz')
    csv_path = base_dir + sub + '.csv'
    if not (os.path.exists(npz_path) and os.path.exists(csv_path)):
        print(f"No data available for {sub} in {base_dir} -- skipping")
        continue

    # Load ROI time series
    data_dict = np.load(npz_path)

    # Load df with log info (trial type, TRs, etc):
    df = pd.read_csv(csv_path)


    # Get list of ROIs
    keys = list(data_dict.keys())
    ROIs = np.unique([k.split('_')[0]  for k in keys])
    ROIs_extended = np.unique(['_'.join(k.split('_')[:2]) for k in keys])
    if 'VISUALFUNC_HIGHORDER' in ROIs_extended:
        ROIs = np.unique(np.append(ROIs, 'VISUALFUNC_HIGHORDER'))

    if sub=='sub-528': #go back and find out why 528 doesn't have VISUALFUNC ROI@
        ROIs = ROIs[ROIs != 'VISUALFUNC']

    RUNS = np.unique([k.split('_')[1]  for k in keys])

    # if LSS, patterns are speprated by condition, get condition names, too 
    if 'LSS' in analysis:
        conditions = np.unique([k.split('_', 2)[-1]  for k in keys])

    def runPCA(data):
        # Run PCA
        pca = PCA()
        pca_projected= pca.fit_transform(data)  # Shape data: (n_trials, n_components)
        pca_projected = pca_projected.T #now shape returned: n_trials, n_componenets

        # Explained variance
        explained_variance = pca.explained_variance_ratio_

        return(pca_projected, explained_variance)

    #%% Iterate over ROIs

    # Initialize an empty df to store the correlations for current subject
    columns = ['subject', 'ses', 'run',  'runType', 'imini_block', 'ROI', 
               'state_from', 'state_to', 'same_irrel_client', 'same_primary_house_feature', 'same_secondary_house_feature',
               'value_state_from', 'value_state_to',
               'context_state_to', 'context_state_from',
               'same_client_img_state_from', 'same_client_img_state_to',
               'same_house_img_state_from', 'same_house_img_state_to',
               'feedbacked_state1', 'feedbacked_state2', 'feedbacked_state3', 'feedbacked_state4', 
               'feedbacked_state1_value', 'feedbacked_state2_value', 'feedbacked_state3_value', 'feedbacked_state4_value', 
               'feedbacked_state1_context', 'feedbacked_state2_context', 'feedbacked_state3_context', 'feedbacked_state4_context',
               'corr', 'corr_ordinal', 'corr_cosine',
               'euc_deriv_state_from', 'euc_deriv_state_to',
               'is_feedbacked_state_from', 'is_feedbacked_state_to', 'isMean', 'same_context',
               'temporal_dist_pre','temporal_dist_post', 'pre_corr', 'post_corr', 'acc_learning_trials_mini_block'] 
    corr_df = pd.DataFrame(columns=columns)

    for iROI, ROI in enumerate(ROIs):
        key = ROI + '_active1_client_passive' if 'LSS' in analysis else ROI + '_active1'
        if key not in data_dict:
            continue

        if data_dict[key].size != 0:
            for iRUN, RUN in enumerate(RUNS):
                if 'active' in RUN:
                    if int(RUN[6:])<12:

                        # Now generate a df with correlation value for single trials 
                        # Load this run's patterns
                        if 'LSS' in analysis: 
                            crun_data_client, crun_data_house = data_dict[ROI + '_' + RUN + '_client_passive'], data_dict[ROI + '_' + RUN + '_house_passive']

                            # Remove voxels that have beta = 0 in all trials
                            # Find columns that are NOT all zeros
                            non_zero_columns = ~(crun_data_client == 0).all(axis=0)

                            # Use boolean indexing to keep only the non-zero columns
                            crun_data_client = crun_data_client[:, non_zero_columns]
                            crun_data_house = crun_data_house[:, non_zero_columns]

                            if zscore_per_run:
                                crun_data_house  = zscore(crun_data_house,  axis=0, nan_policy='omit')
                                crun_data_client = zscore(crun_data_client, axis=0, nan_policy='omit')

                            nvoxels_cROI = crun_data_client.shape[-1]
                        else:
                            crun_data = data_dict[ROI + '_' + RUN]
                            nvoxels_cROI = crun_data.shape[-1]

                        #iterate over mini blocks (2 per run)
                        crun_df = df[df.runType==RUN]

                        # Find the indices where "prompt active" appears
                        prompt_active_indices = crun_df[crun_df['event_type'] == 'prompt active'].index
                        prompt_passive_indices = crun_df[crun_df['event_type'] == 'prompt passive'].index

                        #For later finding the trial number for retrieving it from the LSS maps if LSS in analysis:
                        idx_passive_trials_block = crun_df[(crun_df.trial_type=='passive') & (crun_df.event_type=='client onset')].index

                        if 'LSS' not in analysis:
                            # Build a (n_house_trials_in_run, n_voxels) array of single-TR house
                            # patterns, one row per passive trial in this run (same order as
                            # idx_passive_trials_block, so pre_idx_trial_num/post_idx_trial_num
                            # index into it directly). Same shape as crun_data_house (LSS), so we
                            # can z-score each voxel across trials the same way the LSS branch does.
                            house_onset_rows = idx_passive_trials_block + 1
                            house_trs = crun_df.loc[house_onset_rows, 'closest_tr'].values.astype(int)
                            crun_data_house_trials = crun_data[house_trs, :]
                            if zscore_per_run:
                                crun_data_house_trials = zscore(crun_data_house_trials, axis=0, nan_policy='omit')

                        for imini_block in range(2):
                            if imini_block==0:
                                cmini_block_df = crun_df.loc[:prompt_active_indices[1]]
                            else:
                                cmini_block_df = crun_df.loc[prompt_passive_indices[2]:]


                            # Find deriv for each pair of passive trials
                            idx_passive_trials = cmini_block_df[(cmini_block_df.trial_type=='passive') & (cmini_block_df.event_type=='client onset') ].index
                            n_passive_pairs = len(idx_passive_trials)//2

                            #Find trial number corresponding to each index for LSS analysis
                            idx_passive_trials_num = np.array([np.where(idx_passive_trials_block==idx)[0][0] for idx in idx_passive_trials])

                            state_passive, value_passive, context_passive, same_client_img, same_house_image, same_irrel_client, irrel_client = np.zeros(n_passive_pairs,dtype=object), np.zeros(n_passive_pairs), np.zeros(n_passive_pairs,dtype=object), np.zeros(n_passive_pairs), np.zeros(n_passive_pairs), np.zeros(n_passive_pairs), np.zeros(n_passive_pairs)
                            state_passive[:], value_passive[:], context_passive[:], same_client_img[:], same_house_image[:],same_irrel_client[:],irrel_client[:]= np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
                            primary_house_feature, secondary_house_feature = np.zeros(n_passive_pairs), np.zeros(n_passive_pairs)
                            primary_house_feature[:], secondary_house_feature[:] = np.nan, np.nan
                            state_passive_pre_timing, state_passive_post_timing = np.zeros(n_passive_pairs),np.zeros(n_passive_pairs)
                            state_passive_pre_timing[:], state_passive_post_timing[:] = np.nan, np.nan

                            deriv_passive, euc_dist_passive, pre_pattern, post_pattern = np.zeros((n_passive_pairs,nvoxels_cROI)), np.zeros(n_passive_pairs),np.zeros((n_passive_pairs,nvoxels_cROI)),np.zeros((n_passive_pairs,nvoxels_cROI))
                            deriv_passive[:], euc_dist_passive[:],pre_pattern[:], post_pattern[:] = np.nan, np.nan, np.nan, np.nan

                            idx_active_trials = cmini_block_df[(cmini_block_df.trial_type=='active') & (cmini_block_df.event_type=='client onset')].index
                            assert(len(idx_active_trials)==4)
                            active_trials_state, active_trials_value = np.zeros(4,dtype=object), np.zeros(4)
                            active_trials_state[:], active_trials_value[:] = np.nan, np.nan

                            # Compute accuracy in interleaving learning trials
                            acc_cmini_block = cmini_block_df.loc[idx_active_trials].is_correct.mean()


                            for i_passive_pair in range(n_passive_pairs):

                                pre_idx = idx_passive_trials[i_passive_pair]
                                pre_idx_trial_num = idx_passive_trials_num[i_passive_pair]

                                # State, value and context for current pair
                                state_passive[i_passive_pair] = cmini_block_df.loc[pre_idx].state
                                value_passive[i_passive_pair] = cmini_block_df.loc[pre_idx].phouse_reward
                                context_passive[i_passive_pair] = state_passive[i_passive_pair][0]
                                primary_house_feature[i_passive_pair] = cmini_block_df.loc[pre_idx].primary_house_passive
                                secondary_house_feature[i_passive_pair] = cmini_block_df.loc[pre_idx].secondary_house_passive

                                # Find the post feedback trial
                                post_idx = cmini_block_df[(cmini_block_df.state == state_passive[i_passive_pair] ) & (cmini_block_df.event_type=='client onset')].index[-1]
                                post_idx_trial_num = idx_passive_trials_num[np.where(idx_passive_trials==post_idx)][0]
                                #did both presentations had the same irrelevant client/house fetuare?
                                same_client_img[i_passive_pair] = 1 if cmini_block_df.loc[pre_idx].client_img==cmini_block_df.loc[post_idx].client_img else 0
                                same_house_image[i_passive_pair] = 1 if cmini_block_df.loc[pre_idx+1].phouse_img==cmini_block_df.loc[post_idx+1].phouse_img else 0
                                same_irrel_client[i_passive_pair] = 1 if cmini_block_df.loc[pre_idx].client_img.split('_')[2][0]==cmini_block_df.loc[post_idx].client_img.split('_')[2][0] else 0
                                assert(same_irrel_client[i_passive_pair]==1)
                                if cmini_block_df.loc[post_idx].client_img.split('_')[2][0]=='-':
                                    irrel_client[i_passive_pair]=0
                                else:
                                    irrel_client[i_passive_pair]=1

                                state_passive_pre_timing[i_passive_pair] = cmini_block_df.loc[pre_idx].time
                                state_passive_post_timing[i_passive_pair] = cmini_block_df.loc[post_idx].time

                                # Compute euc dist between pair of patterns

                                if 'LSS' in analysis: 
                                    pre_pattern[i_passive_pair,:] = crun_data_house[pre_idx_trial_num,:]
                                    post_pattern[i_passive_pair,:] = crun_data_house[post_idx_trial_num,:]
                                else:
                                    # Single closest TR only (closest_tr_2 is ~0.9s farther from the
                                    # event+5s target and could blend in the adjacent event's response,
                                    # given the ~3.3s client/house spacing).
                                    # Indexes into crun_data_house_trials (per-run, per-voxel z-scored
                                    # across trials, as in the LSS branch's zscore_per_run step)
                                    # rather than crun_data directly.
                                    pre_pattern[i_passive_pair,:] = crun_data_house_trials[pre_idx_trial_num,:]
                                    post_pattern[i_passive_pair,:] = crun_data_house_trials[post_idx_trial_num,:]

                                # Assume post_pattern and pre_pattern are both 1D arrays of the same shape
                                mean_pattern = (post_pattern + pre_pattern) / 2
                                # Demean each array
                                deriv_passive[i_passive_pair,:] = post_pattern[i_passive_pair,:]-pre_pattern[i_passive_pair,:]
                                euc_dist_passive[i_passive_pair] = np.linalg.norm(post_pattern - pre_pattern)

                                # end of filling in passive pairs

                            # Fill in identity of feedback state in active trials
                            for i_active_trial in range(4):
                                active_idx = idx_active_trials[i_active_trial]
                                if cmini_block_df.loc[active_idx].resp =='right': #CHECK
                                    feedbacked_state = cmini_block_df.loc[active_idx].state[:2]
                                elif cmini_block_df.loc[active_idx].resp =='left':
                                    feedbacked_state = cmini_block_df.loc[active_idx].state[2:]

                                else:
                                    feedbacked_state = 'none'
                                active_trials_state[i_active_trial] = feedbacked_state
                                active_trials_value[i_active_trial] = cmini_block_df.loc[active_idx].reward

                            print(active_trials_state)

                            # Procrustes align pre and post pattern for rotation (as in RNNs)
                            if 'procrustes' in analysis:

                                Q, scale = orthogonal_procrustes(pre_pattern.T, post_pattern.T)
                                pre_pattern = scale * (Q @ pre_pattern)


                            # Fill in fields of correlations between rep shifts (correlation between derivatives)
                            for i in range(len(state_passive)):
                                for j in range(i+1,len(state_passive)):

                                    isMean=0
                                    subject = cmini_block_df.iloc[0].subject
                                    ses = cmini_block_df.iloc[0].ses
                                    run = cmini_block_df.iloc[0].run
                                    runType = cmini_block_df.iloc[0].runType
                                    state_from, state_to = state_passive[i], state_passive[j]
                                    value_state_from, value_state_to = value_passive[i], value_passive[j]
                                    context_state_to, context_state_from = context_passive[i], context_passive[j]
                                    same_client_img_state_from, same_client_img_state_to = same_client_img[i], same_client_img[j]
                                    same_primary_house_feature = 1 if primary_house_feature[i]==primary_house_feature[j] else 0
                                    same_secondary_house_feature = 1 if secondary_house_feature[i]==secondary_house_feature[j] else 0
                                    same_irrel_client_coupdate = 1 if  irrel_client[i]==irrel_client[j] else 0
                                    same_house_img_state_from, same_house_img_state_to = same_house_image[i], same_house_image[j]
                                    feedbacked_state1, feedbacked_state2, feedbacked_state3, feedbacked_state4 =  active_trials_state[0], active_trials_state[1], active_trials_state[2], active_trials_state[3]
                                    feedbacked_state1_value, feedbacked_state2_value, feedbacked_state3_value, feedbacked_state4_value =  active_trials_value[0], active_trials_value[1], active_trials_value[2], active_trials_value[3]
                                    feedbacked_state1_context, feedbacked_state2_context, feedbacked_state3_context, feedbacked_state4_context = active_trials_state[0][0], active_trials_state[1][0], active_trials_state[2][0], active_trials_state[3][0]  
                                    corr =  np.corrcoef(deriv_passive[i], deriv_passive[j])[0,1]
                                    corr_ordinal =  spearmanr(deriv_passive[i], deriv_passive[j])[0]
                                    corr_cosine = cosine(deriv_passive[i], deriv_passive[j])
                                    euc_deriv_state_from, euc_deriv_state_to = euc_dist_passive[i], euc_dist_passive[j]
                                    is_feedbacked_state_from, is_feedbacked_state_to = state_from in active_trials_state, state_to in active_trials_state 
                                    same_context=1 if context_state_to == context_state_from else 0

                                    # Add here a measure of context sepration per mini-block
                                    pre_corr = np.corrcoef(pre_pattern[i,:],pre_pattern[j,:])[0,1]
                                    post_corr = np.corrcoef(post_pattern[i,:],post_pattern[j,:])[0,1]


                                    #fill in teporal dist between two pre patterns and two post patterns:
                                    temporal_dist_pre = np.abs(state_passive_pre_timing[i] - state_passive_pre_timing[j])
                                    temporal_dist_post = np.abs(state_passive_post_timing[i] - state_passive_post_timing[j])

                                    new_row = pd.DataFrame([[subject, ses, run, runType, imini_block, ROI, 
                                               state_from, state_to, same_irrel_client_coupdate,
                                               same_primary_house_feature, same_secondary_house_feature,
                                               value_state_from, value_state_to,
                                               context_state_to, context_state_from,
                                               same_client_img_state_from, same_client_img_state_to,
                                               same_house_img_state_from, same_house_img_state_to,
                                               feedbacked_state1, feedbacked_state2, feedbacked_state3, feedbacked_state4, 
                                               feedbacked_state1_value, feedbacked_state2_value, feedbacked_state3_value, feedbacked_state4_value, 
                                               feedbacked_state1_context, feedbacked_state2_context, feedbacked_state3_context, feedbacked_state4_context,
                                               corr, corr_ordinal,corr_cosine,
                                               euc_deriv_state_from, euc_deriv_state_to,
                                               is_feedbacked_state_from, is_feedbacked_state_to, isMean, same_context,
                                               temporal_dist_pre,temporal_dist_post,pre_corr,post_corr, acc_cmini_block
                                               ]],
                                               columns=corr_df.columns)

                                    corr_df = pd.concat([corr_df, new_row], ignore_index=True)

    # Save DF as csv              
    _zscore_suffix = '_zscored' if zscore_per_run else ''
    corr_df.to_csv(save_dir + sub + '_' + analysis + _zscore_suffix + '.csv', index=False)
    print('saved')


