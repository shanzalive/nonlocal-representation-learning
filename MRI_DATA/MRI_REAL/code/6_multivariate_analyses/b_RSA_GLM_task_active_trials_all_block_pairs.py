#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# RSA-GLM on the learning (active) trials: does an ROI encode the latent states?
#
# For each participant and ROI, correlates the activity patterns of single learning trials across all
# pairs of blocks (within early and within late learning) and labels each trial pair (same state, same
# context, shared irrelevant features, value difference, ...). The regression of these correlations on
# the model regressors (RSA-GLM) is done in b_RSA_GLM_state_encoding_Figure3.py.
# Input:  ROI patterns from a_extract_ROI_signals.py, summary_files_behaviour/learners_dict.json
# Output: <ROI>_corr_results_active_trials_all_pairs_collapsed_sides_zscored.csv
#         (one csv per ROI, one row per trial pair and participant; read by b_RSA_GLM_state_encoding_Figure3.py)
#
# Author: Shany Grossman

import os
import json
import numpy as np
import pandas as pd
from scipy.stats import zscore
from scipy.stats import ttest_1samp, ttest_rel, sem
import glob

#%% Analysis-specific params
analysis = 'RSA_GLM_task'
ROIs_to_analyze =  ['VISUALFUNC_HIGHORDER', 'FFA', 'OFC', 'HPC', '3&4VENTRICLES']
RUNs_to_analyze = ['active1', 'active2','active3', 'active4','active5', 'active6','active7', 'active8','active9', 'active10', 'active11']
states =  ['a1a2', 'a1a3', 'a1a4', 'a2a1', 'a2a3', 'a2a4', 'a3a1', 'a3a2',
       'a3a4', 'a4a1', 'a4a2', 'a4a3', 'b1b2', 'b1b3', 'b1b4', 'b2b1',
       'b2b3', 'b2b4', 'b3b1', 'b3b2', 'b3b4', 'b4b1', 'b4b2', 'b4b3']

trial_phase = 'house'
collapse_sides = True # if true - a1a2 and a2a1 are same_state, and we add another regrssor of 'choice' - 0 for left, 1 for right
zscore_per_run = True  # if True: z-score each voxel across trials within its run before pattern extraction


#%% Paths
base_dir = '/home/mpib/grossman/STATE_FORMATION/MRI_DATA/MRI_REAL/ROI_timeseries_LSS_5mm_smoothing_dropout_30_physio_reg_out_updated/'
save_dir = base_dir + analysis + '/'
os.makedirs(save_dir, exist_ok=True)

_zscore_suffix = '_zscored' if zscore_per_run else ''
for f in glob.glob(os.path.join(save_dir, f"*corr_results_active_trials_*{_zscore_suffix}.csv")):
    if _zscore_suffix == '' and f.endswith('_zscored.csv'):
        continue  # never delete zscored files when running without zscore
    os.remove(f)

#%% Participants: full + partial learners of learners_dict.json (written by plot_accuracy.py)
def _find_summary_files_behaviour():
    d = os.path.dirname(os.path.abspath(__file__))
    while not os.path.isdir(os.path.join(d, 'summary_files_behaviour')):
        if os.path.dirname(d) == d:
            raise FileNotFoundError('summary_files_behaviour/ not found above ' + __file__)
        d = os.path.dirname(d)
    return os.path.join(d, 'summary_files_behaviour') + os.sep
SUMMARY_DIR = _find_summary_files_behaviour()

with open(SUMMARY_DIR + 'learners_dict.json') as f:
    learners_dict = json.load(f)
subjects = np.sort(learners_dict['full_and_partial'])

# Define blocks
early_blocks = ['active1', 'active2', 'active3', 'active4', 'active5',  'active6']
late_blocks  = ['active6', 'active7', 'active8', 'active9', 'active10', 'active11']

# Define a helper function to get all unique block pairs within a list
def get_all_block_pairs(block_list):
    pairs = []
    for i in range(len(block_list)):
        for j in range(i + 1, len(block_list)):
            pairs.append((block_list[i], block_list[j]))
    return pairs

# Get all within-period block pairs
early_pairs = get_all_block_pairs(early_blocks)
late_pairs = get_all_block_pairs(late_blocks)

# Merge into one list with a "period" label
block_pairs = [(a, b, 'early') for a, b in early_pairs] + [(a, b, 'late') for a, b in late_pairs]

#%% define helper functions

# add 2ndary left/right house feature to log
def extract_secondary_feature(img_str):
    try:
        # Split by '_' and get the second index (3rd item)
        val = img_str.split('_')[2]
        # Convert "-1" -> 0, "1" -> 1
        return 0 if val == '-1' else 1 if val == '1' else None
    except Exception:
        return None
    
# standartize sate labels to ignore left/right positions:
def standardize_state_name(s):
    if not isinstance(s, str) or len(s) != 4:
        return s  # Return unchanged if it doesn't fit the expected pattern
    
    part1, part2 = s[:2], s[2:]
    # Sort the two parts alphabetically/numerically
    standardized = ''.join(sorted([part1, part2]))
    return standardized

#%% Collect correlations into a grand df 

for isub, sub in enumerate(subjects):
    data_path = os.path.join(base_dir, sub + '_LSS.npz')
    if not os.path.exists(data_path):
        print(f"Skipping {sub} - no data dict file...")
        continue

    df = pd.read_csv(os.path.join(base_dir, sub + '.csv'))
    df['secondary_lhouse_feature'] = df['lhouse_img'].apply(extract_secondary_feature)
    df['secondary_rhouse_feature'] = df['rhouse_img'].apply(extract_secondary_feature)

    data_dict = np.load(data_path)
    
    rois_this_sub = list(ROIs_to_analyze)

    for iROI, ROI in enumerate(rois_this_sub):
        print(sub)
        
        # one CSV per ROI
        out_csv = os.path.join(
            save_dir,
            f'{ROI}_corr_results_active_trials_all_pairs_collapsed_sides{_zscore_suffix}.csv'
            if collapse_sides else
            f'{ROI}_corr_results_active_trials_all_pairs{_zscore_suffix}.csv')

        if data_dict[ROI + '_active1_client_passive'].size!=0:
            
            rows = []
            for run, run_next, period_label in block_pairs:
    
                key = f"{ROI}_{run}_{trial_phase}_active"
                key_next = f"{ROI}_{run_next}_{trial_phase}_active"
    
                if not (key in data_dict and key_next in data_dict):
                    print(f"Skipping {sub} {ROI} - data missing for {run}/{run_next}")
                    continue
    
                data = data_dict[key]
                data_next = data_dict[key_next]
                if zscore_per_run:
                    data      = zscore(data,      axis=0, nan_policy='omit')
                    data_next = zscore(data_next, axis=0, nan_policy='omit')
    
                log = df[(df.runType == run) &
                         (df.event_type == trial_phase + ' onset') &
                         (df.trial_type == 'active')].reset_index(drop=True)
                log_next = df[(df.runType == run_next) &
                              (df.event_type == trial_phase + ' onset') &
                              (df.trial_type == 'active')].reset_index(drop=True)
    
                # Initialize pattern and feature storage
                patterns_run = np.full((len(states), data.shape[-1]), np.nan)
                patterns_next_run = np.full((len(states), data_next.shape[-1]), np.nan)
                irrel_client_feature_run, irrel_client_feature_next_run = [], []
                irrel_lhouse_feature_run, irrel_lhouse_feature_next_run = [], []
                irrel_rhouse_feature_run, irrel_rhouse_feature_next_run = [], []
                secondary_rhouse_feature_run, secondary_rhouse_feature_next_run = [], []
                secondary_lhouse_feature_run, secondary_lhouse_feature_next_run = [], []
    
                for istate, state in enumerate(states):
                    # Current run
                    idxs = log[(log['state'] == state)].index.to_numpy()
                    assert len(idxs) < 2
                    if len(idxs) == 1:
                        idxs = idxs[0]
                        irrel_client_feature_run.append(log.loc[idxs].irrelevant_client)
                        irrel_lhouse_feature_run.append(log.loc[idxs].irrlevant_house_left)
                        irrel_rhouse_feature_run.append(log.loc[idxs].irrelevant_house_right)
                        secondary_rhouse_feature_run.append(log.loc[idxs].secondary_rhouse_feature)
                        secondary_lhouse_feature_run.append(log.loc[idxs].secondary_lhouse_feature)
                        patterns_run[istate, :] = data[idxs, :]
                    else:
                        patterns_run[istate, :] = np.nan
    
                    # Next run
                    idxs_next = log_next[(log_next['state'] == state)].index.to_numpy()
                    assert len(idxs_next) < 2
                    if len(idxs_next) == 1:
                        idxs_next = idxs_next[0]
                        irrel_client_feature_next_run.append(log_next.loc[idxs_next].irrelevant_client)
                        irrel_lhouse_feature_next_run.append(log_next.loc[idxs_next].irrlevant_house_left)
                        irrel_rhouse_feature_next_run.append(log_next.loc[idxs_next].irrelevant_house_right)
                        secondary_rhouse_feature_next_run.append(log_next.loc[idxs_next].secondary_rhouse_feature)
                        secondary_lhouse_feature_next_run.append(log_next.loc[idxs_next].secondary_lhouse_feature)
                        patterns_next_run[istate, :] = data_next[idxs_next, :]
                    else:
                        patterns_next_run[istate, :] = np.nan
    
                # Correlate all state pairs across block pair
                for istate, state_from in enumerate(states):
                    for jstate, state_to in enumerate(states):
                        same_state_and_side = 1 if state_from == state_to else 0
                        if collapse_sides:
                            state_from = standardize_state_name(state_from)
                            state_to = standardize_state_name(state_to)
    
                        same_state = 1 if state_from == state_to else 0
                        same_context = 1 if state_from[0] == state_to[0] else 0
                        same_house_pair = 1 if (state_from[1] == state_to[1]) and (state_from[3] == state_to[3]) else 0
    
                        same_irrel_client_feature = 1 if irrel_client_feature_run[istate] == irrel_client_feature_next_run[jstate] else 0
                        same_irrel_left_house_feature = 1 if irrel_lhouse_feature_run[istate] == irrel_lhouse_feature_next_run[jstate] else 0
                        same_irrel_right_house_feature = 1 if irrel_rhouse_feature_run[istate] == irrel_rhouse_feature_next_run[jstate] else 0
                        same_secondary_left_house_feature = 1 if secondary_lhouse_feature_run[istate] == secondary_lhouse_feature_next_run[jstate] else 0
                        same_secondary_right_house_feature = 1 if secondary_rhouse_feature_run[istate] == secondary_rhouse_feature_next_run[jstate] else 0
    
                        same_pseudo_state = 0
                        if not same_state and same_irrel_client_feature and same_irrel_left_house_feature and same_irrel_right_house_feature and same_secondary_left_house_feature and same_secondary_right_house_feature:
                            same_pseudo_state = 1
    
                        corr_val = np.corrcoef(patterns_run[istate, :], patterns_next_run[jstate, :])[0, 1]
     
                        rows.append({
                            'subject': sub,
                            'ROI': ROI,
                            'block_from': run,
                            'block_to': run_next,
                            'period': period_label,
                            'state_from': state_from,
                            'state_to': state_to,
                            'same_state': same_state,
                            'same_state_and_side': same_state_and_side,
                            'same_pseudo_state': same_pseudo_state,
                            'same_context': same_context,
                            'same_house_pair': same_house_pair,
                            'same_irrel_client_feature': same_irrel_client_feature,
                            'same_irrel_left_house_feature': same_irrel_left_house_feature,
                            'same_irrel_right_house_feature': same_irrel_right_house_feature,
                            'correlation': corr_val })

                del patterns_run, patterns_next_run
                del irrel_client_feature_run, irrel_client_feature_next_run
                del irrel_lhouse_feature_run, irrel_lhouse_feature_next_run
                del irrel_rhouse_feature_run, irrel_rhouse_feature_next_run
                del secondary_lhouse_feature_run, secondary_lhouse_feature_next_run
                del secondary_rhouse_feature_run, secondary_rhouse_feature_next_run
                
            if len(rows) > 0:
                df_out = pd.DataFrame(rows)
            
                df_out.to_csv(
                    out_csv,
                    mode='a',
                    header=not os.path.exists(out_csv),
                    index=False
                )
            
            rows.clear()

import gc
gc.collect()

print(f"Saved")


