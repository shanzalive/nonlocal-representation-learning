#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# RSA-GLM on the estimation (passive) trials: does an ROI encode the latent states?
#
# Same as b_RSA_GLM_task_active_trials_all_block_pairs.py, for the estimation trials: correlates the
# activity patterns of single estimation trials across all pairs of blocks (within early and within late
# learning) and labels each trial pair. The RSA-GLM itself is done in b_RSA_GLM_state_encoding_Figure3.py.
# Input:  ROI patterns from a_extract_ROI_signals.py, summary_files_behaviour/learners_dict.json
# Output: <ROI>_corr_results_passive_trials_all_2_all_all_pairs_zscored.csv
#         (one csv per ROI, one row per trial pair and participant; read by b_RSA_GLM_state_encoding_Figure3.py)
#
# Author: Shany Grossman

#%% Modules
import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist
import statsmodels.api as sm
from scipy.stats import zscore
import gc

#%% Analysis-specific params
analysis = 'RSA_GLM_task'
trial_type = 'passive' 
ROIs_to_analyze = ['VISUALFUNC_HIGHORDER', 'FFA', 'OFC', 'HPC', '3&4VENTRICLES']
states = ['a1','a2','a3','a4','b1', 'b2', 'b3', 'b4']
house_values = {'a1': 25, 'a2': 62, 'a3': 37, 'a4': 75,
                'b1': 100, 'b2': 25, 'b3': 75, 'b4': 0}
trial_phase = 'house'
analysis_type = 'all_2_all_all_pairs'
# Only used in the output file names. 'all_2_all': every single estimation trial of a state in one
# block is correlated with every single trial of a state in the other block (the first 2 presentations
# of each state per block); 'all_pairs': all pairs of blocks within early and within late learning.
# The time series in b_RSA_GLM_state_encoding_Figure3.py select the neighbouring block pairs from this file.
zscore_per_run = True  # if True: z-score each voxel across trials within its run before pattern extraction

#%% Paths
base_dir = '/home/mpib/grossman/STATE_FORMATION/MRI_DATA/MRI_REAL/ROI_timeseries_LSS_5mm_smoothing_dropout_30_physio_reg_out_updated/'
save_dir = base_dir + analysis + '/'
os.makedirs(save_dir, exist_ok=True)

_zscore_suffix = '_zscored' if zscore_per_run else ''
for f in glob.glob(os.path.join(save_dir, f"*corr_results_passive_trials_{analysis_type}{_zscore_suffix}.csv")):
    os.remove(f)


#%% Learners, loaded from learners classification saved by plot_accuracy.py
logs_dir = os.path.join(os.path.dirname(base_dir.rstrip('/')), 'LOGs')
import os
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
#%% Define early and late blocks
early_blocks = ['passive1', 'active1', 'active2', 'active3', 'active4','active5','active6'] 
late_blocks  = ['active6', 'active7', 'active8', 'active9', 'active10', 'active11','passive2'] #

#%% Collect correlations into a grand df
for isub, sub in enumerate(subjects):

    # Load data and log
    data_path = os.path.join(base_dir, sub + '_LSS.npz')
    if not os.path.exists(data_path):
        print(f"Skipping {sub} - no data dict file...")
        continue

    df = pd.read_csv(os.path.join(base_dir, sub + '.csv'))
    data_dict = np.load(data_path)
    if 'allROIs' in base_dir:
        keys = list(data_dict.keys())
        ROIs_to_analyze = np.unique([k.split('_')[0]  for k in keys])
        ROIs_extended = np.unique(['_'.join(k.split('_')[:2]) for k in keys])
        if 'VISUALFUNC_HIGHORDER' in ROIs_extended:
            ROIs_to_analyze = np.unique(np.append(ROIs_to_analyze, 'VISUALFUNC_HIGHORDER'))

    rois_this_sub = list(ROIs_to_analyze)

    for iROI, ROI in enumerate(rois_this_sub):
        print(sub)
        
        # one CSV per ROI
        out_csv = os.path.join(
            save_dir,
            f'{ROI}_corr_results_passive_trials_{analysis_type}{_zscore_suffix}.csv')
        
        if data_dict[ROI + '_active1_client_passive'].size!=0:

            # ---------- Run pairs: all pairs of blocks within early and within late learning ----------
            run_pairs = []
            for group_name, group in zip(["early", "late"], [early_blocks, late_blocks]):
                for i, run_a in enumerate(group):
                    for run_b in group[i + 1:]:  # unique pairs only
                        run_pairs.append((run_a, run_b, group_name))
    
            # ---------- Iterate through run pairs ----------
            rows = []
            for run, run_next, period in run_pairs:
                key = f"{ROI}_{run}_{trial_phase}_passive"
                key_next = f"{ROI}_{run_next}_{trial_phase}_passive"
                if not (key in data_dict and key_next in data_dict):
                    print(f"Skipping {sub} {ROI} - data missing for {run}/{run_next}")
                    continue
    
                # Load run data and logs
                data, data_next = data_dict[key], data_dict[key_next]
                if zscore_per_run:
                    data      = zscore(data,      axis=0, nan_policy='omit')
                    data_next = zscore(data_next, axis=0, nan_policy='omit')
    
                if trial_phase == 'client':
                    log = df[(df.runType == run) &
                             (df.event_type == trial_phase + ' onset') &
                             (df.trial_type == 'passive')].reset_index(drop=True)
                    log_next = df[(df.runType == run_next) &
                                  (df.event_type == trial_phase + ' onset') &
                                  (df.trial_type == 'passive')].reset_index(drop=True)
                else:
                    log = df[(df.runType == run) &
                             (df.event_type == f'single {trial_phase} onset')].reset_index(drop=True)
                    log_next = df[(df.runType == run_next) &
                                  (df.event_type == f'single {trial_phase} onset')].reset_index(drop=True)
    
                # ---------- Initialize pattern arrays ----------
                patterns_run = np.full((len(states), 8, data.shape[-1]), np.nan)
                patterns_next = np.full((len(states), 8, data_next.shape[-1]), np.nan)
                irrel_client_run, irrel_client_next =  np.full((len(states), 8), np.nan), np.full((len(states), 8), np.nan)
                irrel_house_run, irrel_house_next = np.full((len(states), 8), np.nan), np.full((len(states), 8), np.nan)
                secondary_house_run, secondary_house_next = np.full((len(states), 8), np.nan), np.full((len(states), 8), np.nan)
                # ---------- Extract state-specific patterns ----------
                for istate, state in enumerate(states):
                    idxs = log[log['state'] == state].index.to_numpy()
                    idxs_next = log_next[log_next['state'] == state].index.to_numpy()
                    if len(idxs) == 0 or len(idxs_next) == 0:
                        continue
    
                    patterns_run[istate, :len(idxs), :] = data[idxs]
                    patterns_next[istate, :len(idxs_next), :] = data_next[idxs_next]
                    
                    irrel_client_run[istate,:len(idxs)] = log.loc[idxs]['irrelevant_client']
                    irrel_client_next[istate,:len(idxs_next)] = log_next.loc[idxs_next]['irrelevant_client']

                    irrel_house_run[istate,:len(idxs)] = log.loc[idxs]['irrelevant_house_passive']
                    irrel_house_next[istate,:len(idxs_next)] = log_next.loc[idxs_next]['irrelevant_house_passive']

                    secondary_house_run[istate,:len(idxs)] = log.loc[idxs]['secondary_house_passive']
                    secondary_house_next[istate,:len(idxs_next)] = log_next.loc[idxs_next]['secondary_house_passive']


                # ---------- Correlate all state pairs ----------
                for istate, state_from in enumerate(states):
                    for jstate, state_to in enumerate(states):
    
                        same_state = int(state_from == state_to)
                        same_context = int(state_from[0] == state_to[0])
                        same_house_type = int(state_from[-1] == state_to[-1])
    
                        for irep in range(2):
                            for jrep in range(2):
                                same_irrel_client = int(irrel_client_run[istate, irep] == irrel_client_next[jstate, jrep])
                                same_irrel_house = int(irrel_house_run[istate, irep] == irrel_house_next[jstate, jrep])
                                same_secondary_house = int(secondary_house_run[istate, irep] == secondary_house_next[jstate, jrep])
                                same_pseudo_state = int(
                                    not same_state and same_irrel_client and same_irrel_house and same_secondary_house)
                                corr = np.corrcoef(
                                    patterns_run[istate, irep, :],patterns_next[jstate, jrep, :])[0, 1]

                                rows.append({
                                  'subject': sub,
                                  'ROI': ROI,
                                  'block': run,
                                  'next_block': run_next,
                                  'period': period,
                                  'state_from': state_from,
                                  'state_to': state_to,
                                  'same_state': same_state,
                                  'same_context': same_context,
                                  'same_house_type': same_house_type,
                                  'same_irrel_client_feature': same_irrel_client,
                                  'same_irrel_house_feature': same_irrel_house,
                                  'same_secondary_house_feature': same_secondary_house,
                                  'same_pseudo_state': same_pseudo_state,
                                  'abs_val_diff': abs(house_values[state_from] - house_values[state_to]),
                                  'correlation': corr })


                del patterns_run, patterns_next
                del irrel_house_run, irrel_house_next
                del irrel_client_run, irrel_client_next
                del secondary_house_run, secondary_house_next


            if len(rows) > 0:
                df_out = pd.DataFrame(rows)
            
                df_out.to_csv(
                    out_csv,
                    mode='a',
                    header=not os.path.exists(out_csv),
                    index=False
                )
            
            rows.clear()   
            gc.collect()
