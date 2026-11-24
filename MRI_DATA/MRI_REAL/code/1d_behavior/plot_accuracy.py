#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Classifies the participants into learner groups (the inclusion criterion of all analyses)
# and plots their learning curves.
#
# What it does:
# 1. Marks every trial as correct or incorrect (used here only, not saved): in learning trials
#    the correct choice is the more valuable house; in estimation trials it is "more" or "less"
#    than the probed value.
# 2. Classifies each participant by the number of errors in the last learning block
#    (active11, 24 trials):
#      full (complete) learners:  fewer than 3 errors (>= 91.7% correct)
#      partial learners:          3-4 errors (83.3-87.5% correct)
#      non-learners:              more than 4 errors (<= 79.2% correct)
#    sub-501 (old version of the estimation task) and sub-511 (motion outlier) are excluded
#    here; the analyses use the full and partial learners of learners_dict.json (N = 51).
# 3. Plots the mean accuracy over the learning trials and over the estimation trials
#    (smoothed over 20 trials), and a gallery with every participant's learning curve.
#
# Input:  summary_files_behaviour/State_Formation_behaviour.csv (from convert_logs_to_csv.py)
# Output: summary_files_behaviour/learners_dict.json (participants per learner group; read by all analyses),
#         learner_classification_active11.csv and the accuracy plots (save_dir).
#         State_Formation_behaviour.csv is only read, not changed (it is written by convert_logs_to_csv.py).
#
# Author: Shany Grossman

#%% Import modules

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import json
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import stats
from scipy.stats import ttest_rel


#%% Set params:
    
# What learner class to run on?
learner_type = 'all' #alternatives: 'full', 'partial', 'none', 'all'

# Log csv: path to a csv containing all logs for all subjects 
import os
def _find_summary_files_behaviour():
    d = os.path.dirname(os.path.abspath(__file__))
    while not os.path.isdir(os.path.join(d, 'summary_files_behaviour')):
        if os.path.dirname(d) == d:
            raise FileNotFoundError('summary_files_behaviour/ not found above ' + __file__)
        d = os.path.dirname(d)
    return os.path.join(d, 'summary_files_behaviour') + os.sep
SUMMARY_DIR = _find_summary_files_behaviour()

log_path = SUMMARY_DIR + 'State_Formation_behaviour.csv'

# Where to save CSVs and figures?
save_dir = '/Users/grossman/Volumes/STATE_FORMATION/MRI_DATA/MRI_REAL/LOGs/'

# Set color for learning and passive tirals plots
learning_trials_col = '#AB1866'
passive_trials_col = '#6CBA7D'

#%% Load csv with logs for all participants
log = pd.read_csv(log_path)  # Important  - this inclues only active blocks right now

# Add a column indicating whether subject was correct 
def add_correct_response_to_log(log):
    # First, initialize 'correct_response' as NaN
    log['correct_response'] = np.nan
    
    # --- Active trials ---
    # Define masks
    active = log['is_passive'] == 0
    left_better = active & (log['value_left'] > log['value_right'])
    right_better = active & (log['value_left'] < log['value_right'])
    
    # Assign correct responses for active
    log.loc[left_better, 'correct_response'] = 'left'
    log.loc[right_better, 'correct_response'] = 'right'
    
    # --- Passive trials ---
    passive = log['is_passive'] == 1
    more_valuable = passive & (log['value_single_house'] > log['single_house_prob_value'])
    less_valuable = passive & (log['value_single_house'] < log['single_house_prob_value'])
    no_probe = passive & (log['single_house_prob_value'])
    
    # Assign correct responses for passive
    log.loc[more_valuable, 'correct_response'] = 'more'
    log.loc[less_valuable, 'correct_response'] = 'less'
    
    return log

log = add_correct_response_to_log(log)
log['is_correct'] = log['correct_response']==log['response']
log.loc[log['correct_response'].isna(), 'is_correct'] = np.nan


#%% Compute inclusion criteria from active11 (last learning block)
# Criteria based on number of errors in active11 (24 trials):
#   complete:     < 3 errors (0-2 errors, >= 91.7% accuracy)
#   partial:      3-4 errors (83.3-87.5% accuracy)
#   unsuccessful: > 4 errors (<= 79.2% accuracy)

def classify_learners(log):
    all_subs = ['sub-501', 'sub-502', 'sub-503', 'sub-504', 'sub-505', 'sub-506', 'sub-507', 'sub-508', 'sub-509',
                'sub-510', 'sub-511', 'sub-512', 'sub-513', 'sub-514', 'sub-515', 'sub-516', 'sub-517', 'sub-518',
                'sub-519', 'sub-520', 'sub-521', 'sub-522', 'sub-523', 'sub-524', 'sub-525', 'sub-526',
                'sub-527', 'sub-528', 'sub-529', 'sub-530', 'sub-531', 'sub-532', 'sub-533', 'sub-534',
                'sub-535', 'sub-536', 'sub-537', 'sub-538', 'sub-539', 'sub-540', 'sub-541', 'sub-542',
                'sub-543', 'sub-544', 'sub-545', 'sub-546', 'sub-547', 'sub-548', 'sub-549', 'sub-550',
                'sub-551', 'sub-552', 'sub-553', 'sub-554', 'sub-555', 'sub-556', 'sub-557', 'sub-558',
                'sub-559', 'sub-560', 'sub-561', 'sub-562', 'sub-563', 'sub-564']
    rows = []
    for sub in all_subs:
        df = log[(log['sub'] == sub.replace('-', '')) & (log['block_type_num'] ==11) & (log['is_passive'] == 0)]
        n_errors = 24 - int(df['is_correct'].sum())
        acc = (24 - n_errors) / (24) *100
        if n_errors < 3:
            cat = 'complete'
        elif n_errors <= 4:
            cat = 'partial'
        else:
            cat = 'unsuccessful'
        rows.append({'sub': sub, 'n_errors': n_errors, 'accuracy': round(acc, 1), 'category': cat})
    return pd.DataFrame(rows)

learner_classification = classify_learners(log)
learner_classification.to_csv(save_dir + 'learner_classification_active11.csv', index=False)
print(learner_classification.to_string(index=False))

# Title color per subject for gallery plots
_cat_colors = {'complete': 'black', 'partial': 'orange', 'unsuccessful': 'red'}
subject_title_colors = {
    row['sub']: _cat_colors[row['category']]
    for _, row in learner_classification.iterrows()
}

#%% Define learner dict (data-driven from active11 errors):

# Participants excluded from all analyses. They are removed here, so learners_dict.json
# only contains the participants included in the paper (full + partial learners: N = 51).
EXCLUDED_PARTICIPANTS = {
    'sub-501': 'old version of the estimation task',
    'sub-511': 'motion outlier',
}
_included = learner_classification[~learner_classification['sub'].isin(EXCLUDED_PARTICIPANTS)]

_complete     = _included[_included['category'] == 'complete']['sub'].tolist()
_partial      = _included[_included['category'] == 'partial']['sub'].tolist()
_unsuccessful = _included[_included['category'] == 'unsuccessful']['sub'].tolist()

learners_dict = {
    'full':             _complete,
    'partial':          _partial,
    'none':             _unsuccessful,
    'full_and_partial': _complete + _partial,
    'all':              _included['sub'].tolist(),
}

# Save learners_dict as JSON for use in subsequent analyses
with open(SUMMARY_DIR + 'learners_dict.json', 'w') as f:
    json.dump(learners_dict, f, indent=2)



#%% Get list of participants:
sub_list = np.sort(learners_dict[learner_type])  
n_subjects, n_trials = len(sub_list) , 264

#%% Functions for plooting group mean curves:

def moving_average(data, window_size):
    data = np.asarray(data)
    result = np.full(len(data) - window_size + 1, np.nan)

    for i in range(len(result)):
        window = data[i:i+window_size]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            result[i] = np.mean(valid)
    return result

def plot_accuracy_with_se(df, title, color, save_dir=None):
    """
    Plots mean accuracy per trial with shaded SE and individual subject traces.

    Parameters:
    - df: DataFrame with subjects as rows, trials as columns
    - title: Title of the plot
    - color: Color for the mean line and SE fill
    - save_dir: Optional path to save the figure
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from scipy.stats import sem
    
    # Set global font
    rcParams['font.family'] = 'Arial'

    # Compute mean and SE, ignoring NaNs
    mean_accuracy = np.nanmean(df.values, axis=0)
    mean_accuracy = moving_average(mean_accuracy, window_size=20)

    se_accuracy = sem(df.values, axis=0, nan_policy='omit')
    se_accuracy = moving_average(se_accuracy, window_size=20)

    # Trials for x-axis
    trials = np.arange(1, len(mean_accuracy) + 1)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))


    # Plot mean accuracy and shaded SE
    ax.plot(trials, mean_accuracy, color=color, linewidth=3, label='Mean Accuracy')
    ax.fill_between(trials, 
                    mean_accuracy - se_accuracy, 
                    mean_accuracy + se_accuracy, 
                    color=color, alpha=0.3, label='±1 SE')

    # Horizontal dashed line at 0.5 (chance), matching x-limits
    ax.axhline(50, linestyle='--', color='black', linewidth=1.5,
               xmin=0, xmax=1)  # xmin/xmax are in [0, 1] relative coords

    # Axis limits and spine bounds
    ax.set_ylim(40, 100)
    ax.set_xlim(trials[0], trials[-1])

    # Only show left and bottom spines, adjust bounds and width
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_bounds(40, 100)

    # Offset bottom spine slightly
    ax.spines['bottom'].set_position(('outward', 10))

    # Labels and title
    ax.set_xlabel('Trial #', fontsize=16)
    ax.set_ylabel('Accuracy [% correct]', fontsize=16)
    ax.set_title(title, fontsize=16)

    # Tick styling
    ax.tick_params(axis='both', which='major', length=6, width=1.5, labelsize=14)

    # Save
    plt.tight_layout()
    plt.savefig(save_dir + title.replace(' ' , '_') + '.png' , dpi=600)


#%% Function for plotting individual gallery of learning curves

def plot_subject_gallery(df, title, color, save_dir, title_colors=None):
    """
    Plots a gallery of subject accuracy curves.

    Parameters:
    - df: DataFrame with subjects as rows, trials as columns
    - title: Overall title for the gallery
    - color: Line color for individual subject plots
    - save_dir: Optional path to save the figure
    - title_colors: optional dict mapping subject id -> title color
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams['font.family'] = 'Arial'

    n_subjects = len(df)
    n_cols = 7
    n_rows = int(np.ceil(n_subjects / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.2 * n_cols, 2.6 * n_rows),
        sharex=True, sharey=True
    )
    fig.subplots_adjust(wspace=1.2, hspace=1.2)  # <-- Increased spacing
    axes = axes.flatten()

    for i, (subj_id, row) in enumerate(df.iterrows()):
        smoothed = moving_average(row.values, window_size=20)
        trials = np.arange(1, len(smoothed) + 1)
        ax = axes[i]
        ax.plot(trials, smoothed, color=color, linewidth=3)

        # Horizontal dashed line at y = 50
        ax.axhline(50, linestyle='--', color='black', linewidth=1.2)

        title_col = title_colors.get(subj_id, 'black') if title_colors else 'black'
        ax.set_title(subj_id, fontsize=18, color=title_col)
        ax.set_ylim(30, 101)
        ax.set_xlim(trials[0], trials[-1])

        # Aesthetics: only left and bottom spines
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)

        # Add a "gap" on the left spine to indicate non-zero start
        ax.spines['left'].set_bounds(30, 100)

        ax.tick_params(axis='both', which='major', length=5, width=1.5, labelsize=16)

    # Remove unused axes if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(title, fontsize=24)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_dir + title.replace(' ', '_') + '.png', dpi=600)

#%% Extract, save as CSV and plot active trials accuray 

# Create empty array to store sub x trial correctness
data = np.zeros((n_subjects, n_trials))
data[:] = np.nan

for isub, sub in enumerate(sub_list):
    csub_log = log[(log['sub']==sub.replace('-','')) & (log['block_type_num']<12) & (log['is_passive']==0)].copy()
    print(sub + ': ' + str(csub_log.shape[0]) + ' active trials')
    
    data[isub,:csub_log.shape[0]] = np.array(csub_log.is_correct)*100

df = pd.DataFrame(data, index=sub_list)
df.to_csv(save_dir + 'State_Formation_accuracy_learning_trials.csv', index=False)

# Plot mean accuracy and gallery
plot_accuracy_with_se(df, 'Mean accuracy in learning trials (N=' + str(n_subjects) + ')', learning_trials_col, save_dir)

# Plot gallary of individual subjects
plot_subject_gallery(df, 'Individual learning curves [learning trials]', learning_trials_col, save_dir, title_colors=subject_title_colors)


#%% Extract, save as CSV and plot passive trials accuray 

# Create empty array to store sub x trial correctness
data_passive = np.zeros((n_subjects, 198))
data_passive[:] = np.nan

for isub, sub in enumerate(sub_list):
    csub_log = log[(log['sub']==sub.replace('-','')) & (log['block_type_num']<12) & (log['is_passive']==1)].copy()
    print(sub + ': ' + str(csub_log.shape[0]) + ' passive trials')
    
    data_passive[isub,:csub_log.shape[0]] = np.array(csub_log.is_correct)*100

# Create DataFrame with data and save it 
df_passive = pd.DataFrame(data_passive, index=sub_list)
df_passive.to_csv(save_dir + 'State_Formation_accuracy_estimation_trials.csv', index=False)

# Plot mean passive accuracy 
plot_accuracy_with_se(df_passive, 'Mean accuracy in estimation trials (N=' + str(n_subjects) + ')',passive_trials_col, save_dir)

# Plot gallary of individual subjects
plot_subject_gallery(df_passive, 'Individual accuracy curves [estimation trials]', passive_trials_col, save_dir, title_colors=subject_title_colors)
