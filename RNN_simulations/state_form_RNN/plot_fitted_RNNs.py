#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Figure 7b and Fig. S5: accuracy of the performance-matched networks vs the participants.
#
# Input:  fits saved by fit_RNN_to_participants.py, summary_files_behaviour/
# Output: Figure7b.pdf (group mean) and FigureS5_Fitted_RNNs_acc_gallery.png (every participant)
#
# Author: Shany Grossman

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import os.path
import json

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


# === Plot params ===
# Two evaluation modes:
#   'heldout'          : the held-out seed withheld from parameter selection (paper, Fig. 7b/S5)
#   'mean_train_seeds' : mean (+/- SEM) over the 10 seeds used to select the fitted parameters
seed_eval_mode = 'heldout'
use_mean_seeds = seed_eval_mode == 'mean_train_seeds'
_seed_suffix = {'heldout': '', 'mean_train_seeds': '_meanTrainSeeds'}[seed_eval_mode]
RNN_col = '#a70b0b'
PPT_col =  '#372b2b'
plot_PPT = True
plot_NN = True
ncol_plot=7
win_size=15
example_subjects = ['sub-558', 'sub-561']  # single subject examples to present in main figure (7B)

# Paths
data_dir = '/Users/grossman/Volumes/STATE_FORMATION/RNN_simulations/state_form_RNN_outputs/NN_activations/gradientDescentPerfromanceMatched/grid_search_fits/acc_curve_mse_w15_fc_gain_rnn_gain_rl_temp_1_greedy_choice_10_seeds_9_grid_points/'
save_dir = '/Users/grossman/Projects/STATE_FORMATION/FIGURES/PAPER_FIGURES/'
if use_mean_seeds:
    seeds_dir    = data_dir.rstrip('/') + '_all_train_seeds'
    seed_subdirs = sorted([d for d in os.listdir(seeds_dir)
                           if d.startswith('seed_') and not d.startswith('.')])
    print(f'seed_eval_mode={seed_eval_mode}: found {len(seed_subdirs)} seed dirs in {seeds_dir}')

# Subjects (learners_dict.json, as in the fMRI scripts)
learner_type     = 'full_and_partial'
_ldict_path      = SUMMARY_DIR + 'learners_dict.json'
with open(_ldict_path) as _f:
    learners_dict = json.load(_f)
subj = np.array(sorted([s for s in learners_dict[learner_type]]))
print(f'Including {len(subj)} subjects ({learner_type}).')

# === Plot settings ===
plt.rcParams.update({
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'font.family': 'Arial',
    'pdf.fonttype': 42,
    'font.size': 8,
    'axes.labelsize': 10,
    'axes.titlesize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.linewidth': 1,
    'xtick.major.width': 1,
    'ytick.major.width': 1,
    'xtick.major.size': 3,
    'ytick.major.size': 3})

# %% Plot gallery of accuracies (per participant)


def moving_average(data, window_size=20):
    data = np.asarray(data)
    valid = np.where(~np.isnan(data))[0]

    if len(valid) == 0:
        return np.full(len(data), np.nan)

    n = valid[-1] + 1
    end = max(0, n - 15)

    result = np.full(len(data), np.nan)
    result[:end] = [
        np.nanmean(data[i:min(i + window_size, n)])
        for i in range(end)
    ]

    return result


def _seed_mean_sem_smoothed(seed_correct_list, window_size):
    """Smooth each seed, then compute mean and SEM across seeds."""
    smoothed = np.stack([moving_average(c, window_size) for c in seed_correct_list])
    mean = np.nanmean(smoothed, axis=0)
    sem = np.nanstd(smoothed, axis=0, ddof=1) / np.sqrt(len(seed_correct_list))
    return mean, sem


fig, ax = plt.subplots(
    int(np.ceil(len(subj)/ncol_plot)),
    ncol_plot,
    sharex='col',
    sharey='row',
    figsize=(7.2,6))

fig.tight_layout(pad=1.2, rect=[0.04, 0.04, 0.95, 0.95])
fig.subplots_adjust(hspace=1.1)
fig.text(0.5, 0.02, 'Learning trial #', ha='center', fontsize=12)
fig.text(0.02, 0.5, 'Accuracy [% correct]', va='center', rotation='vertical', fontsize=12)
ax = ax.flatten()

rnn_acc_all     = np.full((len(subj), 300), np.nan)
rnn_acc_sem_all = np.full((len(subj), 300), np.nan)  # filled only when use_mean_seeds
ppt_acc_all     = np.full((len(subj), 300), np.nan)
params          = np.zeros((len(subj), 4))

legend_handles = None  # will store handles once

for i, sub in enumerate(subj):

    sub = sub.replace('-', '')

    # Check whether the subject's data exist
    fitted_params_path = data_dir + sub + '_fitted_params.npy'
    if not os.path.isfile(fitted_params_path):
        continue


    if use_mean_seeds:
        # RNN accuracy: mean across the 10 training seeds (each smoothed first)
        seed_correct_list = []
        for sd in seed_subdirs:
            correct_path = os.path.join(seeds_dir, sd, sub, 'correct.npy')
            if os.path.isfile(correct_path):
                seed_correct_list.append(np.load(correct_path, allow_pickle=True))
        if not seed_correct_list:
            continue
        rnn_acc, rnn_acc_sem = _seed_mean_sem_smoothed(seed_correct_list, win_size)
        _valid_mask = ~np.isnan(rnn_acc)
        rnn_acc     = rnn_acc[_valid_mask]
        rnn_acc_sem = rnn_acc_sem[_valid_mask]

        if plot_PPT:
            ppt_resp = np.load(data_dir + sub + '_ppt_resp_fit.npy', allow_pickle=True)
            ppt_resp_int = np.zeros(len(ppt_resp))
            ppt_resp_int[ppt_resp == 'right'] = 1
            tmp = np.load(data_dir + sub + '/R_all_options.npy', allow_pickle=True)
            correct_resp = np.argmax(tmp, axis=1)
            if correct_resp.shape[0] > 264:
                correct_resp = correct_resp[:264]
            ppt_acc = ppt_resp_int == correct_resp[:len(ppt_resp_int)]
    else:
        if not os.path.isfile(data_dir + sub + '_rnn_resp_fit.npy'):
            continue
        rnn_resp = np.load(data_dir + sub + '_rnn_resp_fit.npy', allow_pickle=True)
        rnn_resp_int = np.zeros(len(rnn_resp))
        rnn_resp_int[rnn_resp == 'righ'] = 1
        if len(rnn_resp_int) == 1:
            continue

        if plot_PPT:
            ppt_resp = np.load(data_dir + sub + '_ppt_resp_fit.npy', allow_pickle=True)
            ppt_resp_int = np.zeros(len(ppt_resp))
            ppt_resp_int[ppt_resp == 'right'] = 1

        tmp = np.load(data_dir + sub + '/R_all_options.npy', allow_pickle=True)
        correct_resp = np.argmax(tmp, axis=1)
        if correct_resp.shape[0] > 264:
            correct_resp = correct_resp[:264]
        if rnn_resp_int.shape[0] < correct_resp.shape[0]:
            correct_resp = correct_resp[:rnn_resp_int.shape[0]]

        rnn_acc = rnn_resp_int == correct_resp
        # smooth with the same moving_average window as the participant curves
        rnn_acc = moving_average(rnn_acc.astype(float), win_size)
        if plot_PPT:
            ppt_acc = ppt_resp_int == correct_resp[:len(ppt_resp_int)]


    params[i, :] = np.load(fitted_params_path, allow_pickle=True)

    # ---- Plot RNN ----
    if plot_NN:
        rnn_acc_all[i, :len(rnn_acc)] = rnn_acc
        # rnn_acc is already smoothed above; rescale to percent.
        rnn_smooth = rnn_acc * 100
        line_rnn, = ax[i].plot(rnn_smooth, color=RNN_col, linewidth=1, label='Neural network')
        if use_mean_seeds:
            rnn_acc_sem_all[i, :len(rnn_acc_sem)] = rnn_acc_sem
            ax[i].fill_between(np.arange(len(rnn_smooth)),
                               rnn_smooth - rnn_acc_sem * 100,
                               rnn_smooth + rnn_acc_sem * 100,
                               color=RNN_col, alpha=0.25, linewidth=0)

    # ---- Plot Participants ----
    if plot_PPT:
        ppt_acc_all[i, :len(ppt_acc)] = ppt_acc
        line_ppt, = ax[i].plot(
            np.convolve(ppt_acc, np.ones(win_size)/win_size, mode='valid')*100,
            color=PPT_col,
            linewidth=1,
            label='Participant')

    # Save legend handles only once
    if legend_handles is None:
        legend_handles = []
        if plot_NN:
            legend_handles.append(line_rnn)
        if plot_PPT:
            legend_handles.append(line_ppt)

    # ---- Titles ----
    if sub.replace('sub', 'sub-') in learners_dict['full']:
        c_sub_color = 'black'
    elif sub.replace('sub', 'sub-') in learners_dict['partial']:
        c_sub_color = 'black'
    else:
        c_sub_color = 'red'

    ax[i].set_title(sub.replace('sub', 'S'), fontsize=7, color=c_sub_color, y=0.9)

    # ---- Ticks ----
    ax[i].set_xticks(np.arange(0, len(rnn_acc)+1, 100))
    ax[i].set_yticks(np.arange(0, 101, 25))

    # Force x tick labels to appear in ALL axes
    ax[i].tick_params(axis='x', labelbottom=True)

    # ---- Spines formatting ----
    ax[i].spines[['right', 'top']].set_visible(False)
    ax[i].spines['bottom'].set_linewidth(0.8)
    ax[i].spines['left'].set_linewidth(0.8)

    ax[i].tick_params(axis='both', width=0.8, length=3, labelsize=6)

    ax[i].axhline(50, color='k', linestyle='--', linewidth=0.8)
    ax[i].set_ylim(35, 105)

    ax[i].spines['bottom'].set_position(('outward', 2))
    ax[i].spines['left'].set_position(('outward', 2))


# ---- Turn off unused axes ----
for j in range(len(subj), len(ax)):
    ax[j].set_axis_off()
empty_ax = ax[j-1]  # last empty one for legend


# ---- Single legend in empty bottom-right panel ----
if legend_handles is not None:
    leg = empty_ax.legend(
    legend_handles,
    [h.get_label() for h in legend_handles],
    loc='center',
    frameon=False,
    fontsize=12,
    handlelength=2.0)

    for line in leg.get_lines():
        line.set_linewidth(2)

fig.savefig(save_dir + f'FigureS5_Fitted_RNNs_acc_gallery{_seed_suffix}.png', bbox_inches='tight')

# %% Plot mean ACC curves + two example subjects (Figure 7B)

# Choose participants to show on the right 
fig_mean = plt.figure(figsize=(4.2, 1.5))

gs = gridspec.GridSpec(
    2, 2,
    width_ratios=[3, 2],   # 60% / 40%
    height_ratios=[1, 1],
    wspace=0.35,
    hspace=1.1)

# ---- Main mean axis (left, spans rows) ----
ax_mean = fig_mean.add_subplot(gs[:, 0])

# ---- Two example axes (right column) ----
ax_ex1 = fig_mean.add_subplot(gs[0, 1])
ax_ex2 = fig_mean.add_subplot(gs[1, 1])
example_axes = [ax_ex1, ax_ex2]

ax_mean.set_xlabel('Learning trial #', fontsize=9)
ax_mean.set_ylabel('Accuracy [% correct]', fontsize=9)

if plot_PPT:
    # mean: smooth each subject first, then average (smooth and mean commute, but this is cleaner)
    ppt_smoothed        = np.vstack([moving_average(ppt_acc_all[j]*100,win_size) for j in range(len(subj))])
    acc_ppt_mean_smooth = np.nanmean(ppt_smoothed, axis=0)
    acc_ppt_se_smooth   = np.nanstd(ppt_smoothed, axis=0) / np.sqrt(len(subj))

    ax_mean.plot(acc_ppt_mean_smooth, color=PPT_col, linewidth=1.5, label='Participants')
    ax_mean.fill_between(
        np.arange(len(acc_ppt_mean_smooth)),
        acc_ppt_mean_smooth - acc_ppt_se_smooth,
        acc_ppt_mean_smooth + acc_ppt_se_smooth,
        facecolor=PPT_col,
        alpha=0.2)

if plot_NN:
    
    rnn_smoothed = rnn_acc_all * 100
    acc_rnn_mean_smooth = np.nanmean(rnn_smoothed, axis=0)
    acc_rnn_se_smooth = np.nanstd(rnn_smoothed, axis=0) / np.sqrt(
    np.sum(~np.isnan(rnn_smoothed), axis=0))
    acc_rnn_se_smooth   = np.nanstd(rnn_smoothed, axis=0) / np.sqrt(len(subj))

    ax_mean.plot(acc_rnn_mean_smooth, color=RNN_col, linewidth=1.5, label='RNNs')
    ax_mean.fill_between(
        np.arange(len(acc_rnn_mean_smooth)),
        acc_rnn_mean_smooth - acc_rnn_se_smooth,
        acc_rnn_mean_smooth + acc_rnn_se_smooth,
        facecolor=RNN_col,
        alpha=0.2)

ax_mean.axhline(50, color='k', linestyle='--', linewidth=1)
ax_mean.spines['bottom'].set_position(('outward', 5))  # Move the y-axis spine slightly outward
ax_mean.spines['left'].set_position(('outward', 5))  # Move the y-axis spine slightly outward


ax_mean.set_yticks(np.arange(0, 101, 20))
ax_mean.set_ylim(40, 100.05)
ax_mean.set_xlim(0,264)

ax_mean.spines[['right','top']].set_visible(False)
plt.setp(ax_mean.spines.values(), linewidth=1)

ax_mean.tick_params(width=1, length=3, labelsize=8)


if plot_PPT and plot_NN:
    ax_mean.legend(
        loc='upper left',
        fontsize=9,
        frameon=False,
        handlelength=1.5)

# ---- INDIVIDUAL EXAMPLE PLOTS ----

for ax_ex, sub_label in zip(example_axes, example_subjects):

    if sub_label in subj:
        idx = list(subj).index(sub_label)

        # RNN (already smoothed in the loop above)
        if plot_NN:
            rnn_ex_smooth = rnn_acc_all[idx] * 100
            rnn_ex_smooth = rnn_ex_smooth[~np.isnan(rnn_ex_smooth)]
            ax_ex.plot(rnn_ex_smooth, color=RNN_col, linewidth=1)
            if use_mean_seeds:
                rnn_ex_sem = rnn_acc_sem_all[idx] * 100
                rnn_ex_sem = rnn_ex_sem[~np.isnan(rnn_ex_sem)]
                ax_ex.fill_between(np.arange(len(rnn_ex_smooth)),
                                   rnn_ex_smooth - rnn_ex_sem,
                                   rnn_ex_smooth + rnn_ex_sem,
                                   color=RNN_col, alpha=0.25, linewidth=0)

        # Participant
        if plot_PPT:
            ax_ex.plot(
                moving_average(ppt_acc_all[idx]*100),
                color=PPT_col,
                linewidth=1)

        ax_ex.set_title(sub_label.replace('sub-','Subject '), fontsize=8,  y=0.9)

    ax_ex.axhline(50, color='k', linestyle='--', linewidth=0.8)
    ax_ex.spines['bottom'].set_position(('outward', 3))  # Move the y-axis spine slightly outward
    ax_ex.spines['left'].set_position(('outward', 3))  # Move the y-axis spine slightly outward

    ax_ex.set_ylim(40, 100.5)
    ax_ex.set_xlim(0,264)
    ax_ex.set_yticks([40,60, 80, 100])

    ax_ex.spines[['right','top']].set_visible(False)
    plt.setp(ax_ex.spines.values(), linewidth=1)

    ax_ex.tick_params(width=1, length=3, labelsize=6)

    ax_ex.set_xlabel('')
    ax_ex.set_ylabel('')


fig_mean.savefig(save_dir + f'Figure7b{_seed_suffix}.pdf', bbox_inches='tight')
