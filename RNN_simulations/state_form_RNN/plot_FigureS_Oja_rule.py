#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Supplementary Figure S7: networks trained with Oja's rule on each participant's curriculum
# (held-out seed), laid out similarly to Figure 7 
#
#
# Data: coupdateAnalysis/localLearningRules/oja/oja_{empirical,dense}.npy
# (computed by coupdate_analysis_RNN.py with simulations = ['oja']).
#
# Author: Shany Grossman

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
from scipy.stats import sem, ttest_rel

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


# Paths 
basedir        = '/Users/grossman/Volumes/STATE_FORMATION/RNN_simulations/'
output_basedir = basedir + 'state_form_RNN_outputs/coupdateAnalysis/localLearningRules/'
sim            = 'oja'
sim_dir        = os.path.join(output_basedir, sim)
emp_file       = os.path.join(sim_dir, f'{sim}_empirical.npy')
dense_file     = os.path.join(sim_dir, f'{sim}_dense.npy')
# Participant behaviour (identical for every simulation), same source as plot_fitted_RNNs.py
ppt_dir        = (basedir + 'state_form_RNN_outputs/NN_activations/gradientDescentPerfromanceMatched/'
                  'grid_search_fits/acc_curve_mse_w15_fc_gain_rnn_gain_rl_temp_1_greedy_choice_10_seeds_9_grid_points/')
oja_data_dir   = basedir + 'state_form_RNN_outputs/NN_activations/localLearningRules/heldOutSeed/oja'
save_dir       = '/Users/grossman/Projects/STATE_FORMATION/FIGURES/PAPER_FIGURES/'
out_name       = 'FigureS7_Oja_rule.pdf'

# Subjects 
# load learner json, then make an order-preserving intersection with the
# subjects in the Oja simulation (sub520 is missing)
with open(SUMMARY_DIR + 'learners_dict.json') as _f:
    _learners_dict = json.load(_f)
included_subjects = [s.replace('-', '') for s in sorted(_learners_dict['full_and_partial'])]
_oja_subjects = set(f[:-4] for f in os.listdir(oja_data_dir)
                     if f.endswith('.npz') and not f.startswith('._'))
subjects = [s for s in included_subjects if s in _oja_subjects]
print(f'{len(subjects)} subjects; not simulated: {sorted(set(included_subjects) - set(subjects))}')

# Load data
results_emp   = np.load(emp_file,   allow_pickle=True).item()
results_dense = np.load(dense_file, allow_pickle=True).item()
assert results_emp['RNN_hidden1']['same_context_coup_all'].shape[0] == len(subjects)

# Settings (as in Figure 7) 
layers_ts        = ['RNN_hidden1', 'hidden2']   # solid, dotted
base_same        = '#5BC8C0'
base_diff        = 'darkblue'
RNN_col          = '#a70b0b'                    
PPT_col          = '#372b2b'
DENSE_SMOOTH_WIN = 10                           # running average, dense only (as in Figure 7)
ACC_WIN          = 15                           # accuracy smoothing window (as in Figure 7b)
N_LEARN_TRIALS   = 264
LAST_BLOCKS_N    = 24 * 3                       # end-of-learning window for the rnns vs participants accuracy stat

context_panel_type = ('Context feature',
                      'same_context_coup_all', 'diff_context_coup_all')
irrel_panel_type   = ('Irrelevant context feature',
                      'same_irrel_context_coup_all', 'diff_irrel_context_coup_all')
columns = [('Empirical sampling', results_emp,   True),
           ('Dense sampling',     results_dense, False)]

plt.rcParams.update({
    'figure.dpi': 150,
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

def _no_leading_zero(v, _):
    # 2 decimals when the 2nd decimal matters (e.g. .25), else 1
    s = f'{v:.2f}' if round(v, 1) != round(v, 2) else f'{v:.1f}'
    if s.startswith('0.'):  return s[1:]
    if s.startswith('-0.'): return '-' + s[2:]
    return s
_fmt_nolz = FuncFormatter(_no_leading_zero)

def _smooth_running(arr, w):
    """Centered moving average along axis=1, nan-safe (same as Figure 7)."""
    if w <= 1:
        return arr
    arr = np.asarray(arr, dtype=float)
    out = np.full_like(arr, np.nan)
    half = w // 2
    n = arr.shape[1]
    for i in range(n):
        out[:, i] = np.nanmean(arr[:, max(0, i - half):min(n, i + half + 1)], axis=1)
    return out

def moving_average(data, window_size=ACC_WIN):
    """Forward moving average, as in plot_fitted_RNNs.py (Figure 7b)."""
    data = np.asarray(data, dtype=float)
    valid = np.where(~np.isnan(data))[0]
    if len(valid) == 0:
        return np.full(len(data), np.nan)
    n = valid[-1] + 1
    end = max(0, n - 15)
    result = np.full(len(data), np.nan)
    result[:end] = [np.nanmean(data[i:min(i + window_size, n)]) for i in range(end)]
    return result

# Participant accuracy (same derivation as plot_fitted_RNNs.py) 
ppt_acc_all = np.full((len(subjects), N_LEARN_TRIALS), np.nan)
for i, sub in enumerate(subjects):
    resp = np.load(ppt_dir + sub + '_ppt_resp_fit.npy', allow_pickle=True)
    corr = np.argmax(np.load(ppt_dir + sub + '/R_all_options.npy', allow_pickle=True), axis=1)[:N_LEARN_TRIALS]
    n = min(len(resp), len(corr))
    ppt_acc_all[i, :n] = ((resp[:n] == 'right').astype(int) == corr[:n]).astype(float)

# Oja network accuracy: per-trial % correct
rnn_acc_all = results_emp['RNN_hidden1']['acc_learning_single_trials'][:, :N_LEARN_TRIALS] / 100.

# End-of-learning stat (last 3 learning blocks), paired across subjects (for the caption)
_rnn_end, _ppt_end = [], []
for i in range(len(subjects)):
    n = int(np.sum(~np.isnan(ppt_acc_all[i])))
    _rnn_end.append(np.nanmean(rnn_acc_all[i, n - LAST_BLOCKS_N:n]))
    _ppt_end.append(np.nanmean(ppt_acc_all[i, n - LAST_BLOCKS_N:n]))
_rnn_end, _ppt_end = np.array(_rnn_end), np.array(_ppt_end)
_d = _rnn_end - _ppt_end
_t, _p = ttest_rel(_rnn_end, _ppt_end)
print(f'[accuracy, last {LAST_BLOCKS_N} trials] Oja {100*_rnn_end.mean():.1f}% vs participants '
      f'{100*_ppt_end.mean():.1f}%; diff = {100*_d.mean():.1f}% ± {100*_d.std(ddof=1):.1f}% (SD), '
      f't({len(_d)-1}) = {_t:.2f}, p = {_p:.3g}')

# Panels 
def draw_accuracy(ax):
    for acc, col, lab in [(ppt_acc_all, PPT_col, 'Participants'), (rnn_acc_all, RNN_col, 'Oja network')]:
        sm = np.vstack([moving_average(acc[j] * 100) for j in range(acc.shape[0])])
        m  = np.nanmean(sm, axis=0)
        e  = np.nanstd(sm, axis=0) / np.sqrt(np.sum(~np.isnan(sm), axis=0))
        x  = np.arange(len(m))
        ax.plot(x, m, color=col, lw=1.5, label=lab)
        ax.fill_between(x, m - e, m + e, color=col, alpha=0.25, linewidth=0)
    ax.axhline(50, color='k', ls='--', lw=0.8)
    ax.set_xlim(0, N_LEARN_TRIALS - ACC_WIN)
    ax.set_ylim(40, 100)
    ax.set_yticks([50, 75, 100])
    ax.set_xticks(np.arange(0, N_LEARN_TRIALS, 50))
    ax.set_xlabel('Learning trial #', fontsize=9)
    ax.set_ylabel('Accuracy\n[% correct]', fontsize=9)
    ax.tick_params(labelsize=7)
    ax.spines[['top', 'right']].set_visible(False)

def draw_ts(ax, results, panel_type, doEmp, ylabel=None, show_xlabel=True):
    _, same_key, diff_key = panel_type
    x = None
    for layer in layers_ts:
        same, diff = results[layer][same_key], results[layer][diff_key]
        x = np.arange(1, np.sum(~np.isnan(same[0, :])) + 1)
        same, diff = same[:, :x[-1]], diff[:, :x[-1]]
        if not doEmp:
            same, diff = _smooth_running(same, DENSE_SMOOTH_WIN), _smooth_running(diff, DENSE_SMOOTH_WIN)
        ls = ':' if layer == 'hidden2' else '-'
        for arr, col in [(same, base_same), (diff, base_diff)]:
            m, e = np.nanmean(arr, axis=0), sem(arr, axis=0, nan_policy='omit')
            ax.plot(x, m, color=col, lw=2, ls=ls)
            ax.fill_between(x, m - e, m + e, color=col, alpha=0.2, linewidth=0)
    ax.axhline(0, ls='--', lw=0.8, color='k')
    ymin, ymax = ax.get_ylim()
    max_abs = max(abs(ymin), abs(ymax))
    ax.set_ylim(-max_abs, max_abs)
    ax.set_xlim(1, x[-1])
    ax.set_xticks(np.arange(2, 21, 2) if doEmp else np.arange(50, x[-1] + 1, 50))
    if show_xlabel:
        ax.set_xlabel('Learning segment #' if doEmp else 'Learning trial #', fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=7)
    ax.yaxis.set_major_formatter(_fmt_nolz)
    ax.spines[['top', 'right']].set_visible(False)

# Figure 
fig = plt.figure(figsize=(7.2, 7.0))
outer = GridSpec(3, 2, figure=fig, height_ratios=[1, 1.4, 1.4], hspace=0.75, wspace=0.4)

ax_acc = fig.add_subplot(outer[0, 0])
draw_accuracy(ax_acc)

# Co-update legend in the free top-right cell
ax_leg = fig.add_subplot(outer[0, 1])
ax_leg.axis('off')
ax_leg.legend(handles=ax_acc.get_legend_handles_labels()[0] + [
    Line2D([0], [0], color=base_same, lw=2, ls='-', label='Same context, RNN layer'),
    Line2D([0], [0], color=base_diff, lw=2, ls='-', label='Cross-context, RNN layer'),
    Line2D([0], [0], color=base_same, lw=2, ls=':', label='Same context, FC layer'),
    Line2D([0], [0], color=base_diff, lw=2, ls=':', label='Cross-context, FC layer')],
    fontsize=7, frameon=False, loc='center')

for col_idx, (col_title, res_col, doEmp) in enumerate(columns):
    ax_rel = fig.add_subplot(outer[1, col_idx])
    draw_ts(ax_rel, res_col, context_panel_type, doEmp,
            ylabel='Co-update [r]\n(context feature)' if col_idx == 0 else None)
    ax_rel.set_title(col_title, fontsize=11, fontweight='bold', pad=8)

    ax_irr = fig.add_subplot(outer[2, col_idx])
    draw_ts(ax_irr, res_col, irrel_panel_type, doEmp,
            ylabel='Co-update [r]\n(irrelevant context feature)' if col_idx == 0 else None)

os.makedirs(save_dir, exist_ok=True)
out_path = os.path.join(save_dir, out_name)
fig.savefig(out_path, dpi=600, transparent=True, bbox_inches='tight', pad_inches=0.05)
fig.savefig(out_path.replace('.pdf', '.png'), dpi=600, bbox_inches='tight', pad_inches=0.05)
plt.show()
print(f'Saved -> {out_path}')
