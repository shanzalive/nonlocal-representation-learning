#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Figure 2: behavioural modelling (perceptual vs compressed model).
#
# Plots the participants' learning curves (learning and estimation trials) and compares the fit
# of the two models to the participants' choices (negative log-likelihood), using the fits of
# fit_lin_func_approx_models_grid_search.py for all participants.
# Input:  model fits (behavioral_modelling_outputs/), summary_files_behaviour/
# Output: Figure_2.pdf
#
# Author: Shany Grossman

import matplotlib.pyplot as plt
import numpy as np
import os.path
import pandas as pd
import seaborn as sns
import scipy
from scipy.stats import sem
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
    'ytick.major.size': 3,
})


# What learner class to run on?
learner_type = 'full_and_partial'  # alternatives: 'full', 'partial', 'none'
data_dir = '/Users/grossman/Volumes/STATE_FORMATION/RNN_simulations/behavioral_modelling_outputs/'
save_dir = '/Users/grossman/Projects/STATE_FORMATION/FIGURES/PAPER_FIGURES/'


_ldict_path = SUMMARY_DIR + 'learners_dict.json'
with open(_ldict_path) as _f:
    _learners_dict = json.load(_f)
included_subjects = sorted(_learners_dict[learner_type])
subj = np.array(included_subjects)
USE_FACE_MODEL = True  # True = use 'full_face' (64 states) as perceptual model; False = use 'full' (32 states)

_perceptual_model = 'full_face' if USE_FACE_MODEL else 'full'
_perceptual_label = 'Face model' if USE_FACE_MODEL else 'Full state model'

models = [_perceptual_model, 'context+house']
model_names = [_perceptual_label, 'True state model']
model_colors = ['#660033', '#E673AC'] #'navy', 'red', 'c'] #'b',
model_colors = ['#440154', '#FDE725'] #'navy', 'red', 'c'] #'b',


#%% Load behavioral log and dd a column indicating whether subject was correct
logs_df = pd.read_csv(SUMMARY_DIR + 'State_Formation_behaviour.csv')
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

logs_df = add_correct_response_to_log(logs_df)
logs_df['is_correct'] = logs_df['correct_response']==logs_df['response']
logs_df.loc[logs_df['correct_response'].isna(), 'is_correct'] = np.nan

#%% Compute mean learning curves  + SEM

SMOOTH_WIN = 15# window size for running average smoothing
def running_average_adaptive(x, window=5):
    """
    Compute running average with adaptive window at the edges.
    At the edges, the window size shrinks to include only available points.
    """
    x = np.array(x)
    y = np.full_like(x, np.nan, dtype=float)
    n = len(x)
    
    for i in range(n):
        start = max(0, i - window//2)
        end   = min(n, i + window//2 + 1)
        y[i] = np.nanmean(x[start:end])
    
    return y


subjects = np.sort(logs_df['sub'].unique())

# Max number of trials for alignment
max_trials_active = logs_df[(logs_df['is_passive'] == 0)].groupby('sub').size().max()
max_trials_passive = logs_df[(logs_df['is_passive'] == 1)].groupby('sub').size().max()

# Initialize arrays
active_acc = np.full((len(subjects), max_trials_active), np.nan)
passive_acc = np.full((len(subjects), max_trials_passive), np.nan)

# Fill arrays with individual trial accuracies
for i, sub in enumerate(subjects):
    act_trials = logs_df[(logs_df['sub'] == sub) & (logs_df['is_passive'] == 0) & (logs_df['block_type_num'] < 12)]
    active_acc[i, :len(act_trials)] = act_trials['is_correct'].astype(float).values

    pas_trials = logs_df[(logs_df['sub'] == sub) & (logs_df['is_passive'] == 1) & (logs_df['block_type_num'] < 12) & (logs_df['block_type_num'] < 12)]
    passive_acc[i, :len(pas_trials)] = pas_trials['is_correct'].astype(float).values

# -------------------------------
# Smooth per subject first, then compute mean and SEM across subjects
# -------------------------------
active_acc_smooth  = np.array([running_average_adaptive(active_acc[i],  SMOOTH_WIN) for i in range(len(subjects))])
passive_acc_smooth = np.array([running_average_adaptive(passive_acc[i], SMOOTH_WIN) for i in range(len(subjects))])

mean_active_smooth = np.nanmean(active_acc_smooth, axis=0)[:264] * 100
sem_active_smooth  = sem(active_acc_smooth, axis=0, nan_policy='omit')[:264] * 100

mean_passive_smooth = np.nanmean(passive_acc_smooth, axis=0)[:198] * 100
sem_passive_smooth  = sem(passive_acc_smooth, axis=0, nan_policy='omit')[:198] * 100

trials_active = np.arange(0, sem_active_smooth.shape[0])
trials_passive = np.arange(0, sem_passive_smooth.shape[0])


#%% Paired t-test: block 1 vs block 11 accuracy (learning and estimation)

from scipy.stats import ttest_rel as _ttest_rel


incl_subs_log = [s.replace('-', '') for s in included_subjects]

for trial_type, type_label in [(0, 'Learning (active)'), (1, 'Estimation (passive)')]:
    acc_b1, acc_b11 = [], []
    for sub in incl_subs_log:
        df_sub = logs_df[(logs_df['sub'] == sub) & (logs_df['is_passive'] == trial_type)]
        b1  = df_sub[df_sub['block_type_num'] == 1]['is_correct'].mean()
        b11 = df_sub[df_sub['block_type_num'] == 11]['is_correct'].mean()
        if not (np.isnan(b1) or np.isnan(b11)):
            acc_b1.append(b1 * 100)
            acc_b11.append(b11 * 100)

    acc_b1  = np.array(acc_b1)
    acc_b11 = np.array(acc_b11)
    t, p = _ttest_rel(acc_b1, acc_b11)
    p_str = 'p < .001' if p < 0.001 else f'p = {p:.3f}'
    print(f"\n{type_label}: block 1 vs block 11 (N={len(acc_b1)})")
    print(f"  Block  1: M = {acc_b1.mean():.1f}%,  SD = {acc_b1.std(ddof=1):.1f}%")
    print(f"  Block 11: M = {acc_b11.mean():.1f}%, SD = {acc_b11.std(ddof=1):.1f}%")
    print(f"  t({len(acc_b1)-1}) = {t:.2f}, {p_str}")


#%% ANALYSIS (RUN ONCE)

results = {}

nll_per_trial = np.full((len(models), len(subj), 264), np.nan)
nll_mean_smooth = {}
nll_se_smooth = {}
diff_mean_smooth = None
diff_se_smooth = None

acc_all_ppts = np.full(len(subj), np.nan)
df = []

inc_subj = [[] for _ in range(len(models))]

for imodel, model in enumerate(models):
    csub = -1
    for isub, sub in enumerate(subj):
        sub = sub.replace('-', '')
        fname = data_dir + model + '_grid_search_1seed_participant_tempOpt_softmax_lr_grid/' + sub + '_choices_p.npy'
        if not os.path.isfile(fname):
            continue
        csub_fitted_params = np.load(data_dir + model + '_grid_search_1seed_participant_tempOpt_softmax_lr_grid/' + sub + '_fitted_params.npy')
        csub += 1
        inc_subj[imodel].append(sub)

        ppt_resp = np.load(data_dir + model + '_grid_search_1seed_participant_tempOpt_softmax_lr_grid/' + sub + '_ppt_resp_fit.npy')
        choices_p = np.load(fname)

        # accuracy
        sub_log = logs_df[
            (logs_df['sub'] == sub) &
            (logs_df['is_passive'] == 0) &
            (logs_df['block_type_num'] < 12) &
            (logs_df['response'] != 'none')
        ]
        acc_all_ppts[csub] = 100 * sub_log['is_correct'].mean()

        # NLL per trial
        for t, a in enumerate(ppt_resp):
            if a != 1000:
                nll_per_trial[imodel, csub, t] = -np.log(choices_p[t, int(a)])

        df.append({
            "subject": sub,
            "model": model,
            "nll": np.nanmean(nll_per_trial[imodel, csub]),
            "learning_rate": csub_fitted_params[0]
        })

    # mean + SE (smoothed)
    mean = np.nanmean(nll_per_trial[imodel], axis=0)
    se = np.nanstd(nll_per_trial[imodel], axis=0) / np.sqrt(csub)

    nll_mean_smooth[model] = np.convolve(mean, np.ones(10)/10, mode='valid')
    nll_se_smooth[model] = np.convolve(se, np.ones(10)/10, mode='valid')

df = pd.DataFrame(df)

# diff traces (assuming model 0 vs 1)
diff = nll_per_trial[0] - nll_per_trial[1]
diff_mean_smooth = np.convolve(np.nanmean(diff, axis=0), np.ones(10)/10, mode='valid')
diff_se_smooth = np.convolve(np.nanstd(diff, axis=0) / np.sqrt(csub), np.ones(10)/10, mode='valid')

results = dict(
    df=df,
    acc=acc_all_ppts,
    nll_mean=nll_mean_smooth,
    nll_se=nll_se_smooth,
    diff_mean=diff_mean_smooth,
    diff_se=diff_se_smooth,
)

results.update(dict(
    diff=diff,                     # (n_sub, n_trials)
    nll_per_trial=nll_per_trial,   # (n_models, n_sub, n_trials), for re-smoothing at plot time
    inc_subj=inc_subj[0],
))


np.save(save_dir + "figure2_cache.npy", results, allow_pickle=True)


#%% PLOT FIGURE 2

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde, sem, ttest_rel, pearsonr
import seaborn as sns
from matplotlib.patches import Patch

# -------------------------------
# Plotting params
# -------------------------------
LABEL_FONTSIZE = 11
TICK_FONTSIZE = 9
ANNOT_FONTSIZE = 9
SWARM_JITTER = 0.05      # horizontal jitter for swarm points
KDE_BW = 0.3             # bandwidth adjustment for KDE
LINE_INSET = 0.25         # inset for connecting lines
KDE_OFFSET = 0.2         # horizontal shift outward
NLL_SMOOTH_WIN = 15     # smoothing window for NLL / ΔNLL traces (1 = no smoothing)
SHOW_DELTA_NLL_SIG = False  # True = gold vertical lines where ΔNLL > 0 (p < .05); False = hide

# -------------------------------
# Load cached results
# -------------------------------
results = np.load(save_dir + "figure2_cache.npy", allow_pickle=True).item()
df = results["df"]
acc_all_ppts = results["acc"]
diff = results["diff"]
nll_per_trial = results["nll_per_trial"]   # (n_models, n_sub, n_trials)

# Smooth per-subject NLL/ΔNLL first, then average and compute SEM
def _smooth_row(row, win=NLL_SMOOTH_WIN):
    out = np.full(len(row) - win + 1, np.nan)
    for i in range(len(out)):
        chunk = row[i:i+win]
        valid = chunk[~np.isnan(chunk)]
        if len(valid) >= max(win // 2, 1):
            out[i] = np.nanmean(chunk)
    return out

# Recompute NLL mean/SE: smooth per-subject first, then average and compute SEM
nll_mean, nll_se = {}, {}
for i, model in enumerate(models):
    _smoothed = np.array([_smooth_row(nll_per_trial[i][s]) for s in range(nll_per_trial[i].shape[0])])
    _n_valid  = np.sum(~np.isnan(_smoothed), axis=0)
    nll_mean[model] = np.nanmean(_smoothed, axis=0)
    nll_se[model]   = np.nanstd(_smoothed, axis=0, ddof=1) / np.sqrt(np.maximum(_n_valid, 1))

diff_smooth_sub = np.array([_smooth_row(diff[i]) for i in range(diff.shape[0])])
_n_valid        = np.sum(~np.isnan(diff_smooth_sub), axis=0)
diff_mean_plot  = np.nanmean(diff_smooth_sub, axis=0)
diff_se_plot    = np.nanstd(diff_smooth_sub, axis=0, ddof=1) / np.sqrt(np.maximum(_n_valid, 1))


# -------------------------------
# Figure layout
# -------------------------------
fig = plt.figure(figsize=(7.2, 5))

# -------------------------------
# GridSpec: 2 rows, 3 columns
# -------------------------------
gs = gridspec.GridSpec(
    2, 3,
    width_ratios=[0.5, 0.2, 0.4],
    height_ratios=[1, 1],
    hspace=0.35,
    wspace=0.5
)
fig.subplots_adjust(left=0.08, right=0.97, bottom=0.08, top=0.95)

# -------------------------------
# Row 1
# -------------------------------
ax_learning_curves = fig.add_subplot(gs[0, 0])
# ax_swarm now spans columns 1 and 2
ax_swarm = fig.add_subplot(gs[0, 2:3])

# -------------------------------
# Row 2
# -------------------------------
# Merge left two columns for NLL + ΔNLL
gs_bottom = gs[1, :1].subgridspec(
    2, 1, height_ratios=[2, 1], hspace=0.15)
ax_mean = fig.add_subplot(gs_bottom[0])
ax_diff = fig.add_subplot(gs_bottom[1])

# Scatter occupies 30% width (last column)
ax_corr = fig.add_subplot(gs[1, 2])

# -------------------------------------------------
# Row 1: Combined active + passive learning curves
# -------------------------------------------------
ax = ax_learning_curves  # reuse one axis

# Determine max length to normalize both curves to the same width
max_len = max(len(mean_active_smooth), len(mean_passive_smooth))

# Normalized x for plotting (0 to 1)
x_active = np.linspace(0, 1, len(mean_active_smooth))
x_passive = np.linspace(0, 1, len(mean_passive_smooth))

# --- Active (bottom axis) ---
ax.plot(x_active, mean_active_smooth, lw=2.5, color=[0/255,104/255,51/255], label="Learning trials")
ax.fill_between(x_active,
                mean_active_smooth - sem_active_smooth,
                mean_active_smooth + sem_active_smooth,
                color=[0/255,104/255,51/255], alpha=0.25, linewidth=0)

ax.set_ylabel("Accuracy [% correct]", fontsize=LABEL_FONTSIZE)
ax.set_xlabel("Learning trial #", fontsize=LABEL_FONTSIZE)

# --- Passive (top axis) ---
ax_top = ax.twiny()
ax_top.plot(x_passive, mean_passive_smooth, lw=2.5, color=[41/255,180/255,115/255], label="Estimation trials")
ax_top.fill_between(x_passive,
                    mean_passive_smooth - sem_passive_smooth,
                    mean_passive_smooth + sem_passive_smooth,
                    color=[41/255,180/255,115/255], alpha=0.25, linewidth=0)

ax_top.set_xlabel("Estimation trial #", fontsize=LABEL_FONTSIZE)

# -------------------------------------------------
# X-ticks mapping to actual trial numbers
# -------------------------------------------------
# Bottom axis (active)
active_tick_numbers = np.array([50, 100, 150, 200, 250])
active_tick_pos = (active_tick_numbers - 1) / (len(mean_active_smooth) - 1)
ax.set_xticks(active_tick_pos)
ax.set_xticklabels(active_tick_numbers, fontsize=TICK_FONTSIZE)

# Top axis (passive)
passive_tick_numbers = np.array([50, 100, 150])
passive_tick_pos = (passive_tick_numbers - 1) / (len(mean_passive_smooth) - 1)
ax_top.set_xticks(passive_tick_pos)
ax_top.set_xticklabels(passive_tick_numbers, fontsize=TICK_FONTSIZE)

# -------------------------------------------------
# Styling
# -------------------------------------------------
ax.set_xlim(0, 1)
ax_top.set_xlim(0, 1)
ax.set_ylim(45, 100)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax_top.spines["right"].set_visible(False)

# Small gaps between y-axis and top/bottom spines
ax.spines["left"].set_position(("outward", 6))
ax_top.spines["left"].set_position(("outward", 6))

# Horizontal dashed line at chance level
ax.axhline(50, color="k", linestyle="--", lw=1)
ax.text(0.78, 53, "chance", va="center", fontsize=TICK_FONTSIZE)

# Legend top-left without frame, include both lines
handles, labels = ax.get_legend_handles_labels()
handles_top, labels_top = ax_top.get_legend_handles_labels()
ax.legend(handles + handles_top, labels + labels_top, loc="upper left", fontsize=TICK_FONTSIZE, frameon=False)

# Make y-axis tick labels same size as x-axis
ax.tick_params(axis='y', labelsize=TICK_FONTSIZE)

# -------------------------------
# Row 1: half-violin + swarm panel
# -------------------------------
import seaborn as sns
from scipy.stats import sem, ttest_rel

x_full, x_true = 0, 1
palette = {_perceptual_model: model_colors[0], "true": model_colors[1]}

# Extract data
full_vals = df[df['model']==_perceptual_model]['nll'].dropna().values
true_vals = df[df['model']=='context+house']['nll'].dropna().values

# --- Half-violin function ---
def half_violin(ax, vals, center_x, side="left", color="k", alpha=0.4, scale=0.35):
    """
    Draw a half violin on ax at center_x
    side: 'left' or 'right'
    """
    if len(vals) < 2:
        return
    parts = ax.violinplot(vals, positions=[center_x], widths=scale,
                          showmeans=False, showmedians=False, showextrema=False)
    for pc in parts['bodies']:
        verts = pc.get_paths()[0].vertices
        m = verts[:,0].mean()
        # Clip left/right
        if side=="left":
            verts[:,0] = np.minimum(verts[:,0], center_x)
        else:
            verts[:,0] = np.maximum(verts[:,0], center_x)
        # Style
        pc.set_facecolor(color)
        pc.set_alpha(alpha)
        pc.set_edgecolor('k')
        pc.set_linewidth(0.8)

# Draw half-violins
half_violin(ax_swarm, full_vals, x_full , side="left", color=palette[_perceptual_model], alpha=0.4, scale=0.45)
half_violin(ax_swarm, true_vals, x_true , side="right", color=palette["true"], alpha=0.4, scale=0.45)

# --- Overlay swarm plot ---
df_swarm = df[df['model'].isin([_perceptual_model, 'context+house'])].copy()
df_swarm['x'] = df_swarm['model'].map({_perceptual_model: x_full, 'context+house': x_true})

sns.swarmplot(x='x', y='nll', hue='model', data=df_swarm, ax=ax_swarm,
              palette={_perceptual_model: model_colors[0], "context+house": model_colors[1]},
              size=3.5, edgecolor='k', linewidth=0.4, zorder=3,
              dodge=False, legend=False)

# --- Paired lines connecting points ---
for fv, tv in zip(full_vals, true_vals):
    ax_swarm.plot([x_full+LINE_INSET, x_true-LINE_INSET], [fv, tv],
                  color='gray', lw=0.6, alpha=0.6, zorder=2)

# --- Mean ± SEM ---
mean_full = np.nanmean(full_vals)
mean_true = np.nanmean(true_vals)
sem_full = sem(full_vals, nan_policy='omit')
sem_true = sem(true_vals, nan_policy='omit')

ax_swarm.errorbar(x_full, mean_full, yerr=sem_full, fmt='o', color='k', markersize=6,
                  capsize=6, elinewidth=1.2, zorder=6)
ax_swarm.errorbar(x_true, mean_true, yerr=sem_true, fmt='o', color='k', markersize=6,
                  capsize=6, elinewidth=1.2, zorder=6)
ax_swarm.plot([x_full, x_true], [mean_full, mean_true], color='k', lw=1, zorder=7)

# --- Paired t-test annotation outside axes ---
t_stat, p_val = ttest_rel(full_vals, true_vals)
_p_str_nll = 'p < .001' if p_val < 0.001 else f'p = {p_val:.3f}'
print(f"\nNLL paired t-test (perceptual vs compressed, N={len(full_vals)})")
print(f"  Perceptual model: M = {full_vals.mean():.3f}, SD = {full_vals.std(ddof=1):.3f}")
print(f"  Compressed model: M = {true_vals.mean():.3f}, SD = {true_vals.std(ddof=1):.3f}")
print(f"  t({len(full_vals)-1}) = {t_stat:.2f}, {_p_str_nll}")
if p_val<0.001:
    str_sig = '***'
    str_p = 'p < .001'
elif p_val<0.01:
    str_sig = '**'
    str_p = f'p = {p_val:.3f}'
elif p_val<0.05:
    str_sig = '*'
    str_p = f'p = {p_val:.3f}'
else:
    str_sig = 'n.s.'
    str_p = f'p = {p_val:.3f}'


# Axis limits based on data
y_min = min(full_vals.min(), true_vals.min())
y_max = max(full_vals.max(), true_vals.max())
y_margin = 0.02 * (y_max - y_min)
ax_swarm.set_ylim(y_min - y_margin, y_max + y_margin)
ax_swarm.set_xlim(-0.25, 1.25)

# Horizontal significance bar above axes
sig_y_frac = 1.1  # fraction above axes
ax_swarm.plot([x_full, x_true], [sig_y_frac]*2, transform=ax_swarm.get_xaxis_transform(),
              lw=1.2, color='k', clip_on=False)
# Vertical ticks
ax_swarm.plot([x_full, x_full], [sig_y_frac, sig_y_frac-0.025], transform=ax_swarm.get_xaxis_transform(),
              lw=1.2, color='k', clip_on=False)
ax_swarm.plot([x_true, x_true], [sig_y_frac, sig_y_frac-0.025], transform=ax_swarm.get_xaxis_transform(),
              lw=1.2, color='k', clip_on=False)
# Stars and p-value
ax_swarm.text((x_full+x_true)/2, sig_y_frac-0.03, str_sig, ha='center', va='bottom', fontsize=LABEL_FONTSIZE+6,
              transform=ax_swarm.get_xaxis_transform())
ax_swarm.text((x_full+x_true)/2, sig_y_frac-0.08, str_p , ha='center', va='bottom', fontsize=ANNOT_FONTSIZE,
              transform=ax_swarm.get_xaxis_transform())

# --- Axes formatting ---
ax_swarm.set_xticks([x_full, x_true])
ax_swarm.tick_params(labelsize=TICK_FONTSIZE)
ax_swarm.set_xticklabels(["Perceptual\nmodel", "Compressed\nmodel"], fontsize=LABEL_FONTSIZE, fontweight='bold')
ax_swarm.set_ylabel("Negative log likelihood [NLL]\nbetter fit << >> worse fit", fontsize=LABEL_FONTSIZE)
ax_swarm.set_xlabel("")
ax_swarm.spines["top"].set_visible(False)
ax_swarm.spines["right"].set_visible(False)


# -------------------------------
# Row 2: Panel C (mean NLL)
# -------------------------------
x_trials = np.arange(1, len(nll_mean[_perceptual_model])+1)

# Plot each model
lines = []
labels = []
for i, model in enumerate(nll_mean):
    line, = ax_mean.plot(x_trials, nll_mean[model], lw=2.5, color=model_colors[i])
    ax_mean.fill_between(x_trials,
                         nll_mean[model]-nll_se[model],
                         nll_mean[model]+nll_se[model],
                         color=model_colors[i], alpha=0.25, linewidth=0)
    lines.append(line)
    labels.append(model.replace("_"," ").title())

# Add a dummy line for ΔNLL to show in legend only
from matplotlib.lines import Line2D
dummy_line = Line2D([0], [0], color='k', lw=2.5)
lines.append(dummy_line)
labels.append("Δ NLL")

# Add legend
ax_mean.legend(lines, labels, fontsize=TICK_FONTSIZE, frameon=False, loc="upper left")

ax_mean.set_ylabel("NLL", fontsize=LABEL_FONTSIZE)
ax_mean.tick_params(labelsize=TICK_FONTSIZE)
ax_mean.spines["top"].set_visible(False)
ax_mean.spines["right"].set_visible(False)
ax_mean.set_xlim(1, len(nll_mean[_perceptual_model]))
ax_mean.set_xticks([])  # no x-ticks

# -------------------------------
# Row 2: Panel D (Δ NLL)
# -------------------------------
x_trials_diff = np.arange(1, len(diff_mean_plot)+1)
ax_diff.plot(x_trials_diff, diff_mean_plot, lw=2.5, color='k')
ax_diff.fill_between(x_trials_diff,
                     diff_mean_plot - diff_se_plot,
                     diff_mean_plot + diff_se_plot,
                     color='k', alpha=0.25, linewidth=0)
ax_diff.axhline(0, color='k', ls='--', lw=1)

# One-sample t-tests on smoothed per-subject ΔNLL; mark where > 0 at p < 0.05
from scipy.stats import ttest_1samp as _ttest1s

n_tp = diff_smooth_sub.shape[1]
sig_mask = np.zeros(n_tp, dtype=bool)
for _t in range(n_tp):
    _vals = diff_smooth_sub[:, _t]
    _vals = _vals[~np.isnan(_vals)]
    if len(_vals) > 1:
        _, _p = _ttest1s(_vals, 0, alternative='greater')
        sig_mask[_t] = _p < 0.05

if SHOW_DELTA_NLL_SIG:
    for _t in np.where(sig_mask)[0]:
        ax_diff.axvline(x_trials_diff[_t], color='gold', alpha=0.4, lw=1, zorder=0)

ax_diff.set_xlabel("Learning trial #", fontsize=LABEL_FONTSIZE)
ax_diff.set_ylabel("Δ NLL", fontsize=LABEL_FONTSIZE)
ax_diff.tick_params(labelsize=TICK_FONTSIZE)

# --- Fix y-ticks as requested ---
ax_diff.set_yticks([0.0, 0.2])

# Optional: nicer x-ticks
ax_diff.set_xticks(np.arange(0, len(diff_mean_plot)+1, 50))
ax_diff.set_xlim(1, len(diff_mean_plot))

ax_diff.spines["top"].set_visible(False)
ax_diff.spines["right"].set_visible(False)


# -------------------------------
# Row 2: scatter (Δ NLL vs accuracy)
# -------------------------------

# Compute ΔNLL per subject (e.g., pre − post)
delta_nll = (df.groupby("subject")["nll"].apply(lambda x: x.iloc[0] - x.iloc[1]).values)

# Scatter
ax_corr.scatter(delta_nll, acc_all_ppts, s=20, color='k')
ax_corr.set_ylim(60, 100)

# Regression line
sns.regplot(
    x=delta_nll,
    y=acc_all_ppts,
    ax=ax_corr,
    scatter=False,
    ci=95,
    color='k',
    line_kws={"lw": 2})

# Correlation
r_val, p_corr = pearsonr(delta_nll, acc_all_ppts)

ax_corr.text(
    delta_nll.min(),
    ax_corr.get_ylim()[0] * 1.015,
    f"Pearson's r = {r_val:.2f}, p < .001" if p_corr < 0.001 else f"r = {r_val:.2f}, p = {p_corr:.3f}",
    fontsize=ANNOT_FONTSIZE,
    fontweight='bold')

# Labels & style
ax_corr.set_xlabel("Δ NLL", fontsize=LABEL_FONTSIZE)
ax_corr.set_ylabel("Overall accuracy [% correct]", fontsize=LABEL_FONTSIZE)
ax_corr.tick_params(labelsize=TICK_FONTSIZE)
ax_corr.spines["top"].set_visible(False)
ax_corr.spines["right"].set_visible(False)

# -------------------------------
# Custom legend for panel C (solid lines)
legend_handles = [Line2D([0], [0], color=model_colors[0], lw=2.5, label="Perceptual model"),
                  Line2D([0], [0], color=model_colors[1], lw=2.5, label="Compressed model")]
ax_mean.legend(handles=legend_handles, loc='upper right', frameon=False, fontsize=TICK_FONTSIZE)

# -------------------------------
# Save as PDF
fig.savefig(save_dir +"Figure_2.pdf", format='pdf', bbox_inches='tight', transparent=False)
plt.show()


        