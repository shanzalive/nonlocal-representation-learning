#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Figure 5 and Fig. S4: are co-updates between cross-contect states structured by value similarity and or visual similarity?
#
# Row 1: co-update as a function of the value difference between the two states.
# Row 2: co-update of states with the same vs a different house (across contexts).
# Input:  coupdate_LSS_zscored_miniblock_level_coupdate_values.csv (c_2nd_level_coupdate_analysis.py)
# Output: Figure5_cross_context_coupdates.pdf             (Fig. 5, OFC)
#         FigureS_cross_context_coupdates_other_ROIs.pdf  (Fig. S4: HPC, FFA, VTC)
#
# Author: Shany Grossman

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import sem, ttest_rel, ttest_1samp, gaussian_kde, pearsonr
from statsmodels.stats.multitest import multipletests
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import seaborn as sns
import json


#%%

# Which co-update table to read (output of c_coupdate_analysis.py / c_2nd_level_coupdate_analysis.py):
doProcrustes = False  # True: read the co-updates computed after aligning the pre to the post patterns of each
                      # mini-block (orthogonal Procrustes; analysis = 'coupdate_LSS_procrustes'). Results are unchanged (not reported)
doZscore = True       # True: read the co-updates computed from voxels z-scored across trials within each run
                      # (zscore_per_run = True in c_coupdate_analysis.py; the '_zscored' files). 
                      # Not to be confused with zscore_coupdate_vals below, which z-scores the co-update values.
sufstr = '_procrustes' if doProcrustes else ''
zscorestr = '_zscored' if doZscore else ''

coupdate_var = "corr"  # co-update = Pearson r between the two change vectors

# === Paths ===
files_dir = '/Users/grossman/Volumes/STATE_FORMATION/MRI_DATA/MRI_REAL/ROI_timeseries_LSS_5mm_smoothing_dropout_30_physio_reg_out_updated/coupdate_LSS' + sufstr + '/'
save_dir = '/Users/grossman/Projects/STATE_FORMATION/FIGURES/PAPER_FIGURES/'
coupdate_single_subject = pd.read_csv(os.path.join(files_dir, 'coupdate_LSS' + zscorestr + sufstr + '_miniblock_level_coupdate_values.csv'), engine='python')

# === ROIs ===
ROIs_ALL = ['OFC', 'HPC', 'FFA', 'VISUALFUNC_HIGHORDER']
MAIN_ROIS = ['OFC']                                    # Figure 5 (main text)
SUPP_ROIS = ['HPC', 'FFA', 'VISUALFUNC_HIGHORDER']      # Supplementary figure: remaining ROIs
ROIs_ttls = {"VISUALFUNC_HIGHORDER": "VTC", "OFC": "OFC", "HPC": "HPC", "FFA": "FFA"}
ROIs_time_series = ['HPC']

# === Colors ===
COLOR_DIFF_HOUSE = '#A94220'        # different house type, cross-context pairs
COLOR_SAME_HOUSE = '#E09478'        # same house type, cross-context pairs
COLOR_VD_MAX = '#5850C4'      # highest |Δval|, cross-context pairs
COLOR_VD_MIN = '#AAA6EE'      # lowest |Δval|, cross-context pairs
palette_house = {"diff_house": COLOR_DIFF_HOUSE, "same_house": COLOR_SAME_HOUSE}
palette_vd    = {"max_valdiff": COLOR_VD_MAX,    "min_valdiff": COLOR_VD_MIN}

ts_colors = {"OFC": "k", "PPA": "k"}

# === Participants ===
import os
def _find_summary_files_behaviour():
    d = os.path.dirname(os.path.abspath(__file__))
    while not os.path.isdir(os.path.join(d, 'summary_files_behaviour')):
        if os.path.dirname(d) == d:
            raise FileNotFoundError('summary_files_behaviour/ not found above ' + __file__)
        d = os.path.dirname(d)
    return os.path.join(d, 'summary_files_behaviour') + os.sep
SUMMARY_DIR = _find_summary_files_behaviour()

_ldict_path = SUMMARY_DIR + 'learners_dict.json'
with open(_ldict_path) as _f:
    _learners_dict = json.load(_f)
_in_csv = set(coupdate_single_subject["subject"].unique())
included_subjects = sorted([s for s in _learners_dict['full_and_partial']
                             if s in _in_csv])
print(f"Including {len(included_subjects)} subjects (full_and_partial ∩ CSV).")

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

#%% === Figure layouts ===
swarm = True
zscore_coupdate_vals = True  # demean+scale corr per subject × ROI across ALL pairs before cross-context filter


# Flag: add a co-update time series subplot next to each swarm/violin panel
# (value-dep panel -> min minus max |Δval| time series; house panel -> same minus diff house time series)
PLOT_COUPDATE_TIMESERIES = True
TS_SMOOTH_WIN = 4    # smoothing window for time series (stride=1); set to 1 to disable
TS_ALPHA_SIG = 0.05  # uncorrected one-sample t-test threshold for yellow significance shading
TS_WIDTH_SCALE = 0.85 * 0.90  # time series panel width, as a fraction of its allocated gridspec span
                               # (0.85 = initial 15% shrink; extra 0.90 = additional 10% requested afterward)
SWARM_WIDTH_SCALE = 1.10  # widen the swarm/violin panel by this factor when the time series is shown
TS_GAP_SCALE = 1.6        # multiply the swarm<->time-series gap by this factor when the time series is shown

# === Utility functions ===
def split_kde(ax, values, center_x, mirror="left", color="k", alpha=0.35, scale=0.5):
    vals = np.array(values)
    vals = vals[~np.isnan(vals)]
    if len(vals) < 3:
        return
    kde = gaussian_kde(vals)
    y_grid = np.linspace(vals.min(), vals.max(), 300)
    kde_vals = kde(y_grid)
    kde_vals = kde_vals / np.max(kde_vals) * scale
    if mirror == "left":
        x = center_x - kde_vals
    else:
        x = center_x + kde_vals
    ax.fill_betweenx(y_grid, center_x, x, color=color, alpha=alpha, linewidth=0, zorder=1)

def star_from_p(p):
    if np.isnan(p): return "n.s."
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."

def one_sample_sig_marker(ax, x, vals, y_top, label=None):
    vals = np.asarray(vals)
    vals = vals[~np.isnan(vals)]
    if vals.size < 2:
        return
    t, p = ttest_1samp(vals, 0, nan_policy='omit')
    if label is not None:
        d = np.mean(vals) / np.std(vals, ddof=1)
        print(f"[1-sample] {label}: t({vals.size - 1})={t:.3f}, p={p:.4g}, d={d:.3f}")
    if p < 0.05:
        ax.text(x, y_top, star_from_p(p), ha='center', va='bottom', fontsize=14, zorder=5, clip_on=False)

def smooth_running_nan(x, w):
    if w <= 1:
        return np.array(x)
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    half = w // 2
    n = len(x)
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        win = x[start:end]
        out[i] = np.nanmean(win) if np.any(~np.isnan(win)) else np.nan
    return out


# === Pre-compute data ===
_pair_df = coupdate_single_subject[coupdate_single_subject["subject"].isin(included_subjects)].copy()
_pair_df['val_diff'] = _pair_df['val_diff'].replace({13: 12, 38: 37, 63: 62})

# Z-score across ALL pairs (same + diff context) per subject × ROI before any filtering
if zscore_coupdate_vals:
    _stats = (_pair_df.groupby(["subject", "ROI"])[coupdate_var]
              .agg(["mean", "std"]).reset_index())
    _pair_df = _pair_df.merge(_stats, on=["subject", "ROI"])
    _pair_df[coupdate_var] = (_pair_df[coupdate_var] - _pair_df["mean"]) / _pair_df["std"]
    _pair_df = _pair_df.drop(columns=["mean", "std"])

# Figure 5 analyses cross-context pairs only
_pair_df = _pair_df[_pair_df['same_context'] == 0].copy()

def _house_contrast_per_subj(pair_df, coupdate_var):
    """Per subject × ROI: mean co-update for same vs diff house type, averaged over miniblocks."""
    mb_mean = (pair_df
               .groupby(["subject", "ROI", "runType", "imini_block", "same_house_type"])[[coupdate_var]]
               .mean().reset_index())
    subj_mean = (mb_mean
                 .groupby(["subject", "ROI", "same_house_type"])[[coupdate_var]]
                 .mean().reset_index())
    same = (subj_mean[subj_mean["same_house_type"] == 1]
            .rename(columns={coupdate_var: "corr_same_house"})
            .drop(columns="same_house_type"))
    diff = (subj_mean[subj_mean["same_house_type"] == 0]
            .rename(columns={coupdate_var: "corr_diff_house"})
            .drop(columns="same_house_type"))
    return same.merge(diff, on=["subject", "ROI"])

house_subj = _house_contrast_per_subj(_pair_df, coupdate_var)

def _vd_contrast_per_subj(pair_df, coupdate_var):
    """
    Per subject × ROI: mean co-update at the min and max |Δval| level present
    in each miniblock, averaged across miniblocks.
    Returns columns: subject, ROI, mean_min, mean_max.
    """
    rows = []
    for (sub, roi, run, mb), mb_df in pair_df.groupby(['subject', 'ROI', 'runType', 'imini_block']):
        valid = mb_df[['val_diff', coupdate_var]].dropna()
        level_means = valid.groupby('val_diff')[coupdate_var].mean()
        levels = sorted(level_means.index.tolist())
        if len(levels) < 2:
            continue
        rows.append({
            'subject': sub, 'ROI': roi,
            'mean_min': level_means[levels[0]],
            'mean_max': level_means[levels[-1]],
        })
    if not rows:
        return pd.DataFrame(columns=['subject', 'ROI', 'mean_min', 'mean_max'])
    df_mb = pd.DataFrame(rows)
    return df_mb.groupby(['subject', 'ROI'])[['mean_min', 'mean_max']].mean().reset_index()

# value-dependent co-update: all cross-context pairs
vd_subj = _vd_contrast_per_subj(_pair_df, coupdate_var)

# --- Chronological mini-block index (mirrors c_contextdep_coupdate_Figure4.py's mini_id) ---
_pair_df['runType_num'] = _pair_df['runType'].str.extract(r"(\d+)").astype(int)
_pair_df['mini_id'] = (_pair_df['imini_block'] + 1) + 2 * (_pair_df['runType_num'] - 1)

def _house_ts_per_subj_mb(pair_df, coupdate_var):
    """Per subject × ROI × mini_id: corr_same_house - corr_diff_house."""
    mb = (pair_df.groupby(['subject', 'ROI', 'mini_id', 'same_house_type'])[coupdate_var]
          .mean().reset_index())
    piv = mb.pivot_table(index=['subject', 'ROI', 'mini_id'], columns='same_house_type',
                          values=coupdate_var).reset_index()
    for col in [0, 1]:
        if col not in piv.columns:
            piv[col] = np.nan
    piv['ts_val'] = piv[1] - piv[0]   # same - diff
    return piv[['subject', 'ROI', 'mini_id', 'ts_val']]

house_ts = _house_ts_per_subj_mb(_pair_df, coupdate_var)

def _vd_ts_per_subj_mb(pair_df, coupdate_var):
    """Per subject × ROI × mini_id: mean co-update at the min minus max |Δval| level present in that mini-block."""
    rows = []
    for (sub, roi, mid), mb_df in pair_df.groupby(['subject', 'ROI', 'mini_id']):
        valid = mb_df[['val_diff', coupdate_var]].dropna()
        level_means = valid.groupby('val_diff')[coupdate_var].mean()
        levels = sorted(level_means.index.tolist())
        if len(levels) < 2:
            continue
        rows.append({'subject': sub, 'ROI': roi, 'mini_id': mid,
                     'ts_val': level_means[levels[0]] - level_means[levels[-1]]})   # min - max
    return pd.DataFrame(rows, columns=['subject', 'ROI', 'mini_id', 'ts_val'])

vd_ts = _vd_ts_per_subj_mb(_pair_df, coupdate_var)


#%% === Panel A: same vs different house type ===
max_ticks = 5

def _draw_house_panel(ax, roi):
    df_roi = house_subj[house_subj["ROI"] == roi]
    diff_vals = df_roi["corr_diff_house"].values
    same_vals = df_roi["corr_same_house"].values

    x_diff, x_same = 0, 1

    split_kde(ax, diff_vals, x_diff, mirror="left",  color=palette_house["diff_house"], alpha=0.6)
    split_kde(ax, same_vals, x_same, mirror="right", color=palette_house["same_house"], alpha=0.6)

    if swarm:
        df_pts = pd.DataFrame({
            "pos": np.concatenate([np.full(len(diff_vals), x_diff),
                                   np.full(len(same_vals), x_same)]),
            "y": np.concatenate([diff_vals, same_vals]),
            "cond": (["diff_house"] * len(diff_vals)) + (["same_house"] * len(same_vals))
        })
        sns.swarmplot(data=df_pts, x="pos", y="y", hue="cond",
                      palette=palette_house,
                      dodge=False, size=2.3, edgecolor="k", linewidth=0.4,
                      alpha=0.7, ax=ax, zorder=3)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        ax.set_ylabel("")
        ax.set_xlabel("")
    else:
        ax.scatter(np.random.normal(x_diff, 0.05, len(diff_vals)), diff_vals, s=16,
                   color=palette_house["diff_house"], edgecolor="k", linewidth=0.4, alpha=0.6, zorder=3)
        ax.scatter(np.random.normal(x_same, 0.05, len(same_vals)), same_vals, s=16,
                   color=palette_house["same_house"], edgecolor="k", linewidth=0.4, alpha=0.4, zorder=3)

    for dv, sv in zip(diff_vals, same_vals):
        ax.plot([x_diff + 0.25, x_same - 0.25], [dv, sv], color="gray", lw=0.6, alpha=0.6, zorder=1)

    mean_diff = np.nanmean(diff_vals)
    mean_same = np.nanmean(same_vals)
    sem_diff  = sem(diff_vals, nan_policy='omit')
    sem_same  = sem(same_vals, nan_policy='omit')
    ax.errorbar(x_diff, mean_diff, yerr=sem_diff, fmt='o', color='k', markersize=6, capsize=6, elinewidth=1.2, zorder=6)
    ax.errorbar(x_same, mean_same, yerr=sem_same, fmt='o', color='k', markersize=6, capsize=6, elinewidth=1.2, zorder=6)
    ax.plot([x_diff, x_same], [mean_diff, mean_same], color='k', lw=1, alpha=1, zorder=7)

    mask = ~np.isnan(diff_vals) & ~np.isnan(same_vals)
    if np.sum(mask) > 1:
        tval, p_pair = ttest_rel(np.array(diff_vals)[mask], np.array(same_vals)[mask], nan_policy='omit')
        diffs = np.array(same_vals)[mask] - np.array(diff_vals)[mask]
        d_pair = np.mean(diffs) / np.std(diffs, ddof=1)
        df_pair = int(np.sum(mask)) - 1
        print(f"[Panel A] {roi}: t({df_pair})={tval:.3f}, p={p_pair:.4g}, d={d_pair:.3f}")
        from scipy.stats import shapiro
        sw_stat, sw_p = shapiro(diffs)
        print(f"[Shapiro-Wilk] {roi}: W={sw_stat:.3f}, p={sw_p:.3f}"
              f"  {'-> NON-NORMAL' if sw_p < 0.05 else '-> normal OK'}")
    else:
        p_pair = np.nan
    sig_pair = star_from_p(p_pair)

    ymin = np.nanmin([np.nanmin(diff_vals), np.nanmin(same_vals)])
    ymax = np.nanmax([np.nanmax(diff_vals), np.nanmax(same_vals)])
    if np.isfinite(ymin) and np.isfinite(ymax):
        yrange = ymax - ymin if (ymax - ymin) != 0 else 1.0
    else:
        ymin, ymax, yrange = -0.01, 0.01, 0.02

    pad = 0.05 * yrange
    ax.set_ylim(ymin - pad, ymax + pad)

    _xl, _xr = x_diff, x_same
    _bh = 0.03 * yrange
    line_y = ymax + pad + 0.04 * yrange
    ax.plot([_xl, _xr], [line_y, line_y], color='k', lw=1, zorder=8, clip_on=False)
    ax.plot([_xl, _xl], [line_y - _bh, line_y], color='k', lw=1, zorder=8, clip_on=False)
    ax.plot([_xr, _xr], [line_y - _bh, line_y], color='k', lw=1, zorder=8, clip_on=False)
    ax.text(0.5, line_y - 0.08 * yrange  if sig_pair != "n.s." else line_y - 0.002 * yrange, sig_pair, ha='center', va='bottom',
            fontsize=(16 if sig_pair != "n.s." else 8), zorder=9, clip_on=False)


    if not np.isnan(p_pair):
        _ps = f"{p_pair:.1g}"; _ps = _ps[1:] if _ps.startswith("0.") else _ps
        ax.text(0.5, line_y - 0.01 * yrange, f"p={_ps}",
                ha='center', va='top', fontsize=6, zorder=9, clip_on=False)

    one_y = ymax - 0.05 * yrange
    one_sample_sig_marker(ax, x_diff, diff_vals, one_y, label=f"{roi} diff_house")
    one_sample_sig_marker(ax, x_same, same_vals, one_y, label=f"{roi} same_house")

    ax.set_xlim(-0.6, 1.6)
    ax.set_xticks([-0.1, 1.1])
    ax.set_xticklabels(["different", "same"], fontsize=10)
    ax.set_xlabel("House type", fontsize=10)
    ax.set_title(ROIs_ttls[roi], fontweight='bold', fontsize=11, pad=14)
    ax.axhline(0, color='k', linestyle='--', lw=0.8, zorder=0)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=max_ticks, prune=None))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.tick_params(axis='y', which='major', pad=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


#%% === Panel VD: max |Δval| (left) vs min |Δval| (right) ===

def _draw_vd_panel(ax, roi):
    roi_vd   = vd_subj[vd_subj['ROI'] == roi]
    min_vals = roi_vd['mean_min'].values
    max_vals = roi_vd['mean_max'].values

    # Max on left (x=0), Min on right (x=1)
    x_max, x_min = 0, 1

    split_kde(ax, max_vals, x_max, mirror="left",  color=palette_vd["max_valdiff"], alpha=0.6)
    split_kde(ax, min_vals, x_min, mirror="right", color=palette_vd["min_valdiff"], alpha=0.4)

    if swarm:
        df_pts = pd.DataFrame({
            "pos": np.concatenate([np.full(len(max_vals), x_max),
                                   np.full(len(min_vals), x_min)]),
            "y": np.concatenate([max_vals, min_vals]),
            "cond": (["max_valdiff"] * len(max_vals)) + (["min_valdiff"] * len(min_vals))
        })
        sns.swarmplot(data=df_pts, x="pos", y="y", hue="cond",
                      palette=palette_vd,
                      dodge=False, size=2.3, edgecolor="k", linewidth=0.4,
                      alpha=0.7, ax=ax, zorder=3)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        ax.set_ylabel("")
        ax.set_xlabel("")

    # connecting lines (max left → min right)
    for xv, mv in zip(max_vals, min_vals):
        if np.isfinite(xv) and np.isfinite(mv):
            ax.plot([x_max + 0.25, x_min - 0.25], [xv, mv],
                    color="gray", lw=0.6, alpha=0.6, zorder=1)

    mean_max = np.nanmean(max_vals)
    mean_min = np.nanmean(min_vals)
    sem_max  = sem(max_vals, nan_policy='omit')
    sem_min  = sem(min_vals, nan_policy='omit')
    ax.errorbar(x_max, mean_max, yerr=sem_max, fmt='o', color='k', markersize=6, capsize=6, elinewidth=1.2, zorder=6)
    ax.errorbar(x_min, mean_min, yerr=sem_min, fmt='o', color='k', markersize=6, capsize=6, elinewidth=1.2, zorder=6)
    ax.plot([x_max, x_min], [mean_max, mean_min], color='k', lw=1, zorder=7)

    mask = ~np.isnan(min_vals) & ~np.isnan(max_vals)
    if np.sum(mask) > 1:
        tval, p_pair = ttest_rel(max_vals[mask], min_vals[mask])
        diffs = max_vals[mask] - min_vals[mask]
        d_pair = np.mean(diffs) / np.std(diffs, ddof=1)
        df_pair = int(np.sum(mask)) - 1
        print(f"[Panel VD] {roi}: t({df_pair})={tval:.3f}, p={p_pair:.4g}, d={d_pair:.3f}")
    else:
        p_pair = np.nan
    sig_pair = star_from_p(p_pair)

    all_vals = np.concatenate([min_vals, max_vals])
    all_vals = all_vals[np.isfinite(all_vals)]
    if len(all_vals):
        ymin_vd = np.min(all_vals); ymax_vd = np.max(all_vals)
        yrange_vd = (ymax_vd - ymin_vd) if (ymax_vd - ymin_vd) != 0 else 1.0
    else:
        ymin_vd, ymax_vd, yrange_vd = -0.01, 0.01, 0.02
    pad = 0.05 * yrange_vd
    ax.set_ylim(ymin_vd - pad, ymax_vd + pad)

    _xl, _xr = x_max, x_min
    _bh = 0.03 * yrange_vd
    line_y = ymax_vd + pad + 0.04 * yrange_vd
    ax.plot([_xl, _xr], [line_y, line_y], color='k', lw=1, zorder=8, clip_on=False)
    ax.plot([_xl, _xl], [line_y - _bh, line_y], color='k', lw=1, zorder=8, clip_on=False)
    ax.plot([_xr, _xr], [line_y - _bh, line_y], color='k', lw=1, zorder=8, clip_on=False)
    ax.text(0.5,line_y - 0.08 * yrange_vd  if sig_pair != "n.s." else line_y - 0.002 * yrange_vd , sig_pair, ha='center', va='bottom',
            fontsize=(16 if sig_pair != "n.s." else 8), zorder=9, clip_on=False)
    if not np.isnan(p_pair):
        _ps = f"{p_pair:.1g}"; _ps = _ps[1:] if _ps.startswith("0.") else _ps
        ax.text(0.5, line_y - 0.01 * yrange_vd, f"p={_ps}",
                ha='center', va='top', fontsize=6, zorder=9, clip_on=False)

    one_y_vd = ymax_vd - 0.05 * yrange_vd
    one_sample_sig_marker(ax, x_max, max_vals, one_y_vd, label=f"{roi} max_valdiff")
    one_sample_sig_marker(ax, x_min, min_vals, one_y_vd, label=f"{roi} min_valdiff")

    ax.axhline(0, color='k', linestyle='--', lw=0.8, zorder=0)
    ax.set_xlim(-0.6, 1.6)
    ax.set_xticks([-0.1, 1.1])
    ax.set_xticklabels(["max.", "min."], fontsize=10)
    ax.set_xlabel("|Δ value|", fontsize=10)
    ax.set_title(ROIs_ttls[roi], fontweight='bold', fontsize=11, pad=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=max_ticks, prune=None))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.tick_params(axis='y', which='major', pad=2)


#%% === Panel TS: co-update time series (as in c_contextdep_coupdate_Figure4.py Panel B) ===

def _draw_ts_panel(ax, roi, ts_df, ylabel_text):
    roi_df = ts_df[ts_df['ROI'] == roi]
    all_mini = np.arange(1, int(ts_df['mini_id'].max()) + 1)
    pv = roi_df.pivot_table(index='subject', columns='mini_id', values='ts_val').reindex(columns=all_mini)

    subjects = pv.index.tolist()
    subj_smoothed = np.vstack([
        smooth_running_nan(pv.loc[s].values.astype(float), TS_SMOOTH_WIN) for s in subjects
    ]) if subjects else np.zeros((0, len(all_mini)))

    mean_smooth = np.nanmean(subj_smoothed, axis=0)
    sem_smooth = np.nanstd(subj_smoothed, axis=0) / np.sqrt(np.sum(~np.isnan(subj_smoothed), axis=0))
    color = ts_colors.get(roi, "k")

    ax.plot(all_mini, mean_smooth, lw=1.7, color=color)
    ax.fill_between(all_mini, mean_smooth - sem_smooth, mean_smooth + sem_smooth, alpha=0.25, color=color)

    for t_idx, mini in enumerate(all_mini):
        col_vals = subj_smoothed[:, t_idx]
        col_vals = col_vals[~np.isnan(col_vals)]
        if col_vals.size > 1:
            _, pval = ttest_1samp(col_vals, 0, nan_policy='omit')
            if pval <= TS_ALPHA_SIG:
                ax.axvspan(mini - 0.5, mini + 0.5, color="yellow", alpha=0.3, zorder=0)

    ax.axhline(0, color='k', linestyle='--', lw=0.8, zorder=0)
    ax.set_title(ROIs_ttls[roi], fontweight='bold', fontsize=11)
    ax.set_ylabel(ylabel_text, fontsize=10)
    ax.set_xlabel("Learning segment #", fontsize=10)
    ax.set_xticks(np.arange(2, 23, 2))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune=None))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    valid_cols = np.any(~np.isnan(subj_smoothed), axis=0)
    if valid_cols.any():
        ax.set_xlim(all_mini[valid_cols][0], all_mini[valid_cols][-1])


#%% === Figure 5 (main text): OFC only. Row 1: value-dep, Row 2: house-type ===
# The canvas is always the fixed 2x4, 7.2x6in grid. Column 0 = OFC swarm. When
# PLOT_COUPDATE_TIMESERIES is True, the time series is drawn into the remaining columns (1-3).
_ofc_col = ROIs_ALL.index('OFC')

fig1 = plt.figure(figsize=(7.2, 6))
gs1 = fig1.add_gridspec(2, len(ROIs_ALL), height_ratios=[1.0, 1.0], hspace=0.75, wspace=0.4)
ax_vd_main    = fig1.add_subplot(gs1[0, _ofc_col])   # Row 1 (top): value-dependent co-update
ax_house_main = fig1.add_subplot(gs1[1, _ofc_col])   # Row 2 (bottom): house-type co-update

_draw_vd_panel(ax_vd_main, 'OFC')
_draw_house_panel(ax_house_main, 'OFC')
ax_vd_main.set_ylabel("Mean co-update [z(r)]\ncross-context pairs", fontsize=11)
ax_house_main.set_ylabel("Mean co-update [z(r)]\ncross-context pairs", fontsize=11)

if PLOT_COUPDATE_TIMESERIES:
    ax_vd_ts_main    = fig1.add_subplot(gs1[0, _ofc_col + 1:len(ROIs_ALL)])
    ax_house_ts_main = fig1.add_subplot(gs1[1, _ofc_col + 1:len(ROIs_ALL)])
    _draw_ts_panel(ax_vd_ts_main, 'OFC', vd_ts, "Value-dep. co-update [z(r)]\n[min − max |Δval|]")
    _draw_ts_panel(ax_house_ts_main, 'OFC', house_ts, "House-dep. co-update [z(r)]\n[same − diff house]")

fig1.tight_layout()
if PLOT_COUPDATE_TIMESERIES:
    # Widen the swarm panel, shrink the time series panel, and enlarge the gap between
    # them, computed from the raw tight_layout positions.
    for _swarm_ax, _ts_ax in ((ax_vd_main, ax_vd_ts_main), (ax_house_main, ax_house_ts_main)):
        _s_pos = _swarm_ax.get_position()
        _t_pos = _ts_ax.get_position()
        _orig_gap = _t_pos.x0 - (_s_pos.x0 + _s_pos.width)

        _new_s_w = _s_pos.width * SWARM_WIDTH_SCALE
        _new_t_w = _t_pos.width * TS_WIDTH_SCALE
        _new_gap = _orig_gap * TS_GAP_SCALE

        _swarm_ax.set_position([_s_pos.x0, _s_pos.y0, _new_s_w, _s_pos.height])
        _ts_ax.set_position([_s_pos.x0 + _new_s_w + _new_gap, _t_pos.y0, _new_t_w, _t_pos.height])
fig1.savefig(os.path.join(save_dir, "Figure5_cross_context_coupdates.pdf"), dpi=600, transparent=True)
plt.show()


#%% === Supplementary Figure: remaining ROIs (HPC, FFA, VTC). Row 1: value-dep, Row 2: house-type ===
# Canvas width fixed to match Figure 5's 7.2in (not scaled down by ROI count).
fig1_supp = plt.figure(figsize=(7.2, 6))
gs1_supp = fig1_supp.add_gridspec(2, len(SUPP_ROIS), height_ratios=[1.0, 1.0], hspace=0.75, wspace=0.4)
axes_vd_supp = [fig1_supp.add_subplot(gs1_supp[0, i]) for i in range(len(SUPP_ROIS))]   # Row 1: value-dep
axes_a_supp  = [fig1_supp.add_subplot(gs1_supp[1, i]) for i in range(len(SUPP_ROIS))]   # Row 2: house type

for i, roi in enumerate(SUPP_ROIS):
    _draw_vd_panel(axes_vd_supp[i], roi)
for i, roi in enumerate(SUPP_ROIS):
    _draw_house_panel(axes_a_supp[i], roi)

axes_vd_supp[0].set_ylabel("Mean co-update [z(r)]\ncross-context pairs", fontsize=11)
axes_a_supp[0].set_ylabel("Mean co-update [z(r)]\ncross-context pairs", fontsize=11)

fig1_supp.tight_layout()
fig1_supp.savefig(os.path.join(save_dir, "FigureS_cross_context_coupdates_other_ROIs.pdf"), dpi=600, transparent=True)
plt.show()
