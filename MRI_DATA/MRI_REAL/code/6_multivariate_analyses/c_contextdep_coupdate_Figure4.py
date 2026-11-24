#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Figure 4 and Fig. S2a: context-dependent co-updates in the brain.
#
# Panel a: co-update of same-context vs different-context state pairs in each ROI.
# Co-update values are z-scored per participant and ROI (doZscore = True), averaged per
# mini-block and then across mini-blocks, and compared with paired t-tests.
# Also: whole-brain ROIs (Fig. 4c) and the time course over learning (IPL: Fig. S2a).
# Input:  coupdate_LSS_zscored_miniblock_level_coupdate_values.csv (c_2nd_level_coupdate_analysis.py),
#         summary_files_behaviour/learners_dict.json
# Output: Figure4.pdf
#
# Author: Shany Grossman

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import sem, ttest_rel, ttest_1samp, gaussian_kde
from statsmodels.stats.multitest import multipletests
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import seaborn as sns
import json

#%%

# Which co-update table to read (output of c_coupdate_analysis.py / c_2nd_level_coupdate_analysis.py):
doProcrustes = False  # True: read the co-updates computed after aligning the pre to the post patterns of each
                      # mini-block (orthogonal Procrustes; analysis = 'coupdate_LSS_procrustes'). Not used in the paper.
doZscore = True       # True: read the co-updates computed from voxels z-scored across trials within each run
                      # (zscore_per_run = True in c_coupdate_analysis.py; the '_zscored' files). Used in the paper.
                      # Not to be confused with zscore_coupdate_vals below, which z-scores the co-update values.
sufstr = '_procrustes' if doProcrustes  else ''
zscorestr = '_zscored' if doZscore  else ''

# === Paths ===
files_dir = '/Users/grossman/Volumes/STATE_FORMATION/MRI_DATA/MRI_REAL/ROI_timeseries_LSS_5mm_smoothing_dropout_30_physio_reg_out_updated/coupdate_LSS' + sufstr + '/' #_compcorr
save_dir = '/Users/grossman/Projects/STATE_FORMATION/FIGURES/PAPER_FIGURES/'
import os
def _find_summary_files_behaviour():
    d = os.path.dirname(os.path.abspath(__file__))
    while not os.path.isdir(os.path.join(d, 'summary_files_behaviour')):
        if os.path.dirname(d) == d:
            raise FileNotFoundError('summary_files_behaviour/ not found above ' + __file__)
        d = os.path.dirname(d)
    return os.path.join(d, 'summary_files_behaviour') + os.sep
SUMMARY_DIR = _find_summary_files_behaviour()

grand_log_dir = SUMMARY_DIR

coupdate_group_summary = pd.read_csv(os.path.join(files_dir, 'coupdate_LSS' + zscorestr + sufstr  + '.csv'))
coupdate_single_subject  = pd.read_csv(os.path.join(files_dir, 'coupdate_LSS'  + zscorestr + sufstr + '_miniblock_level_coupdate_values.csv'), engine='python')

coupdate_var = "corr"  # co-update = Pearson r between the two change vectors
grand_log =  pd.read_csv(grand_log_dir + 'State_Formation_behaviour.csv')

# === ROIs  ===
ROIs = ['OFC', 'HPC', 'FFA', 'VISUALFUNC_HIGHORDER']
ROIs_time_series = ['inferiorparietal'] #OFC 
ROIs_ttls = {"OFC": "OFC", "HPC": "HPC", "FFA": "FFA", "VISUALFUNC_HIGHORDER": "VTC",
             "inferiorparietal": "Inferior Parietal"}

#%%
_ldict_path = SUMMARY_DIR + 'learners_dict.json'
with open(_ldict_path) as _f:
    _learners_dict = json.load(_f)
_in_csv = set(coupdate_group_summary["subject"].unique())
included_subjects = sorted([s for s in _learners_dict['full_and_partial']
                             if s in _in_csv])
print(f"Including {len(included_subjects)} subjects (full_and_partial ∩ CSV).")

# === Colors ===
palette = {"diff": "darkblue", "same": "#5BC8C0"}  # Panel A stays the same
# Option B color scheme for Panel B (change to preference)
ts_colors = {"OFC": "k", "FFA": "k"}  # Option B

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
    'ytick.major.size': 3,})

#%% === Figure layout ===

plot_learning_aligned_coupdate = False
swarm = True   # True = seaborn swarmplot, False = jittered scatter
balance_pair_counts = False  # subsample diff-context pairs per miniblock to match same-context count
N_BALANCE_BOOT = 100         # bootstrap iterations for balancing (used when balance_pair_counts=True)
zscore_coupdate_vals = True  # demean corr per subject × ROI across all pairs before analysis


fig = plt.figure(figsize=(7.2, 6.2))
gs = fig.add_gridspec(2, len(ROIs),
                      height_ratios=[0.85, 0.6],
                      hspace=0.55, wspace=0.4)

# --- Row 1: 4 single-ROI subplots ---
axes_a = [fig.add_subplot(gs[0, i]) for i in range(len(ROIs))]

# --- Row 2: OFC time series (left half) + Cohen's d bar chart (right half) ---
# Bar chart used to span its own row 3; it now shares row 2 in place of the
# FFA time series, so it renders at half the width (and needs smaller type).
axes_b = [
    fig.add_subplot(gs[1, 0:len(ROIs)//2]),
    fig.add_subplot(gs[1, len(ROIs)//2:len(ROIs)])]
axes_c = [axes_b[1]]


# === utility functions ===
def split_kde(ax, values, center_x, mirror="left", color="k", alpha=0.35, scale=0.5):
    """Half-KDE outward from center_x."""
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
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return "n.s."

def one_sample_sig_marker(ax, x, vals, y_top, label=None):
    """ one-sample t-test against 0 (nan-safe). place star at y_top if sig."""
    vals = np.asarray(vals)
    vals = vals[~np.isnan(vals)]
    if vals.size < 2:
        return
    t, p = ttest_1samp(vals, 0, nan_policy='omit')
    if label is not None:
        d = np.mean(vals) / np.std(vals, ddof=1)
        print(f"  one-sample t-test [{label}]: t({len(vals)-1})={t:.3f}, p={p:.3f}, d={d:.3f}")
    if p < 0.05:
        s = star_from_p(p)
        ax.text(x, y_top, s, ha='center', va='bottom', fontsize=12, zorder=5, clip_on=False)

def smooth_running_nan(x, w):
    """Simple centered moving average (nan-safe). w must be >=1. stride=1."""
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

def compute_balanced_miniblock_means(pair_df, coupdate_var, n_boot=200, seed=42):
    """
    Returns miniblock-level means after subsampling diff-context pairs per miniblock
    to match the same-context pair count. Averages over n_boot iterations.
    Miniblocks with no same-context pairs are dropped from both conditions.
    """
    rng = np.random.default_rng(seed)
    keys = ["subject", "ROI", "runType", "imini_block"]
    n_same_per_mb = (
        pair_df[pair_df["same_context"] == 1]
        .groupby(keys).size().rename("n_same").reset_index()
    )
    df = pair_df.merge(n_same_per_mb, on=keys, how="left")
    df["n_same"] = df["n_same"].fillna(0).astype(int)
    same_df = df[df["same_context"] == 1].drop(columns="n_same")
    diff_df_base = df[df["same_context"] == 0].copy()

    boot_results = []
    for _ in range(n_boot):
        diff_work = diff_df_base.copy()
        diff_work["_u"] = rng.random(len(diff_work))
        diff_work["_rank"] = diff_work.groupby(keys)["_u"].rank(method="first") - 1
        diff_sampled = diff_work[diff_work["_rank"] < diff_work["n_same"]].drop(
            columns=["_u", "_rank", "n_same"])
        combined = pd.concat([same_df, diff_sampled], ignore_index=True)
        mb_mean = (combined
                   .groupby(keys + ["same_context"])[[coupdate_var]]
                   .mean().reset_index())
        boot_results.append(mb_mean)

    all_boots = pd.concat(boot_results, ignore_index=True)
    return all_boots.groupby(keys + ["same_context"])[[coupdate_var]].mean().reset_index()


# === Pre-compute data for Panels A, B, C ===
_pair_df = coupdate_single_subject[coupdate_single_subject["subject"].isin(included_subjects)].copy()

if zscore_coupdate_vals:
    _stats = (_pair_df.groupby(["subject", "ROI"])[coupdate_var]
              .agg(["mean", "std"]).reset_index())
    _pair_df = _pair_df.merge(_stats, on=["subject", "ROI"])
    _pair_df[coupdate_var] = (_pair_df[coupdate_var] - _pair_df["mean"])/ _pair_df["std"]
    _pair_df = _pair_df.drop(columns=["mean", "std"])

if balance_pair_counts:
    print(f"Computing balanced pair counts ({N_BALANCE_BOOT} bootstrap iterations)...")
    balanced_mb = compute_balanced_miniblock_means(_pair_df, coupdate_var, n_boot=N_BALANCE_BOOT)
else:
    balanced_mb = (_pair_df
                   .groupby(["subject", "ROI", "runType", "imini_block", "same_context"])[[coupdate_var]]
                   .mean().reset_index())
_subj = (balanced_mb
         .groupby(["subject", "ROI", "same_context"])[[coupdate_var]]
         .mean().reset_index())
_same = (_subj[_subj["same_context"] == 1]
         .rename(columns={coupdate_var: "corr_same_context"})
         .drop(columns="same_context"))
_diff = (_subj[_subj["same_context"] == 0]
         .rename(columns={coupdate_var: "corr_diff_context"})
         .drop(columns="same_context"))
active_group_summary = _same.merge(_diff, on=["subject", "ROI"])
active_group_summary["corr_var_name"] = coupdate_var

# === ROI vs ventricles (noise-control) paired contrast; vent_effect reused below ===
VENTRICLE_ROI = '3&4VENTRICLES'
vent_effect = (active_group_summary[active_group_summary.ROI == VENTRICLE_ROI]
               .set_index("subject")
               .eval("corr_same_context - corr_diff_context"))


# === Panel A: same vs different context (top row) ===
max_ticks=5
for i, roi in enumerate(ROIs):
    ax = axes_a[i]
    
    # Filter for the current ROI, included subjects, and the specific correlation variable
    df_roi = active_group_summary[active_group_summary.ROI == roi]
    df_roi = df_roi[df_roi["subject"].isin(included_subjects)].copy()
    df_roi = df_roi[df_roi['corr_var_name'] == coupdate_var].copy()
    diff_vals = df_roi["corr_diff_context"].values
    same_vals = df_roi["corr_same_context"].values
    print(f"{roi}: N={len(df_roi)} subjects")

    x_diff, x_same = 0, 1

    # KDEs (outward)
    split_kde(ax, diff_vals, x_diff - 0.0, mirror="left", color=palette["diff"], alpha=0.4)
    split_kde(ax, same_vals, x_same + 0.0, mirror="right", color=palette["same"], alpha=0.4)
    
    # individual points: swarm OR jittered scatter
    if swarm:
        import pandas as pd
    
        df_pts = pd.DataFrame({
            "x": np.concatenate([np.full(len(diff_vals), x_diff),
                                  np.full(len(same_vals), x_same)]),
            "y": np.concatenate([diff_vals, same_vals]),
            "cond": (["diff"] * len(diff_vals)) + (["same"] * len(same_vals))
        })
    
        sns.swarmplot(
            data=df_pts,
            x="x",
            y="y",
            hue="cond",
            palette={"diff": palette["diff"], "same": palette["same"]},
            dodge=False,
            size=2.3,
            edgecolor="k",
            linewidth=0.4,
            alpha=0.7,
            ax=ax,
            zorder=3
        )
    
        # remove automatic legend from swarmplot
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        ax.set_ylabel("")
        ax.set_xlabel("")
    
    else:
        ax.scatter(np.random.normal(x_diff, 0.05, len(diff_vals)), diff_vals, s=16,
                   color=palette["diff"], edgecolor="k", linewidth=0.4,
                   alpha=0.6, zorder=3)
    
        ax.scatter(np.random.normal(x_same, 0.05, len(same_vals)), same_vals, s=16,
                   color=palette["same"], edgecolor="k", linewidth=0.4,
                   alpha=0.6, zorder=3)


    # connecting subject lines (thin)
    for dv, sv in zip(diff_vals, same_vals):
        ax.plot([x_diff + 0.25, x_same - 0.25], [dv, sv], color="gray", lw=0.6, alpha=0.6, zorder=1)

    # mean ± sem
    mean_diff = np.nanmean(diff_vals)
    mean_same = np.nanmean(same_vals)
    sem_diff = sem(diff_vals, nan_policy='omit')
    sem_same = sem(same_vals, nan_policy='omit')
    ax.errorbar(x_diff, mean_diff, yerr=sem_diff, fmt='o', color='k', markersize=6, capsize=6, elinewidth=1.2, zorder=6)
    ax.errorbar(x_same, mean_same, yerr=sem_same, fmt='o', color='k', markersize=6, capsize=6, elinewidth=1.2, zorder=6)

    # heavy black line connecting means (no alpha)
    ax.plot([x_diff, x_same], [mean_diff, mean_same], color='k', lw=1, alpha=1, zorder=7)

    # paired t-test
    mask = ~np.isnan(diff_vals) & ~np.isnan(same_vals)
    if np.sum(mask) > 1:
        tval, p_pair = ttest_rel(np.array(diff_vals)[mask], np.array(same_vals)[mask], nan_policy='omit')
        df_pair = int(np.sum(mask)) - 1
        diffs = np.array(same_vals)[mask] - np.array(diff_vals)[mask]
        d_pair = np.mean(diffs) / np.std(diffs, ddof=1)
        print(f"  paired t-test [{roi}] (same vs diff context): t({df_pair})={tval:.3f}, p={p_pair:.3f}, d={d_pair:.3f}")
        from scipy.stats import shapiro
        sw_stat, sw_p = shapiro(diffs)
        print(f"  [Shapiro-Wilk] {roi}: W={sw_stat:.3f}, p={sw_p:.3f}"
              f"  {'-> NON-NORMAL, consider Wilcoxon' if sw_p < 0.05 else '-> normal OK'}")
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
    ax.text(0.5,  line_y - 0.08 * yrange  if sig_pair != "n.s." else line_y - 0.002 * yrange, sig_pair, ha='center', va='bottom',
            fontsize=(16 if sig_pair != "n.s." else 8), zorder=9, clip_on=False)
    if not np.isnan(p_pair):
        _ps = f"{p_pair:.1g}"; _ps = _ps[1:] if _ps.startswith("0.") else _ps
        ax.text(0.5, line_y + 0.001 * yrange, f"p={_ps}", ha='center', va='top', fontsize=6, zorder=9, clip_on=False)

    # single-sample tests vs 0 (place stars above swarm)
    one_y =  ymax - 0.05 * yrange
    one_sample_sig_marker(ax, x_diff, diff_vals, one_y, label=f"{roi} diff-context vs 0")
    one_sample_sig_marker(ax, x_same, same_vals, one_y, label=f"{roi} same-context vs 0")

    # axes formatting
    ax.set_xlim(-0.6, 1.6)
    ax.set_xticks([0, 1])
    ax.set_xticks([-0.1, 1.1])  
    ax.set_xticklabels(["different", "same"], fontsize=10)
    ax.set_xlabel("Context", fontsize=10)

    ax.set_title(ROIs_ttls[roi], fontweight='bold', fontsize=11, pad=14)
    ax.axhline(0, color='k', linestyle='--', lw=0.8, zorder=0)

    # y tick formatting & consistent number of ticks
    ax.yaxis.set_major_locator(MaxNLocator(nbins=max_ticks, prune=None))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # ensure y tick labels are closer to axis to avoid overlap
    ax.tick_params(axis='y', which='major', pad=2)

    # hide top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # left spine remains


axes_a[0].set_ylabel("Mean co-update [z(r)]", fontsize=11)

# === Panel B === Coupdate timeseries for signifcant ROIs
# Parameters (user can change)
SMOOTH_WIN = 4#window size for smoothing (stride is 1). set to 1 to disable smoothing
alpha_sig = 0.05
max_ticks=3

# prepare df and mini id
if balanced_mb is not None:
    df = balanced_mb.copy()
else:
    df = (_pair_df
          .groupby(["subject", "ROI", "runType", "imini_block", "same_context"])
          .agg({coupdate_var: "mean"})
          .reset_index())
 
df["imini_block"] = df["imini_block"].astype(int)
df["runType_num"] = df["runType"].str.extract(r"(\d+)").astype(int)
df["mini_id"] = (df["imini_block"] + 1) + 2 * (df["runType_num"] - 1)
df = df.sort_values(["subject", "ROI", "mini_id"])

# compute per-subject coupdate_diff = same_context(1) - same_context(0) per mini_id
agg = (
    df.groupby(["subject", "ROI", "mini_id", "same_context"])
    .agg(val=(coupdate_var, "mean"))
    .reset_index())

pivot_long = (
    agg.pivot_table(index=["subject", "ROI", "mini_id"], columns="same_context", values="val")
    .reset_index())

for col in [0, 1]:
    if col not in pivot_long.columns:
        pivot_long[col] = np.nan
pivot_long["coupdate_diff"] = pivot_long[1] - pivot_long[0]

all_mini = np.arange(1, int(df["mini_id"].max()) + 1)

# Build subject × mini matrix per ROI
roi_pivots = {}
for roi in ROIs_time_series:
    roi_df = pivot_long[pivot_long["ROI"] == roi]
    pv = roi_df.pivot_table(index="subject", columns="mini_id", values="coupdate_diff")
    pv = pv.reindex(columns=all_mini)
    roi_pivots[roi] = pv

# === Panel B: experiment-aligned time series ===
if not plot_learning_aligned_coupdate:
    for idx, (ax, roi) in enumerate(zip(axes_b[:1], ROIs_time_series)):
        pv = roi_pivots[roi]

        # Smooth each subject first
        subj_smoothed = []
        subjects = pv.index.tolist()
        for subj in subjects:
            subj_vals = pv.loc[subj].values.astype(float)
            subj_sm = smooth_running_nan(subj_vals, SMOOTH_WIN)
            subj_smoothed.append(subj_sm)
        subj_smoothed = np.vstack(subj_smoothed) if len(subj_smoothed) else np.zeros((0, len(all_mini)))

        # Compute mean and SEM on smoothed subject curves
        mean_smooth = np.nanmean(subj_smoothed, axis=0)
        sem_smooth = np.nanstd(subj_smoothed, axis=0) / np.sqrt(np.sum(~np.isnan(subj_smoothed), axis=0))

        # Plot mean ± SEM
        ax.plot(all_mini, mean_smooth, lw=1.7, color=ts_colors.get(roi, "black"))
        ax.fill_between(all_mini, mean_smooth - sem_smooth, mean_smooth + sem_smooth,
                        alpha=0.25, color=ts_colors.get(roi, "black"))

        # 1-sample t-test on smoothed data per time point (uncorrected, p<0.05)
        for t_idx, mini in enumerate(all_mini):
            col_vals = subj_smoothed[:, t_idx]
            col_vals = col_vals[~np.isnan(col_vals)]
            if col_vals.size > 1:
                _, pval = ttest_1samp(col_vals, 0, nan_policy='omit')
                if pval <= alpha_sig:
                    ax.axvspan(mini - 0.5, mini + 0.5, color="yellow", alpha=0.3, zorder=0)

        # horizontal zero line
        ax.axhline(0, color="k", linestyle="--", lw=0.8, zorder=0)
        ax.set_title(ROIs_ttls[roi], fontweight="bold", fontsize=11)
        if idx == 0:
            ax.set_ylabel("Context-dependent co-update\n[same context - different contexts]", fontsize=10)
        else:
            ax.set_ylabel("")

        # y ticks
        ax.yaxis.set_major_locator(MaxNLocator(nbins=max_ticks, prune=None))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        # aesthetics
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.tick_params(axis='x', which='major', length=0)

        # set xlim to first plotted data point
        first_non_nan = np.nanmin(all_mini[np.any(~np.isnan(subj_smoothed), axis=0)])
        last_non_nan = np.nanmax(all_mini[np.any(~np.isnan(subj_smoothed), axis=0)])
        ax.set_xlim(first_non_nan, last_non_nan)

    axes_b[0].set_xlabel("Learning segment #", fontsize=11)

# === Panel C ===  plot Cohen's d for all anatomical ROIs (always shown)
df = active_group_summary[active_group_summary["subject"].isin(included_subjects)].copy()
df["diff"] = df["corr_same_context"] - df["corr_diff_context"]

# Filter out contituents of OFC and other functionally dervied ROIs (no effect there any way)
df = df[df["ROI"] != "lateralorbitofrontal"]
df = df[df["ROI"] != "medialorbitofrontal"]
df = df[df["ROI"] != "parsorbitalis"]
df = df[df["ROI"] != "VISUALFUNC"]
df = df[df["ROI"] != "LO"]
df = df[df["ROI"] != "PPA"]

def cohens_d_paired(diffs):
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) < 2:
        return np.nan
    return np.mean(diffs) / np.std(diffs, ddof=1)

def bootstrap_ci_d(diffs, n_boot=1000, ci=95):
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(42)
    boot_d = [cohens_d_paired(rng.choice(diffs, size=len(diffs), replace=True))
              for _ in range(n_boot)]
    lo = np.percentile(boot_d, (100 - ci) / 2)
    hi = np.percentile(boot_d, 100 - (100 - ci) / 2)
    return lo, hi

# Compute Cohen's d + 95% bootstrap CI per ROI
roi_records = []
for roi, grp in df.groupby("ROI"):
    diffs = grp["diff"].values
    d = cohens_d_paired(diffs)
    lo, hi = bootstrap_ci_d(diffs)
    roi_records.append({"ROI": roi, "cohens_d": d, "ci_lo": lo, "ci_hi": hi})

roi_stats = pd.DataFrame(roi_records).set_index("ROI")
roi_stats = roi_stats.sort_values("cohens_d")

# === All whole-brain ROIs (same set as this histogram) vs ventricles, paired contrast ===
print("\n=== whole-brain ROIs vs ventricles (paired t-test) ===")
for roi in roi_stats.index:
    if roi == VENTRICLE_ROI:
        continue
    roi_effect = (active_group_summary[active_group_summary.ROI == roi]
                  .set_index("subject")
                  .eval("corr_same_context - corr_diff_context"))
    common = roi_effect.index.intersection(vent_effect.index)
    t, p = ttest_rel(roi_effect.loc[common], vent_effect.loc[common])
    diffs = roi_effect.loc[common] - vent_effect.loc[common]
    d = np.mean(diffs) / np.std(diffs, ddof=1)
    print(f"  [{roi} vs ventricles] t({len(common)-1})={t:.3f}, p={p:.3f}, d={d:.3f}, N={len(common)}")

# Asymmetric CI error bars
yerr = np.array([
    (roi_stats["cohens_d"] - roi_stats["ci_lo"]).values,
    (roi_stats["ci_hi"] - roi_stats["cohens_d"]).values
])

# Define colors per ROI: OFC red, rest gray
roi_colors = ["red" if roi == "OFC" else "gray" for roi in roi_stats.index]

# ROIs included in the original (Panel A) analysis get a purple frame
CORE_ROIS = ["VISUALFUNC_HIGHORDER", "FFA", "OFC", "HPC"]
roi_edgecolors = ["purple" if roi in CORE_ROIS else "none" for roi in roi_stats.index]
roi_linewidths = [1.3 if roi in CORE_ROIS else 0.3 for roi in roi_stats.index]

# Map ROI names for display
roi_labels = ["VTC" if roi == "VISUALFUNC_HIGHORDER" else roi for roi in roi_stats.index]
roi_labels = ["ventricles" if roi == "3&4VENTRICLES" else roi for roi in roi_labels]

# Plot bars with 95% bootstrap CI
axes_c[0].bar(
    range(len(roi_stats)),
    roi_stats["cohens_d"],
    yerr=yerr,
    error_kw={'elinewidth': 0.6},
    capsize=1.5,
    linewidth=roi_linewidths,
    edgecolor=roi_edgecolors,
    color=roi_colors)

# Horizontal line at y=0
axes_c[0].axhline(0, color="k", linestyle="--", linewidth=0.6)

# Remove top and right spines
for spine in ["top", "right"]:
    axes_c[0].spines[spine].set_visible(False)

# Axis label (smaller than the row-1 panels since this now shares half a row)
axes_c[0].set_ylabel("Cohen's d ± 95% CI", fontsize=8)
axes_c[0].yaxis.set_major_locator(MaxNLocator(nbins=4, prune=None))
axes_c[0].tick_params(axis='y', which='major', labelsize=7, pad=1)

# X-ticks rotated and labeled with updated names
axes_c[0].set_xticks(range(len(roi_labels)))
axes_c[0].set_xticklabels(roi_labels, rotation=90, fontsize=5.5)
axes_c[0].tick_params(axis='x', which='major', pad=1)
axes_c[0].set_xlim([-0.6, len(roi_labels) - 0.4])


#%% change window is marked when the consecutive X windows show an increase in accuracy 
# ----------------------------
# Parameters
# ----------------------------
window = 24     # sliding window length (trials)
stride = 24  # stride between windows
min_consec_increase = 3 # number of consecutive windows with increasing performance

# ----------------------------
# 1) Filter subjects and trials
# ----------------------------
learners_clean = [s.replace("sub-", "sub") for s in included_subjects]
grand_log = grand_log[
    (grand_log["sub"].isin(learners_clean)) &
    (grand_log["block_type_num"] < 12)].copy()

# ----------------------------
# 2) Add correctness column
# ----------------------------
grand_log["is_correct"] = (grand_log["response"] == grand_log["correct_response"]).astype(int)

# ----------------------------
# 3) Sliding window & find jump onset
# ----------------------------
def sliding_window_accuracy(acc, window=24, stride=1):
    """Compute sliding window means with given stride."""
    win_means, win_starts = [], []
    for start in range(0, len(acc)-window+1, stride):
        win_means.append(np.nanmean(acc[start:start+window]))
        win_starts.append(start)
    return np.array(win_means), np.array(win_starts)

def find_jump_onset_consecutive(acc, window=24, stride=1, min_consec=2):
    """
    Detect onset defined as:
    - first two consecutive diffs strictly > 0
    - followed by (min_consec - 2) diffs that are >= 0
    """
    win_means, win_starts = sliding_window_accuracy(acc, window, stride)
    if len(win_means) < min_consec + 1:
        return None, win_means, win_starts, None

    diffs = np.diff(win_means)

    run_len = 0
    onset_idx = None

    for i, d in enumerate(diffs):

        if run_len == 0:
            # need first strict increase
            if d > 0:
                run_len = 1
                onset_idx = i
            else:
                run_len = 0

        else:
            # after first strict increase: allow non-decrease only
            if d >= 0:
                run_len += 1
            else:
                run_len = 0
                onset_idx = None

        if run_len >= min_consec:
            return (
                int(win_starts[onset_idx + 1]),  # first window with increase
                win_means,
                win_starts,
                diffs[onset_idx:i+1].sum()
            )

    return None, win_means, win_starts, None

def find_jump_onset_strict(acc, window=24, stride=1, min_consec=2):
    """
    Detect onset defined as:
    - min_consec consecutive strictly positive increases in windowed accuracy
    - NO plateaus allowed (all diffs > 0)

    Returns:
        onset_trial (int or None)
        win_means (np.ndarray)
        win_starts (np.ndarray)
        cum_increase (float or None)
    """
    win_means, win_starts = sliding_window_accuracy(acc, window, stride)

    if len(win_means) < min_consec + 1:
        return None, win_means, win_starts, None

    diffs = np.diff(win_means)

    run_len = 0
    onset_idx = None

    for i, d in enumerate(diffs):
        if d > 0:
            if run_len == 0:
                onset_idx = i
            run_len += 1
        else:
            run_len = 0
            onset_idx = None

        if run_len >= min_consec:
            return (
                int(win_starts[onset_idx + 1]),  # first increased window
                win_means,
                win_starts,
                diffs[onset_idx:i+1].sum()
            )

    return None, win_means, win_starts, None

# ----------------------------
# 4) Build raw + aligned curves
# ----------------------------
raw_curves = []
aligned_subject_positions = []
aligned_subject_vals = []
subjects_included = []
max_jumps = []
jumps_onsets = []
mini_id_jumps_onsets = {}

for sub, df_sub in grand_log.groupby("sub"):
    df_l = df_sub[df_sub["is_passive"]==0].sort_values("block_type_num").reset_index(drop=True)
    acc = df_l["is_correct"].values.astype(float)

    raw_curves.append(acc)
    onset, win_means, win_starts, jump = find_jump_onset_strict(
        acc, window=window, stride=stride, min_consec=min_consec_increase)
    if (onset is None): # or (jump*100 < 20):  # optional jump threshold
        continue

    # aligned indices relative to maximal jump onset
    aligned_idx = np.arange(len(acc)) - onset
    aligned_subject_positions.append(aligned_idx)
    aligned_subject_vals.append(acc)
    subjects_included.append(sub)
    max_jumps.append(jump)
    jumps_onsets.append(onset)
    
    # compute mini_id onset
    base = onset // 24 *2 
    mini_id_onset = base if np.mod(onset,24) < 12 else base 
    mini_id_jumps_onsets[sub] = mini_id_onset

# ----------------------------
# 5) Build aligned matrix
# ----------------------------
if len(subjects_included) == 0:
    print("No subjects with valid aligned windows found. Skipping Panel C.")
else:
    global_min = min(pos.min() for pos in aligned_subject_positions)
    global_max = max(pos.max() for pos in aligned_subject_positions)
    aligned_grid = np.arange(global_min, global_max+1)
    n_cols = len(aligned_grid)
    n_subj = len(aligned_subject_positions)

    aligned_mat = np.full((n_subj, n_cols), np.nan)
    for i, (pos, vals) in enumerate(zip(aligned_subject_positions, aligned_subject_vals)):
        inds = pos - global_min  # shift to aligned matrix columns
        inds = inds.astype(int)
        valid_mask = (inds >= 0) & (inds < n_cols)
        aligned_mat[i, inds[valid_mask]] = vals[valid_mask]

    # raw matrix for trial-index-based mean
    Lmax = max(len(c) for c in raw_curves)
    raw_mat = np.full((len(raw_curves), Lmax), np.nan)
    for i, c in enumerate(raw_curves):
        raw_mat[i, :len(c)] = c

    # ----------------------------
    # 6) Compute mean & SEM
    # ----------------------------
    raw_mean = np.nanmean(raw_mat, axis=0)
    raw_sem = np.nanstd(raw_mat, axis=0) / np.sqrt(np.sum(~np.isnan(raw_mat), axis=0))

    aligned_mean = np.nanmean(aligned_mat, axis=0)
    aligned_sem = np.nanstd(aligned_mat, axis=0) / np.sqrt(np.sum(~np.isnan(aligned_mat), axis=0))


    # ----------------------------
    # 8) Optional: print pre/post zero means
    # ----------------------------
    pre_mask = aligned_grid < 0
    post_mask = aligned_grid >= 0
    pre_mean = np.nanmean(aligned_mean[pre_mask]) if pre_mask.any() else np.nan
    post_mean = np.nanmean(aligned_mean[post_mask]) if post_mask.any() else np.nan
    print(f"Aligned mean (pre-zero): {pre_mean:.3f}, (post-zero): {post_mean:.3f}, subjects={n_subj}")


# === Panel B: learning-aligned time series ===
if plot_learning_aligned_coupdate:
    window = np.arange(-2, 10)    # peri-event window relative to jump mini_id
    MIN_SUBJECTS = 20

    df = pivot_long.copy()

    for idx, (ax, roi) in enumerate(zip(axes_b[:1], ROIs_time_series)):
        aligned_all_subjects = []

        for sub in included_subjects:
            sub_clean = sub.replace("sub-", "sub")
            if sub_clean not in mini_id_jumps_onsets:
                continue
            jump_mini_id = mini_id_jumps_onsets[sub_clean]

            df_sub = df[(df["subject"] == sub) & (df["ROI"] == roi)].copy()
            if df_sub.empty:
                continue

            x = df_sub["mini_id"].to_numpy()
            y = df_sub["coupdate_diff"].to_numpy()

            aligned_trace = np.array([
                y[x == (jump_mini_id + w)][0] if (jump_mini_id + w) in x else np.nan
                for w in window
            ])
            aligned_trace = smooth_running_nan(aligned_trace, SMOOTH_WIN)
            aligned_all_subjects.append(aligned_trace)

        aligned_all_subjects = np.vstack(aligned_all_subjects)

        n_valid = np.sum(~np.isnan(aligned_all_subjects), axis=0)
        valid_mask = n_valid >= MIN_SUBJECTS

        mean_trace = np.where(valid_mask, np.nanmean(aligned_all_subjects, axis=0), np.nan)
        sem_trace  = np.where(valid_mask, np.nanstd(aligned_all_subjects, axis=0) / np.sqrt(n_valid), np.nan)

        # one-sample t-test per miniblock (uncorrected, p<0.05)
        sig_mask = np.zeros_like(mean_trace, dtype=bool)
        for i in range(len(window)):
            if not valid_mask[i]:
                continue
            vals = aligned_all_subjects[:, i]
            vals = vals[~np.isnan(vals)]
            if len(vals) > 1:
                _, p_val = ttest_1samp(vals, 0, nan_policy='omit')
                sig_mask[i] = p_val < alpha_sig

        ax.plot(window, mean_trace, color="black", lw=2)
        ax.fill_between(window, mean_trace - sem_trace, mean_trace + sem_trace,
                        color="black", alpha=0.25)
        ax.axhline(0, color="gray", linestyle="--", lw=1)

        for i in range(len(window)):
            if sig_mask[i]:
                ax.axvspan(window[i] - 0.5, window[i] + 0.5, color="yellow", alpha=0.3, zorder=0)

        ax.axvline(0, color="red", linestyle="--", linewidth=1)
        ax.set_title(ROIs_ttls[roi], fontweight="bold", fontsize=11)
        if idx == 0:
            ax.set_ylabel("Context-dependent co-update\n[same context - different contexts]", fontsize=10)
        else:
            ax.set_ylabel("")

        ax.yaxis.set_major_locator(MaxNLocator(nbins=max_ticks, prune=None))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if np.any(valid_mask):
            ax.set_xlim(window[valid_mask][0], window[valid_mask][-1])

    axes_b[0].set_xlabel("Learning-aligned segment (0 = learning onset)", fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "Figure4.pdf"), dpi=600, transparent=True)
plt.show()


#%% === Small figure: fMRI magnitude of change vectors (OFC & FFA) ===

_mag_csv = ('/Users/grossman/Volumes/STATE_FORMATION/MRI_DATA/MRI_REAL/'
            'ROI_timeseries_LSS_5mm_smoothing_dropout_30_physio_reg_out_updated/'
            'magnitude_change/magnitude_change_results_passive_trials.csv')
_mag_raw = pd.read_csv(_mag_csv)
_mag_raw = _mag_raw[_mag_raw['subject'].isin(included_subjects)].copy()

_learning_blocks_mag = [f'active{i}' for i in range(1, 12)]
_no_learn_blocks_mag = ['passive1', 'passive2']

# Per-subject mean per active block, non-feedbacked items only
_mag_learn = _mag_raw[
    (_mag_raw['change_type'] == 'learning') &
    (_mag_raw['block'].isin(_learning_blocks_mag))].copy()
_mag_learn['block_num'] = _mag_learn['block'].str.extract(r'(\d+)').astype(int)
_mag_subj_block = (_mag_learn
                   .groupby(['subject', 'ROI', 'block_num'])['mag_change']
                   .mean().reset_index())

# No-learning baseline (passive1 + passive2), per subject per ROI
_mag_nolearn_subj = (_mag_raw[
    (_mag_raw['change_type'] == 'no_learning') &
    (_mag_raw['block'].isin(_no_learn_blocks_mag))]
    .groupby(['subject', 'ROI'])['mag_change'].mean().reset_index())

_block_nums = np.arange(1, 12)

fig_mag, axes_mag = plt.subplots(1, len(ROIs_time_series),
                                  figsize=(4.8, 2.5), sharey=False)

for _idx, (_ax, _roi) in enumerate(zip(axes_mag, ROIs_time_series)):
    _roi_subs = _mag_subj_block[_mag_subj_block['ROI'] == _roi]['subject'].unique()

    _curves = []
    for _sub in _roi_subs:
        _sd = _mag_subj_block[
            (_mag_subj_block['ROI'] == _roi) & (_mag_subj_block['subject'] == _sub)]
        _c = np.full(11, np.nan)
        for _, _row in _sd.iterrows():
            _bn = int(_row['block_num'])
            if 1 <= _bn <= 11:
                _c[_bn - 1] = _row['mag_change']
        _curves.append(_c)

    _arr = np.vstack(_curves) if _curves else np.full((1, 11), np.nan)
    _mean = np.nanmean(_arr, axis=0)
    _n    = np.sum(~np.isnan(_arr), axis=0)
    _sem  = np.nanstd(_arr, axis=0) / np.sqrt(np.maximum(_n, 1))

    _nl_vals = _mag_nolearn_subj[_mag_nolearn_subj['ROI'] == _roi]['mag_change'].values
    _nl_mean = np.nanmean(_nl_vals)

    _ax.plot(_block_nums, _mean, lw=1.7, color='#8B6FB5')
    _ax.fill_between(_block_nums, _mean - _sem, _mean + _sem,
                     alpha=0.25, color='#8B6FB5')
    _ax.axhline(_nl_mean, ls='--', lw=1, color='gray', label='No-learning baseline')

    _ax.set_title(ROIs_ttls[_roi], fontweight='bold', fontsize=11)
    _ax.set_xlim(1, 11)
    _ax.set_xticks([1, 4, 7, 11])
    _ax.spines['top'].set_visible(False)
    _ax.spines['right'].set_visible(False)
    _ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    _ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    if _idx == 0:
        _ax.set_ylabel("Magnitude of\nchange [a.u.]", fontsize=10)

fig_mag.text(0.5, 0.01, 'Active block #', ha='center', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'Figure4_magnitude.pdf'), dpi=600, transparent=True)
plt.show()


#%% LME on the single co-update values (OFC): co-update as a function of context, house type and value difference
# (not shown in the paper)

import pandas as pd
import statsmodels.formula.api as smf


# --- 1. Data: same participants and co-update values as Fig. 4a (z-scored per participant x ROI) ---
df = _pair_df[_pair_df['ROI']=='OFC'].copy()

# Replace 12 & 13 → 12.5
df["val_diff"] = df["val_diff"].replace({12: 12.5, 13: 12.5})

# Replace 37 & 38 → 37.5
df["val_diff"] = df["val_diff"].replace({37: 37.5, 38: 37.5})

# Replace 62 & 63 → 62.5
df["val_diff"] = df["val_diff"].replace({62: 62.5, 63: 62.5})


df['context'] = df['state_from'].str[0]
# initialize
df["val_diff_rank"] = np.nan

# same_context == 1 → rank within each context
mask = df["same_context"] == 1
df.loc[mask, "val_diff_rank"] = (
    df.loc[mask]
      .groupby("context")["val_diff"]
      .transform(lambda x: x.rank(method="dense").astype(int))
)

# same_context == 0 → rank across all
mask0 = df["same_context"] == 0
df.loc[mask0, "val_diff_rank"] = (
    df.loc[mask0, "val_diff"]
    
      .rank(method="dense").astype(int)
)


# print mean number of rows per subject × val_diff_rank ---
count_table = (
    df.groupby(["val_diff_rank", "subject"])
      .size()
      .reset_index(name="n_rows")
)

mean_rows = (
    count_table.groupby("val_diff_rank")["n_rows"]
               .mean()
               .reset_index(name="mean_rows_per_subject")
)

print("\nMean number of rows per subject for each val_diff_rank:")
print(mean_rows)
print("\n")
# --- 2. Ensure variables are properly typed ---
df["same_context"] = df["same_context"].astype(float)
df["same_house_type"] = df["same_house_type"].astype(float)
df["val_diff"] = df["val_diff"].astype(float)
df["subject"] = df["subject"].astype("category")


# --- 3. Fit linear mixed-effects model ---
# Outcome: co-update (coupdate_var, as in Fig. 4a)
# Fixed effects: same_context, val_diff (centred), same_house_type
# Random intercept: subject
df["val_diff_c"] = df["val_diff"] - df["val_diff"].mean()
model = smf.mixedlm(
    f"{coupdate_var} ~ same_context + val_diff_c + same_house_type",
    df,
    groups=df["subject"]
)

result = model.fit(method="lbfgs")
print(result.summary())


