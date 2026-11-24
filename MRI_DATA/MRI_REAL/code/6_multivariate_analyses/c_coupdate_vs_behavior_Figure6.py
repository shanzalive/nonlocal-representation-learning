#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Figure 6 and Fig. S2b: co-updates and generalization of learning.
#
# Splits mini-blocks by whether learning generalized to the states without direct feedback
# and compares the context-dependent co-update between them.
# Input:  coupdate_LSS_zscored_miniblock_level_coupdate_values.csv (c_2nd_level_coupdate_analysis.py),
#         summary_files_behaviour/
# Output: Figure6.pdf
#
# Author: Shany Grossman

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import json
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
from scipy.stats import sem, ttest_rel, ttest_1samp, gaussian_kde
import sys
import os


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
# =============================================================================
# PATHS
# =============================================================================
import os
def _find_summary_files_behaviour():
    d = os.path.dirname(os.path.abspath(__file__))
    while not os.path.isdir(os.path.join(d, 'summary_files_behaviour')):
        if os.path.dirname(d) == d:
            raise FileNotFoundError('summary_files_behaviour/ not found above ' + __file__)
        d = os.path.dirname(d)
    return os.path.join(d, 'summary_files_behaviour') + os.sep
SUMMARY_DIR = _find_summary_files_behaviour()

sys.path.append(os.path.join(os.path.dirname(SUMMARY_DIR.rstrip(os.sep)), 'RNN_simulations', 'state_form_RNN'))
from stateformation_get_input_output import get_input_output_stateformation
from run_RNN_helper_funcs import rewardfun
csvpath = SUMMARY_DIR
save_dir = '/Users/grossman/Projects/STATE_FORMATION/FIGURES/PAPER_FIGURES/'

# =============================================================================
# PARAMETERS 
# =============================================================================
rois = ['inferiorparietal', 'FFA'] #'OFC','inferiorparietal']

conditions = [
    "no_generalization",
    "generalization",
    "saturated_learning"]

cond_labels = {
    "generalization": "Generalization",
    "no_generalization": "No\ngeneralization",
    "saturated_learning": "Saturated\nlearning",
    "negative_coupdate": "Negative\nco-update",
    "positive_coupdate": "Positive\nco-update"}

sign_conditions = ["negative_coupdate", "positive_coupdate"]

# Show p-values for all tests (including non-significant)
show_p_values = False
zscore_coupdate_vals = True  # z-score coupdate values per subject × ROI across all pairs before analysis

# Flag: for rows 1 & 3 (OFC / FFA), flip the analysis: group mini-blocks by the SIGN
# of their own context-dependent co-update (delta_corr, positive vs negative) instead
# of by generalization condition, and plot the mean generalization index (acc_change =
# acc_post - acc_pre) per group instead of delta_corr. Row 2 is unaffected.
PLOT_COUPDATE_SIGN_VS_GENERALIZATION = False
COUPDATE_SIGN_MARGIN = 0.1  # exclude mini-blocks with |delta_corr| <= this (z-scored units) from the sign split

# =============================================================================
# LOAD DATA
# =============================================================================
coupdate_df = pd.read_csv(
    "/Users/grossman/Volumes/STATE_FORMATION/MRI_DATA/MRI_REAL/"
    "ROI_timeseries_LSS_5mm_smoothing_dropout_30_physio_reg_out_updated/" 
    "coupdate_LSS/coupdate_LSS_zscored.csv")

_behaviour_df = pd.read_csv(csvpath + "State_Formation_behaviour.csv")

coupdate_single_subject  = pd.read_csv(
    "/Users/grossman/Volumes/STATE_FORMATION/MRI_DATA/MRI_REAL/"
    "ROI_timeseries_LSS_5mm_smoothing_dropout_30_physio_reg_out_updated/" 
    "coupdate_LSS/coupdate_LSS_zscored_miniblock_level_coupdate_values.csv") 

if zscore_coupdate_vals:
    _stats = (coupdate_single_subject.groupby(["subject", "ROI"])["corr"]
              .agg(["mean", "std"]).reset_index())
    coupdate_single_subject = coupdate_single_subject.merge(_stats, on=["subject", "ROI"])
    coupdate_single_subject["corr"] = ((coupdate_single_subject["corr"] - coupdate_single_subject["mean"])   / coupdate_single_subject["std"])
                                     #  / coupdate_single_subject["std"])
    coupdate_single_subject = coupdate_single_subject.drop(columns=["mean", "std"])
    # Recompute group summary from z-scored pair data (replaces CSV-loaded coupdate_df)
    _subj = (coupdate_single_subject.groupby(["subject", "ROI", "same_context"])["corr"]
             .mean().reset_index())
    _same = (_subj[_subj["same_context"] == 1]
             .rename(columns={"corr": "corr_same_context"}).drop(columns="same_context"))
    _diff = (_subj[_subj["same_context"] == 0]
             .rename(columns={"corr": "corr_diff_context"}).drop(columns="same_context"))
    coupdate_df = _same.merge(_diff, on=["subject", "ROI"])

#%%

def summarize_corrs_per_miniblock(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize correlation metrics per subject, ROI, block, and mini-block.
    Returns a DataFrame with columns:
    ['sub', 'ROI', 'block', 'i_mini_block', 'delta_corr', 'same_context_corr',
     'diff_context_corr', 'corr_var_name']
    """
    summary_list = []

    corr_vars = ["corr"] 
    
    # Loop over correlation variables
    for corr_var in corr_vars:
        # Group by subject, ROI, block, and mini-block
        group_cols = ['subject', 'ROI', 'block', 'imini_block']
        grouped = df.groupby(group_cols)
        
        for (sub, roi, block, i_mini), group in grouped:


            # Mean corr where same_context == 1
            same_context_corr = group.loc[group['same_context'] == 1, corr_var].mean()
            # Mean corr where same_context == 0
            diff_context_corr = group.loc[group['same_context'] == 0, corr_var].mean()
            
            # Compute delta_corr i.e. coupdate for this miniblock
            delta_corr = same_context_corr-diff_context_corr


            summary_list.append({
                'sub': sub,
                'ROI': roi,
                'block': block-1,
                'i_mini_block': i_mini,
                'delta_corr': delta_corr,
                'same_context_corr': same_context_corr,
                'diff_context_corr': diff_context_corr,
                'corr_var_name': corr_var
            })
    
    summary_df = pd.DataFrame(summary_list)
    return summary_df

coupdate_per_miniblock_df = summarize_corrs_per_miniblock(coupdate_single_subject)

#%%
# =============================================================================
# SUBJECTS
# =============================================================================
_ldict_path = SUMMARY_DIR + 'learners_dict.json'
with open(_ldict_path) as _f:
    _learners_dict = json.load(_f)
_in_csv = set(coupdate_df["subject"].unique())
subjects = np.array(sorted([s for s in _learners_dict['full_and_partial']
                             if s in _in_csv]))
print(f"Including {len(subjects)} subjects (full_and_partial ∩ CSV).")

coupdate_df = coupdate_df[coupdate_df["subject"].isin(subjects)].copy()
coupdate_df["subject"] = coupdate_df["subject"].str.replace("-", "")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def add_acc_change(coupdate_per_miniblock_df, rois, subjects):
    # Build subject map: "sub-501" -> "sub501" (CSV uses no-dash format)
    sub_map = {s: s.replace("-", "") for s in subjects}

    df = _behaviour_df[
        (_behaviour_df["block_type_num"] < 12) &
        (_behaviour_df["sub"].isin(sub_map.values()))
    ].copy()

    # Within-block trial index (passive + active combined, preserving original order)
    df["itrial_inblock"] = df.groupby(["sub", "block_type_num"]).cumcount()

    # Assign passive_type vectorized using within-block index thresholds
    passive = df[df["is_passive"] == 1].copy()
    i = passive["itrial_inblock"]
    is_old = passive["sub"].isin(["sub501", "sub502"])

    pt = pd.Series(pd.NA, index=passive.index, dtype=object)
    pt[i < 5] = "pre_early"
    pt[(i > 7) & (i < 12)] = "post_early"
    pt[ is_old & (i > 27) & (i < 32)] = "pre_late"
    pt[~is_old & (i > 27) & (i < 33)] = "pre_late"
    pt[ is_old & (i > 35) & (i < 40)] = "post_late"
    pt[~is_old & (i > 36) & (i < 42)] = "post_late"
    passive["passive_type"] = pt

    # Keep only valid probes (response must be more/less)
    valid = passive[
        passive["passive_type"].notna() &
        passive["response"].isin(["more", "less"])
    ]

    # Mean accuracy per (sub, block_type_num, passive_type)
    acc = (
        valid.groupby(["sub", "block_type_num", "passive_type"])["is_correct"]
        .mean()
        .unstack("passive_type")
    )

    mini_blocks = ["early", "late"]

    for sub, sub_clean in sub_map.items():
        if sub_clean not in acc.index.get_level_values("sub"):
            continue
        sub_acc = acc.xs(sub_clean, level="sub")

        for block_num in sub_acc.index:
            for i_mb, mb in enumerate(mini_blocks):
                pre_col, post_col = f"pre_{mb}", f"post_{mb}"
                acc_pre  = sub_acc.loc[block_num, pre_col]  if pre_col  in sub_acc.columns else np.nan
                acc_post = sub_acc.loc[block_num, post_col] if post_col in sub_acc.columns else np.nan

                if pd.notna(acc_pre) and pd.notna(acc_post):
                    acc_change = acc_post - acc_pre
                else:
                    acc_pre = acc_post = acc_change = np.nan

                # Apply to all ROIs at once (result is ROI-independent)
                mask = (
                    (coupdate_per_miniblock_df["sub"] == sub) &
                    (coupdate_per_miniblock_df["block"] == block_num) &
                    (coupdate_per_miniblock_df["i_mini_block"] == i_mb)
                )
                coupdate_per_miniblock_df.loc[
                    mask, ["acc_change", "acc_pre", "acc_post"]
                ] = (acc_change, acc_pre, acc_post)

    return coupdate_per_miniblock_df


def assign_condition(row):
    if row["acc_change"] > 0:
        return "generalization"
    elif row["acc_change"] < 0 or (row["acc_change"]==  0 and row["acc_pre"] == 0):
        return "no_generalization"
    elif row["acc_change"] == 0 and row["acc_pre"] == 1 and row["acc_post"] == 1:
        return "saturated_learning"
    else:
        return np.nan


def split_kde(ax, values, center_x, mirror, color, alpha=0.35, scale=0.25):
    values = np.asarray(values)
    values = values[~np.isnan(values)]
    if len(values) < 3:
        return
    kde = gaussian_kde(values)
    y = np.linspace(values.min(), values.max(), 300)
    d = kde(y)
    d = d / d.max() * scale
    x = center_x - d if mirror == "left" else center_x + d
    ax.fill_betweenx(y, center_x, x, color=color, alpha=alpha, lw=0)


def significance_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""


def _tick_no_lead_zero(x, pos):
    if x == 0:
        return "0"
    if abs(x) < 1:
        s = f"{x:.2g}"
        if s.startswith("0."):   return s[1:]
        if s.startswith("-0."): return "-" + s[2:]
        return s
    return f"{x:.4g}"

_lz_fmt = FuncFormatter(_tick_no_lead_zero)

def _fmt_p(p):
    if p < 0.001:
        return "p<.001"
    s = f"{p:.2g}"
    return "p=" + (s[1:] if s.startswith("0.") else s)


# =============================================================================
# PREP DATA
# =============================================================================
coupdate_per_miniblock_df = add_acc_change(
    coupdate_per_miniblock_df, rois, subjects
)

coupdate_per_miniblock_df['corr_diff_context'] = coupdate_per_miniblock_df['diff_context_corr'].copy()
coupdate_per_miniblock_df['corr_same_context'] = coupdate_per_miniblock_df['same_context_corr'].copy()
coupdate_per_miniblock_df["condition"] = (
    coupdate_per_miniblock_df.apply(assign_condition, axis=1)
)
# Sign of the mini-block's own context-dependent co-update, used by PLOT_COUPDATE_SIGN_VS_GENERALIZATION
# Segments with |delta_corr| <= COUPDATE_SIGN_MARGIN are excluded (left as NaN) rather than
# split, to avoid classifying near-zero (essentially noise-level) co-update as positive/negative.
_coupdate_sign = pd.Series(np.nan, index=coupdate_per_miniblock_df.index, dtype=object)
_coupdate_sign[coupdate_per_miniblock_df["delta_corr"] > COUPDATE_SIGN_MARGIN] = "positive_coupdate"
_coupdate_sign[coupdate_per_miniblock_df["delta_corr"] < -COUPDATE_SIGN_MARGIN] = "negative_coupdate"
coupdate_per_miniblock_df["coupdate_sign"] = _coupdate_sign

plot_df = coupdate_per_miniblock_df.dropna(
    subset=["condition", "ROI", "sub", "delta_corr"]
)
plot_df_sign = coupdate_per_miniblock_df.dropna(
    subset=["coupdate_sign", "ROI", "sub", "acc_change"]
)
plot_df_sign = plot_df_sign[plot_df_sign["acc_change"] != 0]

#%% FIGURE
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.colors import to_rgb

connect_subjects = True  # Flag to connect single subjects with lines in paired plots

# Define condition colors for all ROIs (adjustable)
roi_colors = {
    rois[0]: 'k',
    rois[1]: 'k'}

panel_2_colors = {
    'corr_diff_context':'darkblue',
    'corr_same_context':'#5BC8C0'}


fig = plt.figure(figsize=(4.5, 10))
outer_gs = GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.75)  # row3 slightly taller

# ---------------- Row definitions ----------------
gs_row1 = GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_gs[0])
gs_row2 = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_gs[1], wspace=0.8)
gs_row3 = GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_gs[2])


# ---------------- Panel definitions ----------------
row1_panels = [
    (0, 0,  rois[0], "delta_corr", "Context-dependent\nco-update [r]", 2, 4, 2),
]

row2_panels = [
    (1, 0,  rois[0], "corr_diff_context", "Cross-contexts\nco-update [r]", None, -2, 0),
    (1, 1,  rois[0], "corr_same_context", "Within-context\nco-update [r]", None, -2, 0),
]

row3_panels = [
    (2, 0,  rois[1], "delta_corr", "Context-dependent\nco-update [r]", 2, 4, 2),
]

all_panels = [(gs_row1, row1_panels), (gs_row2, row2_panels), (gs_row3, row3_panels)]

for row_idx, (gs_sub, panels) in enumerate(all_panels):
    for (_, col, roi, dv, ylabel, x_tick_pad, y_label_pad, y_tick_pad) in panels:

        ax = fig.add_subplot(gs_sub[0, col])
        roi_color = roi_colors.get(roi, "#4C72B0")

        use_sign_split = PLOT_COUPDATE_SIGN_VS_GENERALIZATION and row_idx != 1
        if use_sign_split:
            group_col = "coupdate_sign"
            conds = sign_conditions
            dv = "acc_change"
            ylabel = "Generalization index\n[Δ accuracy]"
            df_roi = plot_df_sign[plot_df_sign["ROI"] == roi]
        else:
            group_col = "condition"
            conds = conditions if row_idx != 1 else ["no_generalization", "generalization"]
            df_roi = plot_df[plot_df["ROI"] == roi]

        subj_means_full = (
            df_roi.groupby(["sub", group_col])[dv]
            .mean()
            .unstack())

        subj_means = subj_means_full.loc[:, conds]

        xpos = np.arange(len(conds))
        vals = subj_means.values.flatten()
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue
        ymin, ymax = np.min(vals), np.max(vals)
        yr = ymax - ymin if ymax != ymin else 1e-6
        yr = ymax - ymin
        ax.set_ylim(ymin - 0.02 * yr, ymax + 0.02 * yr)
        y_star = ymax*1.15*.9

        # Get colors for each condition
        if row_idx != 1:
            base_color = roi_color
        else:
            base_color = panel_2_colors[dv]

        cond_colors = [base_color for _ in conds]

        # prepare swarmplot data
        df_plot = subj_means.reset_index().melt(id_vars="sub", value_vars=conds,
                                                var_name="condition", value_name=dv)
        df_plot["x"] = df_plot["condition"].map({c: i for i, c in enumerate(conds)})

        # plot swarm points by condition color
        for i, cond in enumerate(conds):
            sns.swarmplot(
                x="x", y=dv, data=df_plot[df_plot["condition"]==cond],
                ax=ax, size=3 if row_idx!=1 else 2.5,
                color=cond_colors[i], dodge=True, zorder=1)


        if connect_subjects:
            x_start = xpos + 0.3
            x_end   = xpos - 0.3

            # for consecutive connections, we want:
            # x_start[i] -> x_end[i+1]
            for sub in subj_means.index:
                y_vals = subj_means.loc[sub].values
                for i in range(len(xpos) - 1):
                    ax.plot(
                        [x_start[i], x_end[i + 1]],
                        [y_vals[i], y_vals[i + 1]],
                        color=base_color,
                        lw=0.8,
                        alpha=0.3,
                        zorder=0)

        # stars for 1-sample t-tests
        for i, cond in enumerate(conds):
            vals = subj_means[cond].dropna().values
            if len(vals) > 1:
                t0, p0 = ttest_1samp(vals, 0)
                d0 = np.mean(vals) / np.std(vals, ddof=1)
                print(f"[1-sample] {roi} {dv} {cond}: t({len(vals) - 1})={t0:.3f}, p={p0:.4g}, d={d0:.3f}")
                ax.text(i, y_star, significance_stars(p0),
                        ha="center", va="bottom", fontsize=12, fontweight="bold")


        _means = subj_means.mean()
        for _i in range(len(xpos) - 1):
            ax.plot([xpos[_i] + 0.3, xpos[_i + 1] - 0.3],
                    [_means.iloc[_i], _means.iloc[_i + 1]],
                    color='darkred', lw=1.5, zorder=3)

        ax.axhline(0, ls="--", lw=0.8, color="k", zorder=0)
        ax.set_xticks(xpos)
        ax.set_xticklabels([cond_labels[c] for c in conds], rotation=0 if row_idx==1 else 0, fontsize=9)

        # Remove overarching X label
        ax.set_xlabel("")

        if y_tick_pad is not None:
            ax.tick_params(axis="y", pad=y_tick_pad)

        if x_tick_pad is not None:
            ax.tick_params(axis="x", pad=x_tick_pad)
        if ylabel:
            ax.set_ylabel(ylabel, labelpad=y_label_pad)

        # ---------- paired t-tests ----------

        cap = 0.03 * yr
        text_gap = 0.03 * yr  # increased spacing between multiple tests
        sig_count = 0
        for (i1, c1), (i2, c2) in itertools.combinations(enumerate(conds), 2):
            v1, v2 = subj_means[c1], subj_means[c2]
            paired = pd.concat([v1, v2], axis=1).dropna()

            if len(paired) > 1:
                t, p = ttest_rel(paired.iloc[:,0], paired.iloc[:,1])
                diffs = paired.iloc[:,0].values - paired.iloc[:,1].values
                d = np.mean(diffs) / np.std(diffs, ddof=1)
                print(f"[paired] {roi} {dv} {c1} vs {c2}: t({len(paired) - 1})={t:.3f}, p={p:.4g}, d={d:.3f}")
                if np.round(p,2) <= 0.1:
                    y = y_star + 0.18 * yr
                    ax.plot([i1, i2], [y, y], color="k", lw=1, clip_on=False)
                    ax.plot([i1, i1], [y - cap, y], color="k", lw=1, clip_on=False)
                    ax.plot([i2, i2], [y - cap, y], color="k", lw=1, clip_on=False)
                    ax.text((i1 + i2)/2, y_star + 0.10 * yr, significance_stars(p),
                            ha="center", va="bottom", fontsize=14, fontweight="bold", clip_on=False)
                    ax.text((i1 + i2)/2, y - 0.02 * yr, _fmt_p(p),
                            ha="center", va="top", fontsize=6, clip_on=False)
                    sig_count += 1

        sns.despine(ax=ax)

        if row_idx == 1:
            ax.tick_params(axis="x", labelsize=8)
            ax.set_xlabel(ax.get_xlabel(), fontsize=8)
            ax.set_ylabel(ax.get_ylabel(), fontsize=8)

        ax.yaxis.set_major_formatter(_lz_fmt)

# ---------------- Row titles ----------------
fig.text(0.5, 0.93,  rois[0], ha="center", va="top", fontsize=11, fontweight="bold")
fig.text(0.5, 0.32,  rois[1], ha="center", va="top", fontsize=11, fontweight="bold")

# ---------------- Row 3 Headlines ----------------
fig.text(0.30, 0.62, "Cross-contexts push", fontsize=10, fontweight="bold", ha="center", color = panel_2_colors['corr_diff_context'])
fig.text(0.70, 0.62, "Within-context pull", fontsize=10, fontweight="bold", ha="center", color = panel_2_colors['corr_same_context'])

plt.subplots_adjust(left=0.18)  # room for the y-axis labels at the narrower (3.5in) figure width
plt.savefig(os.path.join(save_dir, "Figure6.pdf"), dpi=600, transparent=True)
plt.show()


