#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Figure S3: robustness of the OFC context-dependent co-update (multiverse analysis).
#
# Repeats the Fig. 4a OFC test (same vs different context co-update) for each
# preprocessing / analysis variant ("universe") and plots Cohen's d and the p-value of each.
# Each universe is defined by:
#   - dir_name: which ROI_timeseries_* folder to read the mini-block co-update csv from
#     (GLM / extraction choices: CompCor, dropout threshold, smoothing)
#   - coupdate_var: which column of that csv to use as the co-update value
#   - zscore, balance: statistical choices of Fig. 4 (z-scoring per participant x ROI;
#     matching the number of different-context pairs to the same-context pairs)
# Input:  the mini-block co-update csv of every universe (c_2nd_level_coupdate_analysis.py)
# Output: Figure S3 pdf
#
# Author: Shany Grossman

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.stats import ttest_rel, ttest_1samp, t as t_dist
from collections import Counter

# %% === Config ===

BASE_DIR = '/Users/grossman/Volumes/STATE_FORMATION/MRI_DATA/MRI_REAL/'
import os
def _find_summary_files_behaviour():
    d = os.path.dirname(os.path.abspath(__file__))
    while not os.path.isdir(os.path.join(d, 'summary_files_behaviour')):
        if os.path.dirname(d) == d:
            raise FileNotFoundError('summary_files_behaviour/ not found above ' + __file__)
        d = os.path.dirname(d)
    return os.path.join(d, 'summary_files_behaviour') + os.sep
SUMMARY_DIR = _find_summary_files_behaviour()

LEARNERS_DICT_PATH = SUMMARY_DIR + 'learners_dict.json'
SAVE_DIR = '/Users/grossman/Projects/STATE_FORMATION/FIGURES/PAPER_FIGURES/'

ROI = 'OFC'

MINIBLOCK_CSV_NAME_TMPL = '{analysis}_zscored_miniblock_level_coupdate_values.csv'

# --- Display titles: what each universe is actually called in the figure (x-tick labels,
# Panel C subplot titles). Keyed by each UNIVERSES entry's `key` (a stable internal id, never
# shown). Only used for display.
UNIVERSE_TITLES = {
    'default': 'Original',
    'compcor': 'CompCor\nphysiological\nnoise correction',
    'compcor_combined': 'CompCor\n(combined\nWM+CSF mask)',
    'dropout50': 'Dropout\nthreshold:\n50%',
    'no_dropout_threshold': 'No dropout',
    'ordinal': 'Ordinal\ncorrelation',
    'no_zscore': 'No z-scoring\nof co-update\nvalues',
    'balanced_pairs': 'Balanced\npair\ncounts',
    'dropout80': 'Dropout\nthreshold:\n80%',
    'smoothing_3mm': '3mm\nsmoothing',
    'no_smoothing': 'No\nsmoothing',
}

# --- Universes: each dict = one point in the multiverse plot ---
# `key` must match a key in UNIVERSE_TITLES above (display text lives there, not here).
# `is_default` marks the arm matching the published Figure 4 pipeline.
UNIVERSES = [
    dict(key='default',
         dir_name='ROI_timeseries_LSS_5mm_smoothing_dropout_30_physio_reg_out_updated',
         coupdate_var='corr', zscore=True, balance=False, is_default=True),

    dict(key='compcor',
         # CompCor only (add_physio_regressors=False, add_compcor=True,
         # compcor_strategy='anat_separated' in runGLM_nilearn.py)
         dir_name='ROI_timeseries_LSS_5mm_smoothing_dropout_30_compcorr',
         coupdate_var='corr', zscore=True, balance=False, is_default=False),

    dict(key='dropout50',
         dir_name='ROI_timeseries_LSS_5mm_smoothing_dropout_50_physio_reg_out_updated',
         coupdate_var='corr', zscore=True, balance=False, is_default=False),

    dict(key='no_dropout_threshold',
         dir_name='ROI_timeseries_LSS_5mm_smoothing_physio_reg_out_updated',
         coupdate_var='corr', zscore=True, balance=False, is_default=False),

    dict(key='ordinal',
         dir_name='ROI_timeseries_LSS_5mm_smoothing_dropout_30_physio_reg_out_updated',
         coupdate_var='corr_ordinal', zscore=True, balance=False, is_default=False),

    dict(key='no_zscore',
         dir_name='ROI_timeseries_LSS_5mm_smoothing_dropout_30_physio_reg_out_updated',
         coupdate_var='corr', zscore=False, balance=False, is_default=False),

    dict(key='balanced_pairs',
         dir_name='ROI_timeseries_LSS_5mm_smoothing_dropout_30_physio_reg_out_updated',
         coupdate_var='corr', zscore=True, balance=True, is_default=False),

    dict(key='dropout80',
         dir_name='ROI_timeseries_LSS_5mm_smoothing_dropout_80_physio_reg_out_updated',
         coupdate_var='corr', zscore=True, balance=False, is_default=False),

    dict(key='smoothing_3mm',
         dir_name='ROI_timeseries_LSS_3mm_smoothing_dropout_30_physio_reg_out_updated',
         coupdate_var='corr', zscore=True, balance=False, is_default=False),

]

N_BALANCE_BOOT = 5  # bootstrap iterations for balance=True universes, matches
# c_contextdep_coupdate_Figure4.py's balance_pair_counts design: for each miniblock, draw N_BALANCE_BOOT
# independent random matched subsamples of the diff-context pairs (same count as that
# miniblock's same-context pairs), average their miniblock-level means together, and run a
# single downstream test on that one averaged value per group. Note this is a bias/variance
# knob, not a free lunch: as N_BALANCE_BOOT grows, the averaged diff-context value converges
# toward the full, unsubsampled population mean of all diff-context pairs (a basic
# unbiasedness property of sampling without replacement: the average of the sample mean
# over every possible same-size subset exactly equals the population mean), which recovers
# more precision than the same-context side ever has and edges back toward the unbalanced
# comparison this whole procedure exists to avoid. Smaller N_BALANCE_BOOT keeps each
# miniblock's diff-context estimate closer to the same noise level as same-context (matching
# the point of the exercise) at the cost of more run-to-run variance in the final number.

# --- Panel B config: "behavioral link to generalization". Paired t-test on
# context-dependent co-update (delta_corr) between mini-blocks where behavior generalized
# (acc_change > 0) vs did not (acc_change < 0), per universe. acc_change is computed as in
# c_coupdate_vs_behavior_Figure6.py's add_acc_change().
RNN_CSV_PATH = SUMMARY_DIR

# --- Timeseries column config: context-dependent co-update across learning segments, per
# universe (ported from c_contextdep_coupdate_Figure4.py's Panel B: coupdate_diff = same_context minus
# diff_context per chronological mini-block index 'mini_id', smoothed per subject) ---
SMOOTH_WIN = 4  # centered moving-average window over mini_id, matches c_contextdep_coupdate_Figure4.py
TS_ALPHA_SIG = 0.05  # uncorrected 1-sample t-test threshold for the yellow significance shading

doplotTimeseries = True  # if True, compute + add the right-hand column of per-universe learning-segment timeseries panels


def smooth_running_nan(x, w):
    """Simple centered moving average (nan-safe). w must be >=1. stride=1.
    Ported verbatim from c_contextdep_coupdate_Figure4.py."""
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

plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'Arial',
    'pdf.fonttype': 42,
    'font.size': 8,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
})

# %% === Helpers ===

with open(LEARNERS_DICT_PATH) as _f:
    _learners_dict = json.load(_f)
_full_and_partial = set(_learners_dict['full_and_partial'])

_pair_df_cache = {}


def load_pair_df(dir_name, analysis='coupdate_LSS'):
    """Load the miniblock-level co-update CSV for one extraction directory.
    analysis picks the output subfolder + filename prefix."""
    cache_key = (dir_name, analysis)
    if cache_key not in _pair_df_cache:
        csv_name = MINIBLOCK_CSV_NAME_TMPL.format(analysis=analysis)
        path = os.path.join(BASE_DIR, dir_name, analysis, csv_name)
        df = pd.read_csv(path)
        _pair_df_cache[cache_key] = df
    return _pair_df_cache[cache_key]


def compute_balanced_miniblock_means(pair_df, coupdate_var, n_boot=N_BALANCE_BOOT, seed=42):
    """Bootstrap-subsample diff-context pairs per miniblock to match same-context
    pair count. Ported from c_contextdep_coupdate_Figure4.py (kept in sync with Panel A)."""
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


_behaviour_df = pd.read_csv(RNN_CSV_PATH + "State_Formation_behaviour.csv")


def build_acc_change_lookup(subjects):
    """(subject, block_num, i_mini_block) -> acc_change (post - pre passive-probe accuracy).
    Same as add_acc_change() in c_coupdate_vs_behavior_Figure6.py. Purely behavioral and
    independent of ROI/universe, so computed once and reused across all universes below."""
    sub_map = {s: s.replace("-", "") for s in subjects}
    df = _behaviour_df[
        (_behaviour_df["block_type_num"] < 12) &
        (_behaviour_df["sub"].isin(sub_map.values()))
    ].copy()
    df["itrial_inblock"] = df.groupby(["sub", "block_type_num"]).cumcount()

    passive = df[df["is_passive"] == 1].copy()
    i = passive["itrial_inblock"]
    is_old = passive["sub"].isin(["sub501", "sub502"])

    pt = pd.Series(pd.NA, index=passive.index, dtype=object)
    pt[i < 5] = "pre_early"
    pt[(i > 7) & (i < 12)] = "post_early"
    pt[is_old & (i > 27) & (i < 32)] = "pre_late"
    pt[~is_old & (i > 27) & (i < 33)] = "pre_late"
    pt[is_old & (i > 35) & (i < 40)] = "post_late"
    pt[~is_old & (i > 36) & (i < 42)] = "post_late"
    passive["passive_type"] = pt

    valid = passive[passive["passive_type"].notna() & passive["response"].isin(["more", "less"])]
    acc = (valid.groupby(["sub", "block_type_num", "passive_type"])["is_correct"]
           .mean().unstack("passive_type"))

    lookup = {}
    mini_blocks = ["early", "late"]
    for sub, sub_clean in sub_map.items():
        if sub_clean not in acc.index.get_level_values("sub"):
            continue
        sub_acc = acc.xs(sub_clean, level="sub")
        for block_num in sub_acc.index:
            for i_mb, mb in enumerate(mini_blocks):
                pre_col, post_col = f"pre_{mb}", f"post_{mb}"
                acc_pre = sub_acc.loc[block_num, pre_col] if pre_col in sub_acc.columns else np.nan
                acc_post = sub_acc.loc[block_num, post_col] if post_col in sub_acc.columns else np.nan
                if pd.notna(acc_pre) and pd.notna(acc_post):
                    lookup[(sub, block_num, i_mb)] = acc_post - acc_pre
    return lookup


ALL_ACC_CHANGE_LOOKUP = build_acc_change_lookup(sorted(_full_and_partial))


def compute_universe_stats(universe):
    """Same vs diff context paired t-test + Cohen's d for OFC, matching
    c_contextdep_coupdate_Figure4.py Panel A exactly (z-score -> [balance] -> miniblock mean
    -> subject mean -> paired test)."""
    coupdate_var = universe['coupdate_var']
    raw_df = load_pair_df(universe['dir_name'], universe.get('analysis', 'coupdate_LSS'))

    included_subjects = sorted(
        s for s in raw_df['subject'].unique()
        if s in _full_and_partial
    )

    pair_df = raw_df[(raw_df['ROI'] == ROI) & (raw_df['subject'].isin(included_subjects))].copy()

    if universe['zscore']:
        stats = (pair_df.groupby(['subject', 'ROI'])[coupdate_var]
                 .agg(['mean', 'std']).reset_index())
        pair_df = pair_df.merge(stats, on=['subject', 'ROI'])
        pair_df[coupdate_var] = (pair_df[coupdate_var] - pair_df['mean']) / pair_df['std']
        pair_df = pair_df.drop(columns=['mean', 'std'])

    def _stats_from_mb_means(mb_means):
        subj_means = (mb_means
                      .groupby(['subject', 'ROI', 'same_context'])[[coupdate_var]]
                      .mean().reset_index())
        same = (subj_means[subj_means['same_context'] == 1]
                .set_index('subject')[coupdate_var])
        diff = (subj_means[subj_means['same_context'] == 0]
                .set_index('subject')[coupdate_var])
        common_subjects = same.index.intersection(diff.index)
        same_vals = same.loc[common_subjects].values
        diff_vals = diff.loc[common_subjects].values

        mask = ~np.isnan(same_vals) & ~np.isnan(diff_vals)
        tval, pval = ttest_rel(diff_vals[mask], same_vals[mask])
        diffs = same_vals[mask] - diff_vals[mask]
        dval = np.mean(diffs) / np.std(diffs, ddof=1)
        return tval, pval, dval, int(mask.sum())

    if universe['balance']:
        mb_means = compute_balanced_miniblock_means(pair_df, coupdate_var)
    else:
        mb_means = (pair_df
                    .groupby(['subject', 'ROI', 'runType', 'imini_block', 'same_context'])[[coupdate_var]]
                    .mean().reset_index())
    tval, pval, d, n = _stats_from_mb_means(mb_means)

    return dict(key=universe['key'], is_default=universe['is_default'],
                t=tval, p=pval, d=d, n=n)


def compute_universe_generalization_stats(universe):
    """Cohen's d for the 'behavioral link to generalization' effect: paired t-test on
    context-dependent co-update (delta_corr = same_context minus diff_context corr, at the
    miniblock level) between mini-blocks where behavior generalized (acc_change > 0) vs did
    not (acc_change < 0). Mini-blocks with acc_change == 0 are excluded (neither group).
    Reuses the same z-score/balance choices as compute_universe_stats so both panels stay
    internally consistent per universe."""
    coupdate_var = universe['coupdate_var']
    raw_df = load_pair_df(universe['dir_name'], universe.get('analysis', 'coupdate_LSS'))

    included_subjects = sorted(
        s for s in raw_df['subject'].unique()
        if s in _full_and_partial
    )
    pair_df = raw_df[(raw_df['ROI'] == ROI) & (raw_df['subject'].isin(included_subjects))].copy()

    if universe['zscore']:
        stats = (pair_df.groupby(['subject', 'ROI'])[coupdate_var]
                 .agg(['mean', 'std']).reset_index())
        pair_df = pair_df.merge(stats, on=['subject', 'ROI'])
        pair_df[coupdate_var] = (pair_df[coupdate_var] - pair_df['mean']) / pair_df['std']
        pair_df = pair_df.drop(columns=['mean', 'std'])

    def _stats_from_mb_means(mb_means):
        mb_means = mb_means.copy()
        # block number is 1:1 with runType ('active1' -> 1, ...); block_type_num in the
        # behavioral CSV is 0-indexed, hence block - 1 in the lookup below.
        mb_means['block'] = mb_means['runType'].str.replace('active', '', regex=False).astype(int)

        mb_wide = (mb_means
                   .set_index(['subject', 'block', 'imini_block', 'same_context'])[coupdate_var]
                   .unstack('same_context')
                   .rename(columns={1: 'same_context_corr', 0: 'diff_context_corr'})
                   .dropna()
                   .reset_index())
        mb_wide['delta_corr'] = mb_wide['same_context_corr'] - mb_wide['diff_context_corr']

        mb_wide['acc_change'] = mb_wide.apply(
            lambda row: ALL_ACC_CHANGE_LOOKUP.get((row['subject'], row['block'] - 1, row['imini_block'])),
            axis=1)
        mb_wide = mb_wide.dropna(subset=['acc_change'])

        group = pd.Series(np.nan, index=mb_wide.index, dtype=object)
        group[mb_wide['acc_change'] > 0] = 'positive_acc_change'
        group[mb_wide['acc_change'] < 0] = 'negative_acc_change'
        mb_wide['acc_change_group'] = group
        mb_wide = mb_wide.dropna(subset=['acc_change_group'])

        subj_means = (mb_wide.groupby(['subject', 'acc_change_group'])['delta_corr']
                      .mean().unstack('acc_change_group'))
        paired = subj_means.reindex(columns=['positive_acc_change', 'negative_acc_change']).dropna()

        tval, pval = ttest_rel(paired['positive_acc_change'], paired['negative_acc_change'])
        diffs = paired['positive_acc_change'].values - paired['negative_acc_change'].values
        dval = np.mean(diffs) / np.std(diffs, ddof=1)
        return tval, pval, dval, len(paired)

    if universe['balance']:
        mb_means = compute_balanced_miniblock_means(pair_df, coupdate_var)
    else:
        mb_means = (pair_df
                    .groupby(['subject', 'ROI', 'runType', 'imini_block', 'same_context'])[[coupdate_var]]
                    .mean().reset_index())
    tval, pval, d, n = _stats_from_mb_means(mb_means)

    return dict(key=universe['key'], is_default=universe['is_default'],
                t=tval, p=pval, d=d, n=n)


def compute_universe_timeseries(universe):
    """Context-dependent co-update across learning segments for one universe, as in
    c_contextdep_coupdate_Figure4.py's Panel B. coupdate_diff = same_context minus diff_context corr
    per subject x chronological mini-block index ('mini_id', 1..22, spanning all 11 active
    blocks x 2 mini-blocks each), then a centered moving-average smooth per subject before
    averaging across subjects (mean +/- SEM). Reuses the same z-score/balance choices as
    compute_universe_stats for consistency with the other two panels.

    Returns dict(key, all_mini, mean_smooth, sem_smooth, sig_mask) where sig_mask flags
    timepoints with an uncorrected 1-sample t-test p <= TS_ALPHA_SIG on the smoothed data."""
    coupdate_var = universe['coupdate_var']
    raw_df = load_pair_df(universe['dir_name'], universe.get('analysis', 'coupdate_LSS'))

    included_subjects = sorted(
        s for s in raw_df['subject'].unique()
        if s in _full_and_partial
    )
    pair_df = raw_df[(raw_df['ROI'] == ROI) & (raw_df['subject'].isin(included_subjects))].copy()

    if universe['zscore']:
        stats = (pair_df.groupby(['subject', 'ROI'])[coupdate_var]
                 .agg(['mean', 'std']).reset_index())
        pair_df = pair_df.merge(stats, on=['subject', 'ROI'])
        pair_df[coupdate_var] = (pair_df[coupdate_var] - pair_df['mean']) / pair_df['std']
        pair_df = pair_df.drop(columns=['mean', 'std'])

    if universe['balance']:
        mb_means = compute_balanced_miniblock_means(pair_df, coupdate_var)
    else:
        mb_means = (pair_df
                    .groupby(['subject', 'ROI', 'runType', 'imini_block', 'same_context'])[[coupdate_var]]
                    .mean().reset_index())

    mb_means['runType_num'] = mb_means['runType'].str.extract(r"(\d+)").astype(int)
    mb_means['mini_id'] = (mb_means['imini_block'] + 1) + 2 * (mb_means['runType_num'] - 1)

    pivot_long = (mb_means
                  .pivot_table(index=['subject', 'mini_id'], columns='same_context', values=coupdate_var)
                  .reset_index())
    for col in [0, 1]:
        if col not in pivot_long.columns:
            pivot_long[col] = np.nan
    pivot_long['coupdate_diff'] = pivot_long[1] - pivot_long[0]

    all_mini = np.arange(1, int(mb_means['mini_id'].max()) + 1)
    pv = pivot_long.pivot_table(index='subject', columns='mini_id', values='coupdate_diff')
    pv = pv.reindex(columns=all_mini)

    subj_smoothed = np.vstack([
        smooth_running_nan(pv.loc[subj].values.astype(float), SMOOTH_WIN)
        for subj in pv.index
    ]) if len(pv.index) else np.zeros((0, len(all_mini)))

    mean_smooth = np.nanmean(subj_smoothed, axis=0)
    sem_smooth = np.nanstd(subj_smoothed, axis=0) / np.sqrt(np.sum(~np.isnan(subj_smoothed), axis=0))

    sig_mask = np.zeros(len(all_mini), dtype=bool)
    for t_idx in range(len(all_mini)):
        col_vals = subj_smoothed[:, t_idx]
        col_vals = col_vals[~np.isnan(col_vals)]
        if col_vals.size > 1:
            _, pval = ttest_1samp(col_vals, 0, nan_policy='omit')
            sig_mask[t_idx] = pval <= TS_ALPHA_SIG

    return dict(key=universe['key'], is_default=universe['is_default'],
                all_mini=all_mini, mean_smooth=mean_smooth, sem_smooth=sem_smooth,
                sig_mask=sig_mask)


# %% === Run all universes ===

results = []
for u in UNIVERSES:
    try:
        results.append(compute_universe_stats(u))
    except FileNotFoundError as e:
        print(f"[{ROI}] {UNIVERSE_TITLES[u['key']].replace(chr(10), ' ')}: no data yet, skipping ({e.filename})")

for r in results:
    print(f"[{ROI}] {UNIVERSE_TITLES[r['key']].replace(chr(10), ' ')}: "
          f"t({r['n']-1})={r['t']:.3f}, p={r['p']:.4f}, d={r['d']:.3f}, N={r['n']}")

# Panel B: behavioral link to generalization, same universes. A universe can fail this
# even if Panel A succeeded (e.g. too few mini-blocks with usable acc_change in one of the
# two groups after dropna), so the two panels are handled separately.
results2 = []
for u in UNIVERSES:
    try:
        results2.append(compute_universe_generalization_stats(u))
    except FileNotFoundError as e:
        print(f"[{ROI}] {UNIVERSE_TITLES[u['key']].replace(chr(10), ' ')}: no data yet, skipping ({e.filename})")
    except (KeyError, ValueError) as e:
        print(f"[{ROI}] {UNIVERSE_TITLES[u['key']].replace(chr(10), ' ')}: generalization panel failed, skipping ({e})")

for r in results2:
    print(f"[{ROI}] generalization link {UNIVERSE_TITLES[r['key']].replace(chr(10), ' ')}: "
          f"t({r['n']-1})={r['t']:.3f}, p={r['p']:.4f}, d={r['d']:.3f}, N={r['n']}")

results2_by_key = {r['key']: r for r in results2}

# Timeseries column: same universes, independent failure handling.
if doplotTimeseries:
    results3 = []
    for u in UNIVERSES:
        try:
            results3.append(compute_universe_timeseries(u))
        except FileNotFoundError as e:
            print(f"[{ROI}] {UNIVERSE_TITLES[u['key']].replace(chr(10), ' ')}: no data yet, skipping ({e.filename})")

    results3_by_key = {r['key']: r for r in results3}

# %% === Significance threshold reference lines ===
# Horizontal lines at the Cohen's d corresponding to two-sided p=.05/.01/.001 for a
# paired t-test: t = d * sqrt(N), so d_crit = t_crit(df=N-1) / sqrt(N). Uses the most
# common N across universes and warns if N differs between universes.
def _sig_thresholds(results_list):
    n_counts = Counter(r['n'] for r in results_list)
    N_ref, _ = n_counts.most_common(1)[0]
    if len(n_counts) > 1:
        print(f"WARNING: N differs across universes ({dict(n_counts)}) -- "
              f"significance threshold lines use the most common N={N_ref}")
    return {p: t_dist.ppf(1 - p / 2, df=N_ref - 1) / np.sqrt(N_ref) for p in [0.05, 0.01, 0.001]}

d_crits = _sig_thresholds(results)
d_crits2 = _sig_thresholds(results2)

# %% === Plot ===
# Panel A order (sorted by its own d, descending) is the shared x-axis for both panels, so
# a given universe sits at the same x-position in both. Panel B is looked up by key into
# that same order, and simply left blank at any x where it has no valid result.

results = sorted(results, key=lambda r: r['d'], reverse=True)
keys_order = [r['key'] for r in results]
titles_order = [UNIVERSE_TITLES[k] for k in keys_order]
X_SPACING = 1.6  # multiplier on integer universe index -> x-position; >1 widens the gap
# between points, giving the (horizontal, multi-line) x-tick labels more room not to collide.
x = np.arange(len(results)) * X_SPACING

# Colorbar max: 0 is always the floor; the ceiling is the smallest multiple of 0.01 (at
# least 0.05) that still fits the largest p-value actually plotted across both panels A/B,
# so a single non-significant outlier doesn't wash out the whole scale.
_panel_b_plotted = [results2_by_key[k] for k in keys_order if k in results2_by_key]
_p_max_plotted = max([r['p'] for r in results] + [r['p'] for r in _panel_b_plotted])
p_vmax = max(0.05, np.ceil(_p_max_plotted / 0.01 - 1e-9) * 0.01)
p_vmax = round(p_vmax, 2)

norm = Normalize(vmin=0, vmax=p_vmax)
cmap = plt.get_cmap('viridis_r')

fig_width = 7.2 + (3.5 if doplotTimeseries else 0)  # hard-capped; X_SPACING now controls
# label room within this fixed width instead of growing the figure to fit
fig = plt.figure(figsize=(fig_width, 6))
if doplotTimeseries:
    outer_gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.35)
    left_gs = outer_gs[0].subgridspec(2, 1, hspace=1.8)  # extra room: horizontal x-tick labels on ax1 need more clearance from ax2's title
else:
    left_gs = fig.add_gridspec(2, 1, hspace=1.8)
ax1 = fig.add_subplot(left_gs[0])
ax2 = fig.add_subplot(left_gs[1], sharex=ax1)


def _plot_panel(ax, panel_results, d_crits_panel, ylabel, title, y_lim):
    p_vals = np.array([r['p'] for r in panel_results])
    colors = cmap(norm(np.clip(p_vals, 0, p_vmax)))

    ax.axhline(0, color='k', linestyle='--', lw=0.8, zorder=0)
    for p_thresh, d_crit in d_crits_panel.items():
        ax.axhline(d_crit, color='gray', linestyle=':', lw=0.7, zorder=1)
        ax.text(x[-1] + 0.5, d_crit, f'p={p_thresh}', fontsize=6, color='gray',
                va='bottom', ha='left')
    ax.set_xlim(-0.7, x[-1] + 1.1)
    ax.set_ylim(y_lim)

    xi_list = [x[keys_order.index(r['key'])] for r in panel_results]
    for xi, r, c in zip(xi_list, panel_results, colors):
        ax.scatter(xi, r['d'], s=140 if r['is_default'] else 100,
                   color=c, edgecolor='k', linewidth=1.4 if r['is_default'] else 0.6,
                   zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(titles_order, rotation=0, ha='center', fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


_plot_panel(ax1, results, d_crits,
            "Cohen's d ",
            "Mean context-dependent co-update effect (as in Fig. 4b)", [0, 0.6])
_plot_panel(ax2, [results2_by_key[k] for k in keys_order if k in results2_by_key],
            d_crits2,
            "Cohen's d",
            "Context-dependent co-update in generalization vs. no generalization\nsegments (as in Fig. 6a)", [0, 0.6])

sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=[ax1, ax2], orientation='horizontal', location='bottom',
                     pad=0.3, aspect=40)  # pad=0.3 clears ax2's own rotated x-tick labels below it
cbar.set_label('p-value', fontsize=8)
cbar.ax.tick_params(labelsize=7)

# --- Right column: one small timeseries subplot per universe, same top-to-bottom order as
# the shared x-axis of panels A/B (keys_order), so a universe's scatter position and its
# own learning-segment dynamics can be read side by side. ---
if doplotTimeseries:
    right_gs = outer_gs[1].subgridspec(len(keys_order), 1, hspace=0.35)
    ts_axes = [fig.add_subplot(right_gs[i]) for i in range(len(keys_order))]

    _ts_present = [results3_by_key[k] for k in keys_order if k in results3_by_key]
    if _ts_present:
        _ymax = max(np.nanmax(r['mean_smooth'] + r['sem_smooth']) for r in _ts_present)
        _ymin = min(np.nanmin(r['mean_smooth'] - r['sem_smooth']) for r in _ts_present)
        _yr = _ymax - _ymin if _ymax != _ymin else 1e-6
        ts_ylim = (_ymin - 0.05 * _yr, _ymax + 0.05 * _yr)
    else:
        ts_ylim = None

    for i, (key, ts_ax) in enumerate(zip(keys_order, ts_axes)):
        ts_ax.set_title(UNIVERSE_TITLES[key].replace('\n', ' '), fontsize=6, fontweight='bold', pad=2)
        ts_ax.axhline(0, color='k', linestyle='--', lw=0.6, zorder=0)
        if key not in results3_by_key:
            ts_ax.text(0.5, 0.5, 'no data', ha='center', va='center', fontsize=6,
                        color='gray', transform=ts_ax.transAxes)
        else:
            r = results3_by_key[key]
            for t_idx, mini in enumerate(r['all_mini']):
                if r['sig_mask'][t_idx]:
                    ts_ax.axvspan(mini - 0.5, mini + 0.5, color='yellow', alpha=0.3, zorder=0)
            ts_ax.plot(r['all_mini'], r['mean_smooth'], lw=1.0, color='darkred', zorder=3)
            ts_ax.fill_between(r['all_mini'], r['mean_smooth'] - r['sem_smooth'],
                                r['mean_smooth'] + r['sem_smooth'], color='darkred', alpha=0.25, zorder=2)
            valid_cols = ~np.isnan(r['mean_smooth'])
            if valid_cols.any():
                ts_ax.set_xlim(r['all_mini'][valid_cols][0], r['all_mini'][valid_cols][-1])
            if ts_ylim is not None:
                ts_ax.set_ylim(ts_ylim)

        ts_ax.spines['top'].set_visible(False)
        ts_ax.spines['right'].set_visible(False)
        ts_ax.tick_params(axis='both', labelsize=6)
        if i < len(keys_order) - 1:
            ts_ax.tick_params(axis='x', labelbottom=False)

    ts_axes[-1].set_xlabel("Learning segment #", fontsize=8)
    ts_axes[len(ts_axes) // 2].set_ylabel("Context-dependent co-update\n[same − diff context]", fontsize=7)

os.makedirs(SAVE_DIR, exist_ok=True)
out_path = os.path.join(SAVE_DIR, 'FigureS_OFC_coupdate_multiverse.pdf')
fig.savefig(out_path, bbox_inches='tight')
print(f"\nSaved: {out_path}")
