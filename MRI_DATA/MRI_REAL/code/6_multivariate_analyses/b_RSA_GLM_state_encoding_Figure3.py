#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Figure 3, Fig. S1 and Table S1: emergence of latent state encoding (RSA-GLM).
#
# Reads the RSA-GLM results of the learning and estimation trials, tests them with
# linear mixed-effects models (early vs late learning) and plots them per ROI.
# Input:  outputs of b_RSA_GLM_task_active_trials_all_block_pairs.py and
#         b_RSA_GLM_task_passive_trials_all_block_pairs.py
# Output: Figure3_vertical_room_for_ROIs_<model>.pdf, the supplementary figure and Table S1 csv files
# Note: timeseries_model_type = 'LME' is the version in the paper (the 'decoding' option is not used).
#
# Author: Shany Grossman

# This script plots Figure 3: RSA-GLM analysis traking the emergence of state encoding across diferrent ROIs

#%% Modules
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import math
import statsmodels.formula.api as smf
import seaborn as sns
from scipy.stats import sem
from scipy.stats import sem, ttest_rel, ttest_1samp
import matplotlib.ticker as mticker
from itertools import combinations
from statsmodels.stats.multitest import fdrcorrection
import statsmodels.formula.api as smf
from scipy.stats import norm
import json
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

DO_SUPPRESS_WARNINGS = True
if DO_SUPPRESS_WARNINGS:
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    warnings.filterwarnings('ignore', category=UserWarning,
                            message='.*Random effects covariance is singular.*')

#%% Set paths and load data files 
base_dir = '/Users/grossman/Volumes/STATE_FORMATION/MRI_DATA/MRI_REAL/ROI_timeseries_LSS_5mm_smoothing_dropout_30_physio_reg_out_updated/RSA_GLM_task/' 
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

_ldict_path = SUMMARY_DIR + 'learners_dict.json'
with open(_ldict_path) as _f:
    _learners_dict = json.load(_f)
included_subjects = sorted(_learners_dict['full_and_partial'])
print(f"Including {len(included_subjects)} subjects (full_and_partial)")

#%% Value lookup: canonical reward per passive house state (from active trial logs)
house_values = {'a1': 25, 'a2': 62.5, 'a3': 37.5, 'a4': 75,
                'b1': 100, 'b2': 25, 'b3': 75, 'b4': 0}

def add_active_value_columns(df):
    """Add abs_val_diff_from, abs_val_diff_to, abs_val_diff to an active-trials RSA df.
    Each collapsed state (e.g. 'a1a2') consists of two house IDs (state[:2], state[2:]).
    abs_val_diff_from = |val(h1_from) - val(h2_from)|  (internal value diff of state_from)
    abs_val_diff_to   = |val(h1_to)   - val(h2_to)|    (internal value diff of state_to)
    abs_val_diff      = |abs_val_diff_from - abs_val_diff_to|
    """
    def _state_val_diff(state):
        h1, h2 = state[:2], state[2:]
        return abs(house_values[h1] - house_values[h2])
    df = df.copy()
    df['abs_val_diff_from'] = df['state_from'].apply(_state_val_diff)
    df['abs_val_diff_to']   = df['state_to'].apply(_state_val_diff)
    df['abs_val_diff']      = (df['abs_val_diff_from'] - df['abs_val_diff_to']).abs() /  max(house_values.values())
    df['same_val']          = (df['abs_val_diff'] == 0).astype(int)
    df['val_sim']           = 1 - df['abs_val_diff']
    df['vis_sim']           = (
        (df['same_house_pair'] == 1) &
        (df['same_irrel_left_house_feature'] == 1) &
        (df['same_irrel_right_house_feature'] == 1)
    ).astype(int)
    return df

#%% Define early/late blocks, and ROIs
early_blocks = ['active1', 'active2', 'active3', 'active4','active5','active6']
late_blocks  = ['active6','active7','active8', 'active9', 'active10', 'active11'] #

# Canonical scanning order, used to determine run adjacency, independent of early/late_blocks order
_scan_order_idx = {b: i for i, b in enumerate(
    ['passive1', 'active1', 'active2', 'active3', 'active4', 'active5', 'active6',
     'active7', 'active8', 'active9', 'active10', 'active11', 'passive2'])}


# If True, keep only correlations between consecutive runs within each period
subsequent_runs_only = False
ROIs = [ 'OFC', 'HPC', 'FFA', 'VISUALFUNC_HIGHORDER'] 
timeseries_ROI = 'OFC'

# What time series to  plot? decoding/ interaction beta per block
timeseries_model_type = 'LME' # aleternatives: decoding/LME/OLS

# If False: skip the value-similarity time-series row in Figure 3 and skip Figure S2
plotValueTimeSeries = False

#%% Set pltting params 

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

#%% Figure3A analysis: RSA-GLM for learning trials

model_name = 'state_relevance' #'state_value'
# 'state_value'
# 'context_house' 
# 'state
#' context_house' 
# 'pseudo-state'
# 'value' 
# 'value_wo_same_state' 
# 'state_within_same_val' 
# 'state_value_binary'
# 'state_value_within_context'
# 'state_relevance'
# 'state_relevance_binary'

def _fit_lme(model):
    """Try lbfgs first; fall back to Nelder-Mead if it raises an exception."""
    try:
        return model.fit(method='lbfgs', maxiter=2000)
    except Exception:
        return model.fit(method='nm', maxiter=5000)

def _print_lme(result, label):
    _stars = lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    skip = {'Intercept', 'Group Var'}
    print(f"\n=== {label} ===")
    print(f"  {'term':<35} {'beta':>8} {'SE':>8} {'z':>7} {'p':>8}")
    for term in result.params.index:
        if term in skip:
            continue
        b, se, z, p = (result.params[term], result.bse[term],
                       result.tvalues[term], result.pvalues[term])
        print(f"  {term:<35} {b:>8.4f} {se:>8.4f} {z:>7.2f} {p:>8.4f} {_stars(p)}")

results = []
individual_betas = []
lme_results_model_per_period = []

for roi in ROIs:
    # Here we can adjust what block are in early and what are in late
    corr_df_learning = pd.read_csv(base_dir + roi + '_corr_results_active_trials_all_pairs_collapsed_sides_zscored.csv')
    corr_df_learning = corr_df_learning[corr_df_learning['subject'].isin(included_subjects)].reset_index(drop=True)
    corr_df_learning = corr_df_learning.dropna(subset=['correlation']).reset_index(drop=True)
    corr_df_learning = add_active_value_columns(corr_df_learning)
    _learn_missing = sorted(set(included_subjects) - set(corr_df_learning['subject'].unique()))
    print(f"[{roi}] Learning: {corr_df_learning['subject'].nunique()} subjects included" +
          (f" — missing: {_learn_missing}" if _learn_missing else ""))
    corr_df_filt=corr_df_learning.copy()
    corr_df_filt=corr_df_filt[corr_df_filt['ROI']==roi].copy()

    # Keep only rows where for 'early' period, either block_from or block_to is in early_blocks
    # and same for 'late' period with late_blocks
    corr_df_filt = corr_df_filt[
        ((corr_df_filt['period'] == 'early') &
         ((corr_df_filt['block_from'].isin(early_blocks)) &
          (corr_df_filt['block_to'].isin(early_blocks))))
            |
        ((corr_df_filt['period'] == 'late') &
         ((corr_df_filt['block_from'].isin(late_blocks)) &
          (corr_df_filt['block_to'].isin(late_blocks))))]

    if subsequent_runs_only:
        _fi = corr_df_filt['block_from'].map(_scan_order_idx)
        _ti = corr_df_filt['block_to'].map(_scan_order_idx)
        corr_df_filt = corr_df_filt[(_fi - _ti).abs() == 1]

    # Run Linear Mixed model with period (early,late) included
    corr_df_filt['period'] = corr_df_filt['period'].map({'early': -1, 'late': 1})

    df_sub = corr_df_filt[corr_df_filt['ROI'] == roi].copy()

    # Apply row filter BEFORE try block so OLS always sees filtered data
    if model_name == 'value_wo_same_state':
        df_sub = df_sub[df_sub['same_state'] == 0].copy()
    elif model_name == 'state_value_within_context':
        df_sub = df_sub[df_sub['same_context'] == 1].copy()
    elif model_name == 'state_within_same_val':
        df_sub = df_sub[df_sub['same_val'] == 1].copy()
        print(f"\n=== {roi} ===")
        print(f"Total rows after filter: {len(df_sub)}")
        print(f"same_val counts: {df_sub['same_val'].value_counts().to_dict()}")
        print(f"same_state counts: {df_sub['same_state'].value_counts().to_dict()}")
        for p, pname in [(-1, 'early'), (1, 'late')]:
            pdata = df_sub[df_sub['period'] == p]
            for ss in [0, 1]:
                vals = pdata[pdata['same_state'] == ss]['correlation']
                print(f"  {pname}, same_state={ss}: n={len(vals)}, mean={vals.mean():.4f}")
    # value: all rows included (same_state rows kept)
    # state_value_binary, state_value: no row filter

    if model_name in ('state_value', 'state_value_vis') and roi == ROIs[0]:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        _cols = ['same_state', 'val_sim', 'vis_sim'] if model_name == 'state_value_vis' else ['same_state', 'val_sim']
        _X = df_sub[_cols].dropna()
        print(f"\n=== Collinearity check ({model_name}, {roi}) ===")
        for _i, _c1 in enumerate(_cols):
            for _c2 in _cols[_i+1:]:
                print(f"  Pearson r({_c1}, {_c2}) = {_X[_c1].corr(_X[_c2]):.4f}")
        for _i, _c in enumerate(_cols):
            print(f"  VIF {_c} = {variance_inflation_factor(_X.values, _i):.2f}")
        if 'val_sim' in _cols:
            print(f"  val_sim | same_state=1: mean={_X[_X['same_state']==1]['val_sim'].mean():.3f}  same_state=0: mean={_X[_X['same_state']==0]['val_sim'].mean():.3f}")

    try:
        if model_name == 'state':
            formula = (
                "correlation ~ period*same_state + "
                "same_irrel_client_feature + "
                "same_irrel_left_house_feature + "
                "same_irrel_right_house_feature  ")
                #"same_state_and_side")

            regressors = ['same_state', 'same_irrel_client_feature',
                          'same_irrel_left_house_feature', 'same_irrel_right_house_feature'] #, 'same_state_and_side']

        elif model_name in ('value', 'value_wo_same_state'):
            formula = (
                "correlation ~ period*val_sim + "
                "same_irrel_client_feature + "
                "same_irrel_left_house_feature + "
                "same_irrel_right_house_feature")
            regressors = ['val_sim', 'same_irrel_client_feature',
                          'same_irrel_left_house_feature', 'same_irrel_right_house_feature']

        elif model_name in ('state_value', 'state_value_within_context'):
            formula = (
                "correlation ~  period*same_state + val_sim + "
                "same_irrel_client_feature + "
                "same_irrel_left_house_feature + "
                "same_irrel_right_house_feature")
            regressors = ['same_state', 'val_sim', 'same_irrel_client_feature',
                          'same_irrel_left_house_feature', 'same_irrel_right_house_feature']

        elif model_name == 'state_relevance':
            df_sub['irrelevant_sim'] = (
                df_sub['same_irrel_client_feature'] +
                df_sub['same_irrel_left_house_feature'] +
                df_sub['same_irrel_right_house_feature']) / 3
            # period* interaction only on same_state (as for state_value). val_sim and
            # irrelevant_sim are covariates without a period* term, because they share
            # variance with same_state (same_state/val_sim r ~ 0.5).
            formula = "correlation ~ period*same_state + val_sim + irrelevant_sim"
            regressors = ['same_state', 'irrelevant_sim', 'val_sim']

        elif model_name == 'state_relevance_binary':
            # Same as state_relevance, but irrelevant_sim is a strict AND across
            # features (1 only if every irrelevant feature matches) instead of the
            # averaged fraction (binary, on the same 0/1 scale as same_state).
            df_sub['irrelevant_sim'] = (
                (df_sub['same_irrel_client_feature'] == 1) &
                (df_sub['same_irrel_left_house_feature'] == 1) &
                (df_sub['same_irrel_right_house_feature'] == 1)
            ).astype(int)
            formula = "correlation ~ period*same_state + val_sim + irrelevant_sim"
            regressors = ['same_state', 'irrelevant_sim', 'val_sim']

        elif model_name == 'state_value_vis':
            formula = (
                "correlation ~ period*same_state + period*val_sim + period*vis_sim + "
                "same_irrel_client_feature")
            regressors = ['same_state', 'val_sim', 'vis_sim', 'same_irrel_client_feature']

        elif model_name == 'state_within_same_val':
            formula = ("correlation ~ period*same_state + "
                       "same_irrel_client_feature")
            regressors = ['same_state', 'same_irrel_client_feature']

        elif model_name == 'state_value_binary':
            formula = (
                "correlation ~ period*same_state + period*same_val + "
                "same_irrel_client_feature + "
                "same_irrel_left_house_feature + "
                "same_irrel_right_house_feature")
            regressors = ['same_state', 'same_val', 'same_irrel_client_feature',
                          'same_irrel_left_house_feature', 'same_irrel_right_house_feature']

        elif model_name == 'pseudo-state':
            formula = "correlation ~ period*same_pseudo_state + period*same_state"
            regressors = ['same_pseudo_state', 'same_state']

        elif model_name=='context_house':
            formula = (
                "correlation ~ period * same_context + period * same_house_pair + "
                "same_irrel_client_feature + "
                "same_irrel_left_house_feature + "
                "same_irrel_right_house_feature" )

            regressors = ['same_context', 'same_house_pair', 'same_irrel_client_feature',
                          'same_irrel_left_house_feature', 'same_irrel_right_house_feature']

        # Random slope mirrors the key fixed-effect regressors in each model
        if model_name == 'state_value_binary':
            re_f = "~same_state + same_val + period"
        elif model_name == 'state_value_vis':
            re_f = "~ same_state + val_sim + vis_sim + period"
        elif model_name in ('state_value', 'state_value_within_context'):
            re_f = "~ same_state + val_sim + period"
        elif model_name in ('value', 'value_wo_same_state'):
            re_f = "~val_sim + period"
        elif model_name == 'context_house':
            re_f = "~same_context + same_house_pair + period"
        elif model_name == 'pseudo-state':
            re_f = None  # random intercept only: variance near boundary with random slopes
        else:
            re_f = "~same_state + period"
        model = smf.mixedlm(
                formula,
                df_sub,
                groups=df_sub["subject"])
        result = _fit_lme(model)
        _print_lme(result, f"{roi} Learning LME — all periods")

        # Collect main effects & interaction terms
        for reg in regressors:
            results.append({
                'ROI': roi,
                'regressor': reg,
                'effect': 'main',
                'beta': result.params.get(reg, np.nan),
                'pval': result.pvalues.get(reg, np.nan),
                'se': result.bse.get(reg, np.nan)})
            
            results.append({
                'ROI': roi,
                'regressor': reg,
                'effect': 'period:interaction',
                'beta': result.params.get(f'period:{reg}', np.nan),
                'pval': result.pvalues.get(f'period:{reg}', np.nan),
                'se': result.bse.get(f'period:{reg}', np.nan)})

    except Exception as e:
        print(f"Model failed for ROI {roi}: {e}")
        continue


    # --- Per-subject and period OLS betas (for plotting)
    for subj in df_sub['subject'].unique():
        subj_df = df_sub[df_sub['subject'] == subj]
        for period_name, blocks in [(-1, early_blocks), (1, late_blocks)]:
            subj_period_df = subj_df[subj_df['period']== period_name]
            if len(subj_period_df) < 5:  # skip very small subsets
                continue
            try:
                if model_name == 'state':
                    ols_model = smf.ols(
                        "correlation ~ 1 + same_state + same_irrel_client_feature + "
                        "same_irrel_left_house_feature + same_irrel_right_house_feature",
                        data=subj_period_df
                    ).fit()
                    for reg in regressors:
                        individual_betas.append({
                            'ROI': roi,
                            'period': period_name,
                            'regressor': reg,
                            'subject': subj,
                            'beta': ols_model.params.get(reg, np.nan)})

                elif model_name in ('value', 'value_wo_same_state'):
                    ols_model = smf.ols(
                        "correlation ~ val_sim + same_irrel_client_feature + "
                        "same_irrel_left_house_feature + same_irrel_right_house_feature",
                        data=subj_period_df).fit()
                    for reg in regressors:
                        individual_betas.append({
                            'ROI': roi,
                            'period': period_name,
                            'regressor': reg,
                            'subject': subj,
                            'beta': ols_model.params.get(reg, np.nan)})

                elif model_name == 'state_within_same_val':
                    ols_model = smf.ols(
                        "correlation ~ same_state + same_irrel_client_feature",
                        data=subj_period_df).fit()
                    for reg in regressors:
                        individual_betas.append({
                            'ROI': roi,
                            'period': period_name,
                            'regressor': reg,
                            'subject': subj,
                            'beta': ols_model.params.get(reg, np.nan)})

                elif model_name in ('state_relevance', 'state_relevance_binary'):
                    ols_model = smf.ols(
                        "correlation ~ same_state + irrelevant_sim + val_sim",
                        data=subj_period_df).fit()
                    for reg in regressors:
                        individual_betas.append({
                            'ROI': roi,
                            'period': period_name,
                            'regressor': reg,
                            'subject': subj,
                            'beta': ols_model.params.get(reg, np.nan)})

                elif model_name in ('state_value', 'state_value_within_context'):
                    ols_model = smf.ols(
                        "correlation ~ same_state + val_sim + same_irrel_client_feature + "
                        "same_irrel_left_house_feature + same_irrel_right_house_feature",
                        data=subj_period_df).fit()
                    for reg in regressors:
                        individual_betas.append({
                            'ROI': roi,
                            'period': period_name,
                            'regressor': reg,
                            'subject': subj,
                            'beta': ols_model.params.get(reg, np.nan)})

                elif model_name == 'state_value_vis':
                    ols_model = smf.ols(
                        "correlation ~ same_state + val_sim + vis_sim + same_irrel_client_feature",
                        data=subj_period_df).fit()
                    for reg in regressors:
                        individual_betas.append({
                            'ROI': roi,
                            'period': period_name,
                            'regressor': reg,
                            'subject': subj,
                            'beta': ols_model.params.get(reg, np.nan)})

                elif model_name == 'state_value_binary':
                    ols_model = smf.ols(
                        "correlation ~ same_state + same_val + same_irrel_client_feature + "
                        "same_irrel_left_house_feature + same_irrel_right_house_feature",
                        data=subj_period_df
                    ).fit()
                    for reg in regressors:
                        individual_betas.append({
                            'ROI': roi,
                            'period': period_name,
                            'regressor': reg,
                            'subject': subj,
                            'beta': ols_model.params.get(reg, np.nan)})

                elif model_name == 'pseudo-state':
                    ols_model = smf.ols(
                        "correlation ~ same_pseudo_state + same_state",
                        data=subj_period_df
                    ).fit()
                    for reg in regressors:
                        individual_betas.append({
                            'ROI': roi,
                            'period': period_name,
                            'regressor': reg,
                            'subject': subj,
                            'beta': ols_model.params.get(reg, np.nan)})

                elif model_name=='context_house':
                    
                    ols_model = smf.ols(
                        "correlation ~ same_context + same_house_pair + "
                        "same_irrel_client_feature + "
                        "same_irrel_left_house_feature + same_irrel_right_house_feature",
                        data=subj_period_df
                    ).fit()
                    for reg in regressors:
                        individual_betas.append({
                            'ROI': roi,
                            'period': period_name,
                            'regressor': reg,
                            'subject': subj,
                            'beta': ols_model.params.get(reg, np.nan)})
            except:
                continue
            
    # --- Per-time period LME model (for reporting statistical significance of single periods)
    for period_name, blocks in [(-1, early_blocks), (1, late_blocks)]:
        period_df = df_sub[df_sub['period']== period_name]

        try:
            if model_name == 'state':
                formula = (
                    "correlation ~ same_state + "
                    "same_irrel_client_feature + "
                    "same_irrel_left_house_feature + "
                    "same_irrel_right_house_feature  ")
                regressors = ['same_state', 'same_irrel_client_feature',
                              'same_irrel_left_house_feature', 'same_irrel_right_house_feature']

            elif model_name in ('value', 'value_wo_same_state'):
                formula = (
                    "correlation ~ val_sim + "
                    "same_irrel_client_feature + "
                    "same_irrel_left_house_feature + "
                    "same_irrel_right_house_feature")
                regressors = ['val_sim', 'same_irrel_client_feature',
                              'same_irrel_left_house_feature', 'same_irrel_right_house_feature']

            elif model_name == 'state_within_same_val':
                formula = (
                    "correlation ~ same_state + "
                    "same_irrel_client_feature + "
                    "same_irrel_left_house_feature + "
                    "same_irrel_right_house_feature")
                regressors = ['same_state', 'same_irrel_client_feature',
                              'same_irrel_left_house_feature', 'same_irrel_right_house_feature']

            elif model_name in ('state_relevance', 'state_relevance_binary'):
                formula = "correlation ~ same_state + irrelevant_sim + val_sim"
                regressors = ['same_state', 'irrelevant_sim', 'val_sim']

            elif model_name in ('state_value', 'state_value_within_context'):
                formula = (
                    "correlation ~ same_state + val_sim + "
                    "same_irrel_client_feature + "
                    "same_irrel_left_house_feature + "
                    "same_irrel_right_house_feature")
                regressors = ['val_sim', 'same_state', 'same_irrel_client_feature',
                              'same_irrel_left_house_feature', 'same_irrel_right_house_feature']

            elif model_name == 'state_value_vis':
                formula = (
                    "correlation ~ same_state + val_sim + vis_sim + "
                    "same_irrel_client_feature")
                regressors = ['same_state', 'val_sim', 'vis_sim', 'same_irrel_client_feature']

            elif model_name == 'state_value_binary':
                formula = (
                    "correlation ~ same_state + same_val + "
                    "same_irrel_client_feature + "
                    "same_irrel_left_house_feature + "
                    "same_irrel_right_house_feature")
                regressors = ['same_state', 'same_val', 'same_irrel_client_feature',
                              'same_irrel_left_house_feature', 'same_irrel_right_house_feature']

            elif model_name == 'pseudo-state':
                formula = "correlation ~ same_pseudo_state + same_state"
                regressors = ['same_pseudo_state', 'same_state']

            elif model_name == 'context_house':
                formula = (
                    "correlation ~ same_context + same_house_pair + "
                    "same_irrel_client_feature + "
                    "same_irrel_left_house_feature + "
                    "same_irrel_right_house_feature")
                regressors = ['same_context', 'same_house_pair', 'same_irrel_client_feature',
                              'same_irrel_left_house_feature', 'same_irrel_right_house_feature']

            model = smf.mixedlm(
                    formula,
                    period_df,
                    groups=period_df["subject"])
            result = _fit_lme(model)
            _period_label = 'Early' if period_name == -1 else 'Late'

            # Collect main effects & interaction terms
            for reg in regressors:
                lme_results_model_per_period.append({
                    'ROI': roi,
                    'regressor': reg,
                    'period': period_name,
                    'beta': result.params.get(reg, np.nan),
                    'pval': result.pvalues.get(reg, np.nan)})

        except:
            continue


results_df_learning = pd.DataFrame(results)
individual_df_learning = pd.DataFrame(individual_betas)
lme_per_period_df_learning = pd.DataFrame(lme_results_model_per_period)

#%% Figure3B analysis: RSA-GLM for estimation trials


results = []
individual_betas = []
lme_results_model_per_period = []

for roi in ROIs:

    corr_df_estimation = pd.read_csv(base_dir + roi + '_corr_results_passive_trials_all_2_all_all_pairs_zscored.csv') 
    corr_df_estimation = corr_df_estimation[corr_df_estimation['subject'].isin(included_subjects)].reset_index(drop=True)
    corr_df_estimation = corr_df_estimation.dropna(subset=['correlation']).reset_index(drop=True)
    _est_missing = sorted(set(included_subjects) - set(corr_df_estimation['subject'].unique()))
    print(f"[{roi}] Estimation: {corr_df_estimation['subject'].nunique()} subjects included" +
          (f" — missing: {_est_missing}" if _est_missing else ""))
    # Recompute abs_val_diff from state names using exact house_values (CSV stores rounded integers).
    # Estimation states are 2-char codes (e.g. 'a1'), so map directly.
    corr_df_estimation['abs_val_diff'] = (
        corr_df_estimation['state_from'].map(house_values) -
        corr_df_estimation['state_to'].map(house_values)
    ).abs()
    corr_df_estimation['same_val'] = (corr_df_estimation['abs_val_diff'] == 0).astype(int)
    corr_df_estimation['val_sim']  = 1 - corr_df_estimation['abs_val_diff'] / max(house_values.values())
    corr_df_estimation['vis_sim']  = (
        (corr_df_estimation['same_house_type'] == 1) &
        (corr_df_estimation['same_irrel_house_feature'] == 1)
    ).astype(int)
    corr_df_filt = corr_df_estimation.copy()

    # Keep only rows where for 'early' period, either block_from or block_to is in early_blocks
    # and same for 'late' period with late_blocks
    corr_df_filt = corr_df_filt[
        ((corr_df_filt['period'] == 'early') &
         ((corr_df_filt['block'].isin(early_blocks)) &
          (corr_df_filt['next_block'].isin(early_blocks))))
            |
        ((corr_df_filt['period'] == 'late') &
         ((corr_df_filt['block'].isin(late_blocks)) &
          (corr_df_filt['next_block'].isin(late_blocks))))]

    if subsequent_runs_only:
        _fi = corr_df_filt['block'].map(_scan_order_idx)
        _ti = corr_df_filt['next_block'].map(_scan_order_idx)
        corr_df_filt = corr_df_filt[(_fi - _ti).abs() == 1]

    corr_df_filt['period'] = corr_df_filt['period'].map({'early': -1, 'late': 1})


    df_sub = corr_df_filt[corr_df_filt['ROI'] == roi].copy()

    # Apply row filter BEFORE try block so OLS always sees filtered data
    if model_name == 'value_wo_same_state':
        df_sub = df_sub[df_sub['same_state'] == 0].copy()
    elif model_name == 'state_value_within_context':
        df_sub = df_sub[df_sub['same_context'] == 1].copy()
    elif model_name == 'state_within_same_val':
        df_sub = df_sub[df_sub['same_val'] == 1].copy()
    # value: all rows included (same_state rows kept)
    # state_value_binary, state_value: no row filter

    try:
        if model_name == 'state':
            formula = (
                "correlation ~ period*same_state + "
                "same_irrel_client_feature + "
                "same_irrel_house_feature")
            regressors = ['same_state', 'same_irrel_client_feature', 'same_irrel_house_feature']

        elif model_name in ('value', 'value_wo_same_state'):
            formula = (
                "correlation ~ period*val_sim + "
                "same_irrel_client_feature + "
                "same_irrel_house_feature")
            regressors = ['val_sim', 'same_irrel_client_feature', 'same_irrel_house_feature']

        elif model_name in ('state_value', 'state_value_within_context'):
            formula = (
                "correlation ~ period*same_state + val_sim + "
                "same_irrel_client_feature + "
                "same_irrel_house_feature")
            regressors = ['same_state', 'val_sim', 'same_irrel_client_feature', 'same_irrel_house_feature']

        elif model_name == 'state_relevance':
            df_sub['irrelevant_sim'] = (
                df_sub['same_irrel_client_feature'] +
                df_sub['same_irrel_house_feature']) / 2
            # period* interaction only on same_state (see the learning-section comment above).
            formula = "correlation ~ period*same_state + val_sim + irrelevant_sim"
            regressors = ['same_state', 'irrelevant_sim', 'val_sim']

        elif model_name == 'state_relevance_binary':
            df_sub['irrelevant_sim'] = (
                (df_sub['same_irrel_client_feature'] == 1) &
                (df_sub['same_irrel_house_feature'] == 1)
            ).astype(int)
            formula = "correlation ~ period*same_state + val_sim + irrelevant_sim"
            regressors = ['same_state', 'irrelevant_sim', 'val_sim']

        elif model_name == 'state_value_vis':
            formula = (
                "correlation ~ period*same_state + period*val_sim + period*vis_sim + "
                "same_irrel_client_feature")
            regressors = ['same_state', 'val_sim', 'vis_sim', 'same_irrel_client_feature']

        elif model_name == 'state_within_same_val':
            # Irrel regressors dropped: within same_val==1, the only cross-state pairs are
            # cross-context (a1↔b2, a4↔b3), making irrel features structurally near-collinear
            # with same_state at the per-subject level.
            formula = "correlation ~ period*same_state"
            regressors = ['same_state']

        elif model_name == 'state_value_binary':
            formula = (
                "correlation ~ period*same_state + period*same_val + "
                "same_irrel_client_feature + "
                "same_irrel_house_feature")
            regressors = ['same_state', 'same_val', 'same_irrel_client_feature', 'same_irrel_house_feature']

        elif model_name == 'pseudo-state':
            formula = "correlation ~ period*same_pseudo_state + period*same_state"
            regressors = ['same_pseudo_state', 'same_state']

        elif model_name=='context_house':
            formula = (
                "correlation ~ period * same_context + period * same_house_type + "
                "same_irrel_client_feature + "
                "same_irrel_house_feature  "
            )
            regressors = ['same_context', 'same_house_type', 'same_irrel_client_feature',
                          'same_irrel_house_feature']

        # --- Mixed model with period included
        model = smf.mixedlm(formula, df_sub, groups=df_sub["subject"])
        result = _fit_lme(model)
        _print_lme(result, f"{roi} Estimation LME — all periods")

        # Collect main effects & interaction terms
        for reg in regressors:
            results.append({
                'ROI': roi,
                'regressor': reg,
                'effect': 'main',
                'beta': result.params.get(reg, np.nan),
                'pval': result.pvalues.get(reg, np.nan),
                'se': result.bse.get(reg, np.nan),
            })
            results.append({
                'ROI': roi,
                'regressor': reg,
                'effect': 'period:interaction',
                'beta': result.params.get(f'period:{reg}', np.nan),
                'pval': result.pvalues.get(f'period:{reg}', np.nan),
                'se': result.bse.get(f'period:{reg}', np.nan),
            })

    except Exception as e:
        print(f"Model failed for ROI {roi}: {e}")
        continue

    # --- Per-subject OLS betas (still separated into early/late for plotting)
    for subj in df_sub['subject'].unique():
        subj_df = df_sub[df_sub['subject'] == subj]
        for period_name, blocks in [(-1, early_blocks), (1, late_blocks)]:
            subj_period_df = subj_df[subj_df['period']== period_name]
            if len(subj_period_df) < 5:  # skip very small subsets
                continue
            try:
                if model_name == 'state':
                    ols_model = smf.ols(
                        "correlation ~ same_state + same_irrel_client_feature + same_irrel_house_feature",
                        data=subj_period_df).fit()
                    for reg in regressors:
                        individual_betas.append({
                            'ROI': roi,
                            'period': period_name,
                            'regressor': reg,
                            'subject': subj,
                            'beta': ols_model.params.get(reg, np.nan)})

                elif model_name in ('value', 'value_wo_same_state'):
                    ols_model = smf.ols(
                        "correlation ~ val_sim + same_irrel_client_feature + same_irrel_house_feature",
                        data=subj_period_df).fit()
                    for reg in regressors:
                        individual_betas.append({
                            'ROI': roi,
                            'period': period_name,
                            'regressor': reg,
                            'subject': subj,
                            'beta': ols_model.params.get(reg, np.nan)})

                elif model_name == 'state_within_same_val':
                    ols_model = smf.ols(
                        "correlation ~ same_state",
                        data=subj_period_df).fit()
                    for reg in regressors:
                        individual_betas.append({
                            'ROI': roi,
                            'period': period_name,
                            'regressor': reg,
                            'subject': subj,
                            'beta': ols_model.params.get(reg, np.nan)})

                elif model_name in ('state_relevance', 'state_relevance_binary'):
                    ols_model = smf.ols(
                        "correlation ~ same_state + irrelevant_sim + val_sim",
                        data=subj_period_df).fit()
                    for reg in regressors:
                        individual_betas.append({
                            'ROI': roi,
                            'period': period_name,
                            'regressor': reg,
                            'subject': subj,
                            'beta': ols_model.params.get(reg, np.nan)})

                elif model_name in ('state_value', 'state_value_within_context'):
                    ols_model = smf.ols(
                        "correlation ~ same_state + val_sim + same_irrel_client_feature + same_irrel_house_feature",
                        data=subj_period_df).fit()
                    for reg in regressors:
                        individual_betas.append({
                            'ROI': roi,
                            'period': period_name,
                            'regressor': reg,
                            'subject': subj,
                            'beta': ols_model.params.get(reg, np.nan)})

                elif model_name == 'state_value_vis':
                    # same_state dropped for estimation: confounded with house+face identity
                    ols_model = smf.ols(
                        "correlation ~ val_sim + vis_sim + same_irrel_client_feature",
                        data=subj_period_df).fit()
                    for reg in regressors:
                        individual_betas.append({
                            'ROI': roi,
                            'period': period_name,
                            'regressor': reg,
                            'subject': subj,
                            'beta': ols_model.params.get(reg, np.nan)})

                elif model_name == 'state_value_binary':
                    ols_model = smf.ols(
                        "correlation ~ same_state + same_val + same_irrel_client_feature + same_irrel_house_feature",
                        data=subj_period_df).fit()
                    for reg in regressors:
                        individual_betas.append({
                            'ROI': roi,
                            'period': period_name,
                            'regressor': reg,
                            'subject': subj,
                            'beta': ols_model.params.get(reg, np.nan)})

                elif model_name == 'pseudo-state':
                    ols_model = smf.ols(
                        "correlation ~ same_pseudo_state + same_state",
                        data=subj_period_df).fit()
                    for reg in regressors:
                        individual_betas.append({
                            'ROI': roi,
                            'period': period_name,
                            'regressor': reg,
                            'subject': subj,
                            'beta': ols_model.params.get(reg, np.nan)})


                elif model_name=='context_house':
                    
                    ols_model = smf.ols(
                        "correlation ~ same_context + same_house_type + "
                        "same_irrel_client_feature + "
                        "same_irrel_house_feature  ",
                        data=subj_period_df
                    ).fit()
                    for reg in regressors:
                        individual_betas.append({
                            'ROI': roi,
                            'period': period_name,
                            'regressor': reg,
                            'subject': subj,
                            'beta': ols_model.params.get(reg, np.nan)})
            except:
                continue
            
    # --- Per-time period LME model (for reporting statistical significance of single periods)
    for period_name, blocks in [(-1, early_blocks), (1, late_blocks)]:
        period_df = df_sub[df_sub['period']== period_name]

        try:
            if model_name == 'state':
                formula = (
                    "correlation ~ same_state + "
                    "same_irrel_client_feature + "
                    "same_irrel_house_feature")
                regressors = ['same_state', 'same_irrel_client_feature',
                              'same_irrel_house_feature']

            elif model_name in ('value', 'value_wo_same_state'):
                formula = (
                    "correlation ~ val_sim + "
                    "same_irrel_client_feature + "
                    "same_irrel_house_feature")
                regressors = ['val_sim', 'same_irrel_client_feature',
                              'same_irrel_house_feature']

            elif model_name == 'state_within_same_val':
                formula = "correlation ~ same_state"
                regressors = ['same_state']

            elif model_name in ('state_relevance', 'state_relevance_binary'):
                formula = "correlation ~ same_state + irrelevant_sim + val_sim"
                regressors = ['same_state', 'irrelevant_sim', 'val_sim']

            elif model_name in ('state_value', 'state_value_within_context'):
                formula = (
                    "correlation ~ same_state + val_sim + "
                    "same_irrel_client_feature + "
                    "same_irrel_house_feature")
                regressors = ['val_sim', 'same_state', 'same_irrel_client_feature',
                              'same_irrel_house_feature']

            elif model_name == 'state_value_vis':
                # same_state dropped for estimation: confounded with house+face identity
                formula = (
                    "correlation ~ val_sim + vis_sim + "
                    "same_irrel_client_feature")
                regressors = ['val_sim', 'vis_sim', 'same_irrel_client_feature']

            elif model_name == 'state_value_binary':
                formula = (
                    "correlation ~ same_state + same_val + "
                    "same_irrel_client_feature + "
                    "same_irrel_house_feature")
                regressors = ['same_state', 'same_val', 'same_irrel_client_feature',
                              'same_irrel_house_feature']

            elif model_name == 'pseudo-state':
                formula = "correlation ~ same_pseudo_state + same_state"
                regressors = ['same_pseudo_state', 'same_state']

            elif model_name == 'context_house':
                formula = (
                    "correlation ~ same_context + same_house_type + "
                    "same_irrel_client_feature + "
                    "same_irrel_house_feature")
                regressors = ['same_context', 'same_house_type', 'same_irrel_client_feature',
                              'same_irrel_house_feature']

            model = smf.mixedlm(
                    formula,
                    period_df,
                    groups=period_df["subject"])
            result = _fit_lme(model)
            _period_label = 'Early' if period_name == -1 else 'Late'

            # Collect main effects & interaction terms
            for reg in regressors:
                lme_results_model_per_period.append({
                    'ROI': roi,
                    'regressor': reg,
                    'period': period_name,
                    'beta': result.params.get(reg, np.nan),
                    'pval': result.pvalues.get(reg, np.nan)})

        except:
            continue

results_df_estimation = pd.DataFrame(results)
individual_df_estimation = pd.DataFrame(individual_betas)
lme_per_period_df_estimation = pd.DataFrame(lme_results_model_per_period)

# state_relevance: each regressor has its own period* term, so period:same_state,
# period:val_sim and period:irrelevant_sim are already in the 'period:interaction'
# rows above and in Table S1 below.


#%% Figure 3 2nd row Running-window RSA-GLM for state, context, house models

#For presenting OLS/LME beta time series 
def running_window_glm(corr_df, ROIs, model_name, win_size, stride,
                       trials_type, timeseries_model_type):

    # Run RSA GLM analyses using either OLS (per subject) or LME (across subjects).

    # For model_type == 'LME':
    #     • Only correlations with the subsequent block are included
    #     • 'block' is modeled as a numeric variable (block_num)
    #     • The model includes interactions of block_num with each regressor

    all_betas = []

    # --- Determine block order ---
    if trials_type == 'learning':
        blocks = sorted(set(corr_df['block_from']).union(set(corr_df['block_to'])))
    elif trials_type == 'estimation':
        blocks = sorted(set(corr_df['block']).union(set(corr_df['next_block'])))
    else:
        raise ValueError("trials_type must be 'learning' or 'estimation'")

    # Sort key: passive1 first (-1), activeN by number, passive2 last (10000)
    def _block_sort_key(b):
        if b == 'passive1':        return -1
        if b.startswith('active'): return int(''.join(filter(str.isdigit, b)))
        if b == 'passive2':        return 10000
        return 99999

    # For estimation LME: include passive1 (reference) and passive2 (after active11)
    # if they are present in the data; otherwise keep only active blocks.
    if trials_type == 'estimation' and timeseries_model_type == 'LME':
        _allowed_passive = {b for b in (early_blocks + late_blocks) if b in ('passive1', 'passive2')}
        _passive_present = {b for b in blocks if b in _allowed_passive}
        blocks = sorted(
            [b for b in blocks if b.startswith('active') or b in _passive_present],
            key=_block_sort_key)
    else:
        blocks = sorted([b for b in blocks if b.startswith('active')],
                        key=lambda x: int(''.join(filter(str.isdigit, x))))
    n_blocks = len(blocks)

    # --- Loop over ROIs ---
    for roi in ROIs:
        df_roi = corr_df[corr_df['ROI'] == roi].copy()

        # --- if OLS/LME --> Define regressors ---
        if timeseries_model_type=='OLS' or timeseries_model_type=='LME':
            if trials_type == 'learning':
                if model_name == 'state':
                    regressors = ['same_state', "same_irrel_client_feature",
                    "same_irrel_left_house_feature" , "same_irrel_right_house_feature"]
                elif model_name == 'context_house':
                    regressors = ['same_context', 'same_house_pair',  "same_irrel_client_feature",
                    "same_irrel_left_house_feature" , "same_irrel_right_house_feature"]
                elif model_name in ('value', 'value_wo_same_state'):
                    regressors = ['val_sim', "same_irrel_client_feature",
                    "same_irrel_left_house_feature", "same_irrel_right_house_feature"]
                elif model_name in ('state_value', 'state_value_within_context'):
                    regressors = ['same_state', 'val_sim', "same_irrel_client_feature",
                    "same_irrel_left_house_feature", "same_irrel_right_house_feature"]
                elif model_name == 'state_value_vis':
                    regressors = ['same_state', 'val_sim', 'vis_sim', "same_irrel_client_feature"]
                else:
                    regressors = ['same_state', "same_irrel_client_feature",
                    "same_irrel_left_house_feature", "same_irrel_right_house_feature"]
            else:  # estimation
                if model_name == 'state':
                    regressors = ['same_state', "same_irrel_client_feature",
                    "same_irrel_house_feature"]
                elif model_name == 'context_house':
                    regressors = ['same_context', 'same_house_type', "same_irrel_client_feature",
                    "same_irrel_house_feature"]
                elif model_name in ('value', 'value_wo_same_state'):
                    regressors = ['val_sim', "same_irrel_client_feature",
                    "same_irrel_house_feature"]
                elif model_name in ('state_value', 'state_value_within_context'):
                    regressors = ['same_state', 'val_sim', "same_irrel_client_feature",
                    "same_irrel_house_feature"]
                elif model_name == 'state_value_vis':
                    regressors = ['val_sim', 'vis_sim', "same_irrel_client_feature"]
                else:
                    regressors = ['same_state', "same_irrel_client_feature",
                    "same_irrel_house_feature"]

        # --- OLS computation ---
        if timeseries_model_type == 'OLS':
            for start_idx in range(0, n_blocks - win_size + 1, stride):
                win_blocks = blocks[start_idx:start_idx + win_size]
                win_label = f"{win_blocks[0]}-{win_blocks[-1]}"
                block_pairs = list(combinations(win_blocks, 2))

                all_win_data = []
                for subj in df_roi['subject'].unique():
                    subj_df = df_roi[df_roi['subject'] == subj]
                    if trials_type == 'learning':
                        mask = [
                            ((subj_df['block_from'] == b1) & (subj_df['block_to'] == b2)) |
                            ((subj_df['block_from'] == b2) & (subj_df['block_to'] == b1))
                            for b1, b2 in block_pairs
                        ]
                    else:
                        mask = [
                            ((subj_df['block'] == b1) & (subj_df['next_block'] == b2)) |
                            ((subj_df['block'] == b2) & (subj_df['next_block'] == b1))
                            for b1, b2 in block_pairs
                        ]
                    if len(mask) == 0:
                        continue
                    win_df = subj_df[np.logical_or.reduce(mask)].copy()
                    win_df["window"] = win_label
                    if len(win_df) > 4:
                        all_win_data.append(win_df)
                if not all_win_data:
                    continue

                win_data = pd.concat(all_win_data, ignore_index=True)
                for subj in win_data['subject'].unique():
                    subj_df = win_data[win_data['subject'] == subj]
                    try:
                        ols_model = smf.ols(
                            f"correlation ~ {' + '.join(regressors)}",
                            data=subj_df
                        ).fit()
                    except Exception as e:
                        print(f"OLS failed for {roi}, subj {subj}, window {win_label}: {e}")
                        continue
                    for reg in regressors:
                        all_betas.append({
                            'ROI': roi,
                            'subject': subj,
                            'window': win_label,
                            'regressor': reg,
                            'beta': ols_model.params.get(reg, np.nan)
                        })

        # --- LME computations: sequential block correlation and numeric block variable ---
        elif timeseries_model_type == 'LME':
        
            if trials_type == 'learning':
                df_roi["block"] = df_roi["block_from"]
                df_roi["next_block"] = df_roi["block_to"]
            else:
                df_roi["block"] = df_roi["block"]
                df_roi["next_block"] = df_roi["next_block"]
        
            # Keep only subsequent pairs (active1→active2, etc.)
            valid_pairs = [(blocks[i], blocks[i + 1]) for i in range(len(blocks) - 1)]
            df_roi = df_roi[df_roi.apply(lambda x: (x["block"], x["next_block"]) in valid_pairs, axis=1)].copy()
        
            # Assign block_num. When passive blocks are present: passive1→0, activeN→N,
            # passive2→max_active+1. Otherwise fall back to digit extraction (activeN→N)
            # so active11 is never declared as a category it doesn't occupy as block_from.
            _passive_in_blocks = [b for b in blocks if b in ('passive1', 'passive2')]
            if _passive_in_blocks:
                _max_active = max(int(''.join(filter(str.isdigit, b)))
                                  for b in blocks if b.startswith('active'))
                def _to_block_num(b):
                    if b == 'passive1':        return 0
                    if b.startswith('active'): return int(''.join(filter(str.isdigit, b)))
                    if b == 'passive2':        return _max_active + 1
                    return -1
                df_roi["block_num"] = pd.Categorical(
                    df_roi["block"].apply(_to_block_num), ordered=True)
            else:
                df_roi["block_num"] = pd.Categorical(
                    df_roi["block"].apply(lambda b: int(''.join(filter(str.isdigit, b)))),
                    ordered=True)
        
            # build formula (don’t wrap C() if already categorical)
            formula_terms = [f"{reg}*block_num" for reg in regressors]
            formula = "correlation ~ " + " + ".join(formula_terms)
        
            # Fit LME
            model = smf.mixedlm(formula, df_roi, groups=df_roi["subject"])
            result = model.fit()
            cov = result.cov_params()
        
            # Extract per-block effects
            ref_block = df_roi['block_num'].cat.categories[0]
        
            for reg in regressors:
                for block in df_roi['block_num'].cat.categories:
                    if block == ref_block:
                        beta = result.params.get(reg, np.nan)
                        se = np.sqrt(cov.loc[reg, reg]) if reg in cov.index else np.nan
                    else:
                        interaction_term = f"{reg}:block_num[T.{block}]"
                        beta = result.params.get(reg, 0) + result.params.get(interaction_term, 0)


                        interaction_term = f"{reg}:block_num[T.{block}]"
                        var_main = cov.loc[reg, reg]
                        var_inter = cov.loc[interaction_term, interaction_term]
                        cov_main_inter = cov.loc[reg, interaction_term]
                        se = np.sqrt(var_main + var_inter + 2 * cov_main_inter)


                    zval = beta / se if pd.notnull(se) and se > 0 else np.nan
                    pval = 2 * (1 - norm.cdf(abs(zval))) if pd.notnull(zval) else np.nan
        
                    all_betas.append({
                        'ROI': roi,
                        'window': str(block),
                        'model': model_name,
                        'regressor': reg,
                        'beta': beta,
                        'se': se,
                        'pval': pval,
                        'model_regressor': f"{model_name}:{reg}"})


    betas_df = pd.DataFrame(all_betas)

    # --- Group-level stats for OLS only ---
    if timeseries_model_type == 'OLS':
        stats_results = []
        for roi in ROIs:
            df_roi = betas_df[betas_df['ROI'] == roi]
            for reg in df_roi['regressor'].unique():
                for win in df_roi['window'].unique():
                    vals = df_roi.loc[(df_roi['regressor'] == reg) &
                                      (df_roi['window'] == win), 'beta'].dropna()
                    if len(vals) < 2:
                        continue
                    tstat, pval = ttest_1samp(vals, 0)
                    stats_results.append({
                        'ROI': roi,
                        'regressor': reg,
                        'window': win,
                        'tstat': tstat,
                        'pval': pval,
                        'mean_beta': vals.mean()
                    })
        stats_df = pd.DataFrame(stats_results)
    elif timeseries_model_type=='LME':
        stats_df = betas_df.copy()  # already includes betas + pvals from LME

    return betas_df, stats_df


# Funciton for computing mean decoding accuracy per ROI and trial type
def running_window_decoding(ROIs, decoded_feature, bin_size,subjects):
    doSmooth = True
    # --- load and concat precomputed single trial decoding results:
    data_dir = '/Users/grossman/Volumes/STATE_FORMATION/MRI_DATA/MRI_REAL/ROI_timeseries_LSS_5mm_smoothing_dropout_30_physio_reg_out/Decoding_active_trials/'
    all_dfs = []
    
    for sub in subjects:
        fpath = os.path.join(data_dir, f"{sub}_Decoding_active_trials_decoding_results_demean.csv")
        
        if not os.path.exists(fpath):
            print(f"Missing file: {fpath}")
            continue
        
        try:
            df_sub = pd.read_csv(fpath)
            if df_sub.empty:
                continue

            df_sub_filt = df_sub[df_sub['feature'] == decoded_feature].copy()  
            all_dfs.append(df_sub_filt)
            
        except pd.errors.EmptyDataError:
            print(f"Empty CSV skipped: {fpath}")

    df_results = pd.concat(all_dfs, ignore_index=True)
    print(f" Loaded decoding results for {len(all_dfs)} subjects -> {df_results.shape[0]} rows")

    # --- Loop over ROIs ---
    results = []
    for roi in ROIs:
        df_roi = df_results[df_results['ROI'] == roi].copy()
       
        all_acc = []
        max_bins = 0

        for sub, group in df_roi.groupby("subject"):
            correct = group["correct"].values
            if doSmooth:
                smooth_bin = bin_size * 2
                stride = bin_size  # half of smooth_bin
                n_bins = (len(correct) - smooth_bin) // stride + 1
                acc = [np.mean(correct[b*stride:b*stride+smooth_bin]) for b in range(n_bins)]
            else:
                n_bins = len(correct) // bin_size
                acc = [np.mean(correct[b*bin_size:(b+1)*bin_size]) for b in range(n_bins)]

            max_bins = max(max_bins, len(acc))
            all_acc.append(acc)

        if len(all_acc) == 0:
            results.append(None)
            continue

        acc_matrix = np.full((len(all_acc), max_bins), np.nan)
        for i, acc in enumerate(all_acc):
            acc_matrix[i, :len(acc)] = acc

        mean_acc = np.nanmean(acc_matrix, axis=0)
        sem_acc = np.nanstd(acc_matrix, axis=0) / np.sqrt(np.sum(~np.isnan(acc_matrix), axis=0))
        # same-val pairwise decoding is binary → chance = 0.5
        if decoded_feature == 'state_learning_same_val' or decoded_feature == 'state_estimation_same_val':
            chance = 0.5
        else:
            n_classes = df_sub_filt["y_true"].nunique()
            chance = 1 / n_classes if n_classes > 0 else np.nan

        results.append((roi, acc_matrix, mean_acc, sem_acc, chance))

    return results


#%% --- Run for learning and estimation ---

if timeseries_model_type=='decoding':
    _ofc_df = pd.read_csv(base_dir + timeseries_ROI  + '_corr_results_active_trials_all_pairs_collapsed_sides_zscored.csv')
    subjects = _ofc_df['subject'].unique()
    
    decoding_state_learning = running_window_decoding([timeseries_ROI], 'state_learning',24, subjects) #'state_learning_same_val' client_type_learning

    decoding_state_estimation = running_window_decoding([timeseries_ROI], 'state_estimation', 18, subjects) #state_estimation_same_val client_type_estimation

    decoding_value_learning = running_window_decoding([timeseries_ROI], 'diff_reward', 24, subjects)

    decoding_value_estimation = running_window_decoding([timeseries_ROI], 'abs_reward_estimation', 18, subjects)

else:


    corr_df_learning_time_series = pd.read_csv(base_dir + timeseries_ROI + '_corr_results_active_trials_all_pairs_collapsed_sides_zscored.csv')
    corr_df_learning_time_series = add_active_value_columns(corr_df_learning_time_series)

    betas_learning_state, stats_learning_state = running_window_glm(
        corr_df_learning_time_series,   [timeseries_ROI], 'state', 2, 1, 'learning', timeseries_model_type)

    betas_learning_context, stats_learning_context = running_window_glm(
        corr_df_learning_time_series,  [timeseries_ROI], 'context_house', 2, 1, 'learning', timeseries_model_type)


    corr_df_estimation_time_series = pd.read_csv(base_dir + timeseries_ROI + '_corr_results_passive_trials_all_2_all_all_pairs_zscored.csv')
    corr_df_estimation_time_series['val_sim'] = (
        1 - corr_df_estimation_time_series['abs_val_diff'] / max(house_values.values()))
    
    betas_est_state, stats_est_state = running_window_glm(
        corr_df_estimation_time_series,   [timeseries_ROI], 'state', 2, 1, 'estimation', timeseries_model_type)
    
    betas_est_context, stats_est_context = running_window_glm(
        corr_df_estimation_time_series,  [timeseries_ROI], 'context_house', 2, 1, 'estimation', timeseries_model_type)

    if plotValueTimeSeries:
        betas_learning_value, stats_learning_value = running_window_glm(
            corr_df_learning_time_series,  [timeseries_ROI], 'state_value', 2, 1, 'learning', timeseries_model_type)

        betas_est_value, stats_est_value = running_window_glm(
            corr_df_estimation_time_series,  [timeseries_ROI], 'state_value', 2, 1, 'estimation', timeseries_model_type)


#%% Plotting Figure 3C

def plot_running_window_styled(betas_learning, betas_est, width,
                               colors, alpha_sig, model_type,
                               axes=None,
                               regressors_learning=None,
                               regressors_estimation=None,
                               ylabel='same state X block\nbeta'):
    """
    Plot mean beta ± SEM per window for learning and estimation trials,
    with significance markers. Works for both OLS and LME.
    If axes are provided, plot into them; otherwise, create a new figure.
    """
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter

    sig_y_offsets = {
        'same_state': 0.22,
        'same_context': 0.16,
        'same_house_pair': 0.10,
        'same_house_type': 0.10
    }

    # --- Use existing axes if provided, else create ---
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(width, 2), sharey=False)
    else:
        fig = None  # don't create a new figure

    if regressors_learning is None:
        regressors_learning = ['same_state']
    if regressors_estimation is None:
        regressors_estimation = ['same_state']
    panel_titles = ['Learning trials', 'Estimation trials']

    for i, (ax, betas_df, regressors, title) in enumerate(zip(
        axes,
        [betas_learning, betas_est],
        [regressors_learning, regressors_estimation],
        panel_titles)):

        all_y = []

        for reg in regressors:
            df_reg = betas_df[betas_df['regressor'] == reg].copy()
            if df_reg.empty:
                continue

            # Sort windows numerically
            x_labels = sorted(df_reg['window'].unique(),
                              key=lambda w: int(''.join(filter(str.isdigit, w.split('-')[0]))))

            # Drop passive bookend windows: passive1 (n=0) is the silent reference;
            # passive2 (n>=12) is the endpoint; neither appears on the x-axis.
            def _is_active_win(w):
                n = int(''.join(filter(str.isdigit, w.split('-')[0])) or '0')
                return 0 < n < 12
            x_labels = [w for w in x_labels if _is_active_win(w)]
            x_display = [str(int(''.join(filter(str.isdigit, w.split('-')[0])))) for w in x_labels]
            x = np.arange(2, 2 + len(x_labels))

            y = df_reg.groupby('window')['beta'].mean().loc[x_labels].values

            # --- SEM ---
            if model_type == 'LME':
                if 'se' not in df_reg.columns:
                    raise ValueError("LME dataframe must include 'se' column per regressor")
                sem_val = df_reg['se'].iloc[0]
                sem = np.full_like(y, sem_val)
            else:
                sem = df_reg.groupby('window')['beta'].sem().loc[x_labels].values

            all_y.extend(y + sem)
            all_y.extend(y - sem)

            c = colors.get(reg, 'gray')

            # --- Plot mean ± SEM ---
            ax.plot(x, y, lw=1.2, label=reg, color=c)
            ax.fill_between(x, y - sem, y + sem, alpha=0.25, lw=0, color=c)

            # --- Significance markers ---
            sig_mask = np.zeros_like(y, dtype=bool)
            if model_type == 'OLS':
                pvals = []
                for w in x_labels:
                    samples = df_reg.query("window == @w")['beta']
                    if len(samples) > 1:
                        _, p = ttest_1samp(samples, 0)
                        pvals.append(p)
                    else:
                        pvals.append(np.nan)
                pvals = np.array(pvals)
                valid_mask = ~np.isnan(pvals)
                if np.any(valid_mask):
                    reject, _ = fdrcorrection(pvals[valid_mask], alpha=alpha_sig)
                    sig_mask[valid_mask] = reject
            elif model_type == 'LME':
                pvals = df_reg.groupby('window')['pval'].mean().loc[x_labels].values
                sig_mask = pvals < alpha_sig

            if np.any(sig_mask):
                if model_type == 'LME':
                    _ptp = np.ptp(all_y) if np.ptp(all_y) > 0 else max(abs(max(all_y, default=0.01)), 0.01)
                    y_sig = max(all_y) + 0.08 * _ptp
                    for xi, sig_pt in zip(x, sig_mask):
                        if sig_pt:
                            ax.plot([xi - 0.3, xi + 0.3], [y_sig, y_sig],
                                    color='gold', lw=2)
                else:
                    offset = sig_y_offsets.get(reg, 0.15)
                    y_pos = max(all_y) + offset * np.ptp(all_y)
                    for xi, sig_pt in zip(x, sig_mask):
                        if sig_pt:
                            ax.text(xi, y_pos, '*', color=c, ha='center', va='bottom',
                                    fontsize=10, fontweight='bold')

        # --- Axes styling ---
        ax.axhline(0, color='k', linestyle='--', lw=0.8)
        for spine in ['top','right']:
            ax.spines[spine].set_visible(False)

        if axes is None:
            ax.set_title(title, fontsize=10, fontweight='bold', pad=6)
        
        # --- X limits and ticks from data ---
        x_pad = 0.5
        ax.set_xlim(x.min() - x_pad, x.max() + x_pad)

        # x_display is already built above (block numbers as strings, passive bookends excluded)
        # Show even block numbers as ticks
        xtick_positions, xtick_labels = [], []
        for xi, lbl in zip(x, x_display):
            if lbl.lstrip('-').isdigit() and int(lbl) % 2 == 0:
                xtick_positions.append(xi)
                xtick_labels.append(lbl)
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(xtick_labels)

        # No tick lines, labels tight to spine
        ax.tick_params(axis='x', length=0, pad=2, labelsize=6)

        # X label slightly elevated
        ax.set_xlabel('block #', fontsize=8, labelpad=-0.5)

        # --- Y limits: pad 15% above/below data range ---
        if all_y:
            y_lo, y_hi = min(all_y), max(all_y)
            y_pad = 0.15 * (y_hi - y_lo) if y_hi != y_lo else 0.005
            ax.set_ylim(y_lo - y_pad, y_hi + y_pad)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.tick_params(axis='y', which='both', right=False, labelsize=6)
        ax.yaxis.set_ticks_position('left')


    axes[0].set_ylabel(ylabel, fontsize=8)
    
    # Only show figure if we created it here
    if fig is not None:
        plt.tight_layout()
        plt.show()

#%% Function for plotting decoding in a running window:
def plot_running_window_styled_decoding(decoding_state_learning,
                                       decoding_state_estimation,
                                       width,
                                       colors,
                                       alpha_sig,
                                       bin_sizes,
                                       axes=None):
    
    from scipy.stats import ttest_1samp
    from statsmodels.stats.multitest import fdrcorrection
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator
    import numpy as np
    import matplotlib.pyplot as plt

    # --- Use existing axes if provided, else create ---
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(width, 2), sharey=False)
    else:
        fig = None

    ROI, acc_matrix_learning, mean_acc_learning, sem_acc_learning, chance_learning = decoding_state_learning
    ROI, acc_matrix_estimation, mean_acc_estimation, sem_acc_estimation, chance_estimation = decoding_state_estimation

    bins_learning = np.arange(1, len(mean_acc_learning) + 1)
    bins_estimation = np.arange(1, len(mean_acc_estimation) + 1)

    for ax, mean_acc, sem_acc, x, bin_size, chance, acc_matrix in zip(
        axes,
        [mean_acc_learning, mean_acc_estimation],
        [sem_acc_learning, sem_acc_estimation],
        [bins_learning, bins_estimation],
        [bin_sizes[0], bin_sizes[1]],
        [chance_learning, chance_estimation],
        [acc_matrix_learning, acc_matrix_estimation]):

        c = colors.get('same_state', 'gray')
        mean_acc, sem_acc, chance, acc_matrix  = mean_acc*100, sem_acc*100, chance*100, acc_matrix*100

        # --- Plot mean ± SEM ---
        ax.plot(x, mean_acc, lw=1.2, color=c)
        ax.fill_between(x, mean_acc - sem_acc, mean_acc + sem_acc, alpha=0.25, lw=0, color=c)

        # --- Chance line ---
        ax.axhline(chance, color='k', linestyle='--', lw=0.8)

        # --- Statistics: 1-sample t-test vs chance ---
        pvals = np.array([
            ttest_1samp(acc_matrix[:, ii], chance, nan_policy='omit')[1]
            for ii in range(acc_matrix.shape[1])
        ])
        reject, _ = fdrcorrection(pvals, alpha=alpha_sig)

        # --- Expand y-limits to make room for sig bars ---
        ymin = np.min(mean_acc - sem_acc)
        ymax = np.max(mean_acc + sem_acc)

        # Ensure chance is visible
        epsilon = 1.0  # percent below chance
        if ymin > chance - epsilon:
            ymin = chance - epsilon

        yrange = ymax - ymin
        ax.set_ylim(ymin - 0.05 * yrange, ymax + 0.15 * yrange)

        # --- Yellow significance bars ---
        y_sig = ax.get_ylim()[1] - 0.05 * np.ptp(ax.get_ylim())
        for xi, sig in zip(x, reject):
            if sig:
                ax.plot([xi - 0.3, xi + 0.3], [y_sig, y_sig], color='gold', lw=2)

        # --- Axes styling ---
        for spine in ['top','right']:
            ax.spines[spine].set_visible(False)

        xticks = [t for t in x if t % 2 == 0 and t <= 10]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks)

        ax.tick_params(axis='x', length=2, pad=2, labelsize=6)
        ax.tick_params(axis='y', length=2, pad=2, labelsize=6)

        ax.set_xlabel(f'Time bin \n[bin size: {bin_size*2} trials; stride: {bin_size} trials]',
                      fontsize=8, labelpad=-0.5)

        ax.margins(x=0.01)
        ax.set_xlim(1, len(x))

        # --- Integer y ticks ---
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.tick_params(axis='y', which='both', right=False)
        ax.yaxis.set_ticks_position('left')

    axes[0].set_ylabel('State decoding\naccuracy [%]', fontsize=8)

    if fig is not None:
        plt.tight_layout()
        plt.show()


#%% Figure 3AB: plot a single panel function 
import seaborn as sns
import matplotlib.pyplot as plt

def swarm_at_x(ax, x_center, y, color, size, width, zorder=10, edgecolor='k'):
    # temporary invisible swarm to get offsets
    tmp_fig, tmp_ax = plt.subplots()
    sns.swarmplot(x=[0]*len(y), y=y, ax=tmp_ax, size=size, color=color)
    offsets = tmp_ax.collections[0].get_offsets()
    plt.close(tmp_fig)

    # rescale & recenter x offsets
    x_offsets = offsets[:, 0]
    x_offsets = (x_offsets - x_offsets.mean())
    x_offsets *= width / (np.max(np.abs(x_offsets)) + 1e-6)

    ax.scatter(
        x_center + x_offsets,
        y,
        s=size,
        color=color,
        edgecolor=edgecolor,
        alpha=1,
        linewidth=0.5,
        zorder=zorder)


def plot_panel(ax, df, results_df, lme_per_period_df, roi,
               regressor_order, regressor_labels,
               palette, irrelevant_colors):

    import numpy as np

    # Main/value regressors (have period interaction) get 1.5 spacing;
    # consecutive irrel regressors (no interaction) get 0.7 spacing.
    _value_regs = {'val_sim', 'same_val', 'vis_sim', 'irrelevant_sim'}
    x_positions = np.zeros(len(regressor_order))
    for _j in range(1, len(regressor_order)):
        _prev_is_irrel = (_j - 1 != 0) and (regressor_order[_j - 1] not in _value_regs)
        _curr_is_irrel = (_j != 0) and (regressor_order[_j] not in _value_regs)
        _gap = 0.95 if (_prev_is_irrel and _curr_is_irrel) else 1.85
        x_positions[_j] = x_positions[_j - 1] + _gap

    x_offset_main, jitter_main = 0.44, 0.08
    scatter_main, mean_main = 5, 4
    scale_irrel = 0.35
    irrel_gap_scale = 0.3

    roi_df = df[df['ROI'] == roi].copy()
    if roi_df.empty:
        print(f"No data for ROI {roi}")
        return

    roi_df['period_label'] = roi_df['period'].map({-1: 'Early', 1: 'Late'})

    # =======================
    # Plot violins and swarms
    # =======================
    for j, reg in enumerate(regressor_order):
        xpos = x_positions[j]
        is_irrel = (j != 0) and (reg not in _value_regs)
        colors = ([palette['Early'], palette['Late']]
                  if j == 0 else irrelevant_colors[j - 1])

        # --- Violins ---
        for period, color, side in zip(['Early', 'Late'], colors, [-1, 1]):
            vals = roi_df.query(
                "regressor == @reg and period_label == @period"
            )['beta'].dropna()
            if len(vals) < 2:
                continue

            width = scale_irrel if is_irrel else 0.7
            parts = ax.violinplot(
                vals, positions=[xpos], widths=width,
                showmeans=False, showmedians=False, showextrema=False
            )

            outer_shift = x_offset_main if not is_irrel else 0.2
            for pc in parts['bodies']:
                verts = pc.get_paths()[0].vertices
                m = np.mean(verts[:, 0])

                if side < 0:
                    mask = verts[:, 0] <= m
                    verts[mask, 0] = xpos - outer_shift - (m - verts[mask, 0])
                    verts[~mask, 0] = xpos - outer_shift
                else:
                    mask = verts[:, 0] >= m
                    verts[mask, 0] = xpos + outer_shift + (verts[mask, 0] - m)
                    verts[~mask, 0] = xpos + outer_shift

                pc.set_facecolor(color)
                pc.set_alpha(0.4)
                pc.set_edgecolor('none')

        # --- Swarms + means ---
        mean_vals = {}
        for period_code, shift in zip([-1, 1], [-x_offset_main, x_offset_main]):
            shift *= irrel_gap_scale if is_irrel else 1

            color = (irrelevant_colors[j - 1][0 if period_code == -1 else 1]
                     if j != 0 else
                     palette['Early' if period_code == -1 else 'Late'])

            subj_data = roi_df.query(
                "regressor == @reg and period == @period_code"
            )
            if subj_data.empty:
                continue

            y = subj_data['beta'].values

            swarm_at_x(
                ax, xpos + shift, y,
                color,
                scatter_main / 2 if is_irrel else scatter_main,
                jitter_main * 2,
                zorder=10,
                edgecolor='none')

            if len(y) > 1:
                mean_val = np.mean(y)
                sem_val = np.std(y, ddof=1) / np.sqrt(len(y))

                mean_vals[period_code] = mean_val

                ax.errorbar(
                    xpos + shift, mean_val, yerr=sem_val,
                    fmt='o', color='k',
                    elinewidth=1,
                    capsize=3 if is_irrel else 4,
                    markersize=mean_main/2 if is_irrel else mean_main,
                    zorder=15)

        # --- Connect Early/Late means (only for regressors with period interaction) ---
        if not is_irrel and -1 in mean_vals and 1 in mean_vals:
            ax.plot(
                [xpos - x_offset_main,
                 xpos + x_offset_main],
                [mean_vals[-1], mean_vals[1]],
                color='k', lw=1.2, zorder=14)


        # --- Connect Early/Late individual subjects for state and value regressors ---
        if reg in ('same_state', 'val_sim', 'same_val', 'vis_sim', 'irrelevant_sim'):
            early_s = roi_df.query(
                "regressor == @reg and period == -1"
            ).set_index('subject')['beta']
            late_s = roi_df.query(
                "regressor == @reg and period == 1"
            ).set_index('subject')['beta']
            common_subjects = early_s.index.intersection(late_s.index)
            for subj in common_subjects:
                ax.plot(
                    [xpos - x_offset_main * 0.45,
                     xpos + x_offset_main * 0.45],
                    [early_s[subj], late_s[subj]],
                    color='grey', alpha=0.4, lw=0.4, zorder=5)

    # =======================
    # DATA-DRIVEN Y LIMITS
    # =======================
    y_min = np.nanmin(roi_df['beta'].values)
    y_max = np.nanmax(roi_df['beta'].values)
    data_range = y_max - y_min

    y_pad = 0.05 * data_range
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    # =======================
    # Annotation positions
    # =======================
    ylim = ax.get_ylim()
    data_height = ylim[1] - ylim[0]

    single_text_y = ylim[1] - 0.08 * data_height
    inter_line_y  = ylim[1] + 0.08 * data_height
    inter_text_y  = ylim[1] + 0.10 * data_height

    # --- Interaction significance ---
    for j, reg in enumerate(regressor_order):
        xpos = x_positions[j]

        inter_rows = results_df.loc[
            (results_df['ROI'] == roi) &
            (results_df['regressor'] == reg) &
            (results_df['effect'] == 'period:interaction'),
            'pval'
        ]
        # Skip regressors that weren't modelled with a period interaction
        if inter_rows.empty or np.isnan(inter_rows.values[0]):
            continue

        p_val = inter_rows.values[0]

        sig = ('***' if p_val < 0.001 else
               '**' if p_val < 0.01 else
               '*' if p_val < 0.05 else "n.s.")

        cap_h = 0.015 * data_height
        _is_irrel = (j != 0) and (reg not in _value_regs)
        _bracket_hw = 0.2 if _is_irrel else x_offset_main

        ax.plot(
            [xpos - _bracket_hw, xpos + _bracket_hw],
            [inter_line_y] * 2,
            color='k', lw=0.8, clip_on=False
        )
        ax.plot(
            [xpos - _bracket_hw] * 2,
            [inter_line_y, inter_line_y - cap_h],
            color='k', lw=0.8, clip_on=False
        )
        ax.plot(
            [xpos + _bracket_hw] * 2,
            [inter_line_y, inter_line_y - cap_h],
            color='k', lw=0.8, clip_on=False
        )
        ax.text(
            xpos,
            inter_text_y if sig == 'n.s.' else 0.9 * inter_text_y,
            sig,
            ha='center',
            va='bottom',
            fontsize=6 if sig == 'n.s.' else 12,
            zorder=20,
            clip_on=False
        )

    # --- Single-sample LME stars ---
    for j, reg in enumerate(regressor_order):
        xpos = x_positions[j]
        is_irrel = (j != 0) and (reg not in _value_regs)

        for period_code, shift in zip([-1, 1], [-x_offset_main, x_offset_main]):
            shift *= irrel_gap_scale if is_irrel else 1

            rows = lme_per_period_df.query(
                "regressor == @reg and period == @period_code and ROI == @roi"
            ).pval.values
            if len(rows) == 0 or np.isnan(rows[0]):
                continue
            p = rows[0]

            sig = ('***' if p < 0.001 else
                   '**' if p < 0.01 else
                   '*' if p < 0.05 else '')

            if sig:
                ax.text(
                    xpos + shift,
                    single_text_y,
                    sig,
                    ha='center',
                    va='bottom',
                    fontsize=10,
                    zorder=20,
                    clip_on=False)

    # =======================
    # Axis aesthetics
    # =======================
    ax.axhline(0, color='k', linestyle='--', lw=1)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='y', direction='out')
    ax.tick_params(axis='x', length=0)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([regressor_labels[r] for r in regressor_order])

    for j, (label, reg) in enumerate(zip(ax.get_xticklabels(), regressor_order)):
        _tick_is_irrel = (j != 0) and (reg not in _value_regs)
        if _tick_is_irrel:
            label.set_fontsize(8)
            label.set_rotation(35)
            label.set_ha('right')
            label.set_rotation_mode('anchor')
        else:
            label.set_fontsize(10)

    # Slight x padding
    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0] - 0.25, xlim[1] + 0.1)


#%% Plot Figure 3AB with an preceding empty columns for ROIs images 

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

# --- ROI mapping & color palettes ---
roi_name_map = {
    "VISUALFUNC": "Visual ctx.",
    "OFC": "OFC",
    "HPC": "Hippocampus"}

palette = {'Early': 'pink', 'Late': '#C00000'}   # darker red
value_sim_palette = {'Early': '#D4AAEE', 'Late': '#6B2FA0'}   # light / dark purple
vis_sim_palette   = {'Early': '#A8D5C2', 'Late': '#1B7A5A'}   # light / dark teal


custom_colors = {
    'same_state': '#C00000',
    'same_context': '#016B61',
    'same_house_pair': '#70B2B2',
    'same_house_type': '#70B2B2'}
value_colors = {'same_state': value_sim_palette['Late'],
                'val_sim':    value_sim_palette['Late']}   # dark purple matching val_sim late

if model_name in ('value', 'value_wo_same_state'):
    regressors_learning = ['val_sim', 'same_irrel_client_feature',
                           'same_irrel_left_house_feature', 'same_irrel_right_house_feature']
    labels_learning = {'val_sim': 'value',
                       'same_irrel_client_feature': 'client',
                       'same_irrel_left_house_feature': 'left\nhouse',
                       'same_irrel_right_house_feature': 'right\nhouse'}
    regressors_estimation = ['val_sim', 'same_irrel_client_feature', 'same_irrel_house_feature']
    labels_estimation = {'val_sim': 'value',
                         'same_irrel_client_feature': 'client',
                         'same_irrel_house_feature': 'house'}
elif model_name == 'state_within_same_val':
    regressors_learning = ['same_state', 'same_irrel_client_feature']
    labels_learning = {'same_state': 'state', 'same_irrel_client_feature': 'client'}
    regressors_estimation = ['same_state', 'same_irrel_client_feature', 'same_irrel_house_feature']
    labels_estimation = {'same_state': 'state', 'same_irrel_client_feature': 'client',
                         'same_irrel_house_feature': 'house'}
elif model_name in ('state_value', 'state_value_within_context'):
    regressors_learning = ['same_state', 'val_sim', 'same_irrel_client_feature',
                           'same_irrel_left_house_feature', 'same_irrel_right_house_feature']
    labels_learning = {'val_sim': 'value', 'same_state': 'state',
                       'same_irrel_client_feature': 'client',
                       'same_irrel_left_house_feature': 'left house',
                       'same_irrel_right_house_feature': 'right house'}
    regressors_estimation = ['same_state', 'val_sim', 'same_irrel_client_feature', 'same_irrel_house_feature']
    labels_estimation = {'val_sim': 'value', 'same_state': 'state',
                         'same_irrel_client_feature': 'client',
                         'same_irrel_house_feature': 'house'}
elif model_name == 'state_value_vis':
    regressors_learning = ['same_state', 'val_sim', 'vis_sim', 'same_irrel_client_feature']
    labels_learning = {'same_state': 'state', 'val_sim': 'value',
                       'vis_sim': 'vis sim', 'same_irrel_client_feature': 'client'}
    regressors_estimation = ['same_state', 'val_sim', 'vis_sim', 'same_irrel_client_feature']
    labels_estimation = {'same_state': 'state', 'val_sim': 'value',
                         'vis_sim': 'vis sim', 'same_irrel_client_feature': 'client'}
elif model_name == 'state_value_binary':
    regressors_learning = ['same_state', 'same_val', 'same_irrel_client_feature',
                           'same_irrel_left_house_feature', 'same_irrel_right_house_feature']
    labels_learning = {'same_state': 'state', 'same_val': 'same\nval',
                       'same_irrel_client_feature': 'client',
                       'same_irrel_left_house_feature': 'left\nhouse',
                       'same_irrel_right_house_feature': 'right\nhouse'}
    regressors_estimation = ['same_state', 'same_val', 'same_irrel_client_feature', 'same_irrel_house_feature']
    labels_estimation = {'same_state': 'state', 'same_val': 'same\nval',
                         'same_irrel_client_feature': 'client',
                         'same_irrel_house_feature': 'house'}
elif model_name == 'context_house':
    regressors_learning = ['same_context', 'same_house_pair', 'same_irrel_client_feature',
                           'same_irrel_left_house_feature', 'same_irrel_right_house_feature']
    labels_learning = {'same_context': 'context', 'same_house_pair': 'house\npair',
                       'same_irrel_client_feature': 'client',
                       'same_irrel_left_house_feature': 'left\nhouse',
                       'same_irrel_right_house_feature': 'right\nhouse'}
    regressors_estimation = ['same_context', 'same_house_type', 'same_irrel_client_feature',
                             'same_irrel_house_feature']
    labels_estimation = {'same_context': 'context', 'same_house_type': 'house\ntype',
                         'same_irrel_client_feature': 'client',
                         'same_irrel_house_feature': 'house'}
elif model_name == 'pseudo-state':
    regressors_learning   = ['same_pseudo_state', 'same_state']
    labels_learning       = {'same_pseudo_state': 'pseudo\nstate', 'same_state': 'state'}
    regressors_estimation = ['same_pseudo_state', 'same_state']
    labels_estimation     = {'same_pseudo_state': 'pseudo\nstate', 'same_state': 'state'}
elif model_name in ('state_relevance', 'state_relevance_binary'):
    regressors_learning = regressors_estimation = ['same_state', 'irrelevant_sim', 'val_sim']
    labels_learning = labels_estimation = {
        'same_state': 'state', 'irrelevant_sim': 'irrelevant\nfeature', 'val_sim': 'value'}
else:
    regressors_learning = ['same_state', 'same_irrel_client_feature',
                           'same_irrel_left_house_feature', 'same_irrel_right_house_feature']
    labels_learning = {'same_state': 'state', 'same_irrel_client_feature': 'client',
                       'same_irrel_left_house_feature': 'left\nhouse',
                       'same_irrel_right_house_feature': 'right\nhouse'}
    regressors_estimation = ['same_state', 'same_irrel_client_feature', 'same_irrel_house_feature']
    labels_estimation = {'same_state': 'state', 'same_irrel_client_feature': 'client',
                         'same_irrel_house_feature': 'house'}

n_irrel_learning   = len(regressors_learning) - 1
n_irrel_estimation = len(regressors_estimation) - 1
irrel_colors_learning   = [('lightsteelblue', 'navy')] * n_irrel_learning
irrel_colors_estimation = [('lightsteelblue', 'navy')] * n_irrel_estimation

# Color val_sim / same_val in purple, vis_sim in teal
if model_name in ('state_value', 'state_value_binary'):
    irrel_colors_learning[0]   = (value_sim_palette['Early'], value_sim_palette['Late'])
    irrel_colors_estimation[0] = (value_sim_palette['Early'], value_sim_palette['Late'])
elif model_name == 'state_value_vis':
    irrel_colors_learning[0]   = (value_sim_palette['Early'], value_sim_palette['Late'])  # val_sim
    irrel_colors_learning[1]   = (vis_sim_palette['Early'],   vis_sim_palette['Late'])    # vis_sim
    irrel_colors_estimation[0] = (value_sim_palette['Early'], value_sim_palette['Late'])
    irrel_colors_estimation[1] = (vis_sim_palette['Early'],   vis_sim_palette['Late'])
elif model_name in ('state_relevance', 'state_relevance_binary'):
    # regressors = ['same_state', 'irrelevant_sim', 'val_sim'] -> irrel_colors index 1 is val_sim
    # (irrelevant_sim, index 0, keeps the default lightsteelblue/navy pair)
    irrel_colors_learning[1]   = (value_sim_palette['Early'], value_sim_palette['Late'])
    irrel_colors_estimation[1] = (value_sim_palette['Early'], value_sim_palette['Late'])

# --- Table S1: LME summary (rows = regressors, columns = ROIs) ---
def _lme_summary_table(results_df, regressors, rois, interaction_reg='same_state'):
    """
    Rows = regressors in order: interaction_reg, period:interaction_reg, then remaining.
    Columns = ROIs.
    Each cell (two lines):
        beta=x.xxx±x.xxx
        z=x.xx, p=.xxx***
    """
    _reg_labels = {
        'same_state':                     'same state',
        f'period:{interaction_reg}':      'learning phase x\nsame state',
        'val_sim':                        'value similarity',
        'same_irrel_client_feature':      'same irrelevant\nclient feature',
        'same_irrel_left_house_feature':  'same irrelevant\nleft house feature',
        'same_irrel_right_house_feature': 'same irrelevant\nright house feature',
        'same_irrel_house_feature':       'same irrelevant\nhouse feature',
        'irrelevant_sim':                 'irrelevant similarity',
    }

    def _stars(p):
        return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''

    def _fmt_p(p):
        if p < 0.001:
            return 'p<.001'
        elif p > 0.6:
            return 'p=' + f'{p:.1f}'.lstrip('0')
        else:
            return 'p=' + f'{p:.3f}'.lstrip('0')

    def _fmt(b, se, p):
        if any(np.isnan(v) for v in (b, se, p)):
            return 'n/a'
        z = b / se if se != 0 else np.nan
        z_str = f'z={z:.2f}' if pd.notna(z) else 'z=n/a'
        return f'beta={b:.3f}±{se:.3f}\n{z_str}, {_fmt_p(p)}{_stars(p)}'

    remaining  = [r for r in regressors if r != interaction_reg]
    row_order  = [interaction_reg, f'period:{interaction_reg}'] + remaining

    data = {reg: {} for reg in row_order}
    for roi in rois:
        df_roi = results_df[results_df['ROI'] == roi]
        for reg in regressors:
            r = df_roi[(df_roi['regressor'] == reg) & (df_roi['effect'] == 'main')]
            data[reg][roi] = _fmt(*r[['beta','se','pval']].values[0]) if not r.empty else 'n/a'
        ikey = f'period:{interaction_reg}'
        r = df_roi[(df_roi['regressor'] == interaction_reg) & (df_roi['effect'] == 'period:interaction')]
        data[ikey][roi] = _fmt(*r[['beta','se','pval']].values[0]) if not r.empty else 'n/a'

    df = pd.DataFrame(data).T.loc[row_order, rois]
    df.index = [_reg_labels.get(r, r) for r in row_order]
    return df

table_s1_learning   = _lme_summary_table(results_df_learning,   regressors_learning,   ROIs)
table_s1_estimation = _lme_summary_table(results_df_estimation, regressors_estimation, ROIs)

table_s1_learning.to_csv(  save_dir + f'TableS1_learning_{model_name}.csv')
table_s1_estimation.to_csv(save_dir + f'TableS1_estimation_{model_name}.csv')
print("\n=== Table S1 — Learning ===")
print(table_s1_learning.to_string(float_format=lambda x: f'{x:.4f}'))
print("\n=== Table S1 — Estimation ===")
print(table_s1_estimation.to_string(float_format=lambda x: f'{x:.4f}'))

# --- Figure size ---
_n_ts_rows = 2 if plotValueTimeSeries else 1
total_rows = len(ROIs) + _n_ts_rows  # extra rows: state decoding + (optionally) value timeseries
n_regressors = max(len(regressors_learning), len(regressors_estimation))
fig_width = 7.2

# height ratios: 1.0 per RSA row, 0.55 per decoding row
height_ratios = [1.0] * len(ROIs)
height_ratios.insert(1, 0.55)  # state decoding row after OFC (index 1)
if plotValueTimeSeries:
    height_ratios.insert(2, 0.55)  # value decoding row immediately below
fig_height = min(10.0, sum(height_ratios) * 2.5)  # capped at 9 inches
fig = plt.figure(figsize=(fig_width, fig_height))

# Learning/estimation column widths: 1:0.8 by default (most models show more
# regressor bars in learning than estimation), but state_relevance has the same
# 3 regressors in both sections, so its columns should be equal width.
_learn_col_ratio, _est_col_ratio = (
    (1, 1) if model_name in ('state_relevance', 'state_relevance_binary') else (1, 0.8))

gs = gridspec.GridSpec(total_rows, 3, figure=fig,
                       width_ratios=[0.65, _learn_col_ratio, _est_col_ratio],
                       height_ratios=height_ratios,
                       hspace=0.8, wspace=0.3)

axes = []
row_offset = 0
for i, roi in enumerate(ROIs):
    # Insert Figure 3C after OFC (second ROI)
    if i == 1:
        row_3c = i + row_offset 
        ax_empty_3c = fig.add_subplot(gs[row_3c, 0])
        ax_empty_3c.axis('off')
        ax_learn_3c = fig.add_subplot(gs[row_3c, 1])
        ax_est_3c   = fig.add_subplot(gs[row_3c, 2])

        if timeseries_model_type =='decoding':
            if decoding_state_learning[0] is not None and decoding_state_estimation[0] is not None:
                plot_running_window_styled_decoding(
                    decoding_state_learning[0],
                    decoding_state_estimation[0],
                    width=5,
                    colors=custom_colors,
                    alpha_sig=0.05,
                    bin_sizes=[24, 18],
                    axes=[ax_learn_3c, ax_est_3c])
            elif decoding_state_learning[0] is not None:
                plot_running_window_styled_decoding(
                    decoding_state_learning[0],
                    decoding_state_learning[0],
                    width=5,
                    colors=custom_colors,
                    alpha_sig=0.05,
                    bin_sizes=[24, 24],
                    axes=[ax_learn_3c, ax_est_3c])
                ax_learn_3c.set_title('Learning (same-val pairs)', fontsize=7)
                ax_est_3c.set_visible(False)
            else:
                ax_learn_3c.text(0.5, 0.5, 'No decoding data', ha='center', va='center',
                                 transform=ax_learn_3c.transAxes, fontsize=7, color='gray')
                ax_est_3c.axis('off')

        else:

            plot_running_window_styled(
                betas_learning_state,
                betas_est_state,
                width=5,
                colors=custom_colors,
                alpha_sig=0.05,
                model_type=timeseries_model_type,
                axes=[ax_learn_3c, ax_est_3c])

        row_offset += 1  # shift down for state decoding row

        # --- Value decoding row (diff_reward / abs_reward_estimation) ---
        if plotValueTimeSeries:
            row_3c_val = i + row_offset
            ax_empty_3c_val = fig.add_subplot(gs[row_3c_val, 0])
            ax_empty_3c_val.axis('off')
            ax_learn_3c_val = fig.add_subplot(gs[row_3c_val, 1])
            ax_est_3c_val   = fig.add_subplot(gs[row_3c_val, 2])

            if timeseries_model_type == 'decoding':
                if decoding_value_learning[0] is not None and decoding_value_estimation[0] is not None:
                    # Both available: full plot
                    plot_running_window_styled_decoding(
                        decoding_value_learning[0],
                        decoding_value_estimation[0],
                        width=5,
                        colors= value_colors,
                        alpha_sig=0.05,
                        bin_sizes=[24, 18],
                        axes=[ax_learn_3c_val, ax_est_3c_val])
                elif decoding_value_learning[0] is not None:
                    # Learning only (abs_reward_estimation not yet available)
                    plot_running_window_styled_decoding(
                        decoding_value_learning[0],
                        decoding_value_learning[0],
                        width=5,
                        colors=value_colors,
                        alpha_sig=0.05,
                        bin_sizes=[24, 24],
                        axes=[ax_learn_3c_val, ax_est_3c_val])
                    ax_learn_3c_val.set_title('diff_reward (learning)', fontsize=6)
                    ax_est_3c_val.set_visible(False)
                else:
                    ax_learn_3c_val.text(0.5, 0.5, 'Value decoding results\nnot yet available',
                                         ha='center', va='center', transform=ax_learn_3c_val.transAxes,
                                         fontsize=7, color='gray')
                    ax_est_3c_val.axis('off')

            else:
                plot_running_window_styled(
                    betas_learning_value,
                    betas_est_value,
                    width=5,
                    colors=value_colors,
                    alpha_sig=0.05,
                    model_type=timeseries_model_type,
                    axes=[ax_learn_3c_val, ax_est_3c_val],
                    regressors_learning=['val_sim'],
                    regressors_estimation=['val_sim'],
                    ylabel='value similarity X block\nbeta')

            row_offset += 1  # shift down for value decoding row

    # Current ROI row (shifted)
    ax_empty = fig.add_subplot(gs[i + row_offset, 0])
    ax_empty.axis("off")
    ax_learn = fig.add_subplot(gs[i + row_offset, 1])
    ax_est   = fig.add_subplot(gs[i + row_offset, 2])

    axes.append((ax_learn, ax_est))

    # Plot panels
    plot_panel(ax_learn, individual_df_learning, results_df_learning, lme_per_period_df_learning, roi,
               regressors_learning, labels_learning, palette, irrel_colors_learning)

    plot_panel(ax_est, individual_df_estimation, results_df_estimation, lme_per_period_df_estimation, roi,
               regressors_estimation, labels_estimation, palette, irrel_colors_estimation)

    ax_learn.set_ylabel(f"{roi_name_map.get(roi, roi)}\nBeta", fontsize=10, fontweight='bold')

# --- Global legend ---
ax0 = axes[0][0]
legend_handles = [
    (Line2D([0],[0], color=palette['Early'], lw=4),
     Line2D([0],[0], color=palette['Late'], lw=4)),
    (Line2D([0],[0], color=value_sim_palette['Early'], lw=4),
     Line2D([0],[0], color=value_sim_palette['Late'], lw=4)),
    (Line2D([0],[0], color='lightsteelblue', lw=4),
     Line2D([0],[0], color='navy', lw=4)),
]
legend_labels = ["Same state: early / late",
                 "Value similarity: early / late",
                 # "Visual similarity: early / late",
                 "Irrelevant features: early / late"]
ax0.legend(
    legend_handles,
    legend_labels,
    fontsize=10,
    handler_map={tuple: HandlerTuple(ndivide=None)},
    frameon=False,
    loc='lower center',
    bbox_to_anchor=(1.04, 1.06),
    ncol=4
)

# --- Equalize y-ticks ---
target_nticks = 5
for row_axes in axes:
    for ax in row_axes:
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=target_nticks, prune=None, integer=False))

# Titles
axes[0][0].set_title("Learning trials", fontsize=10, fontweight='bold', pad=30)
axes[0][1].set_title("Estimation trials", fontsize=10, fontweight='bold', pad=30)

# Bring the two time-series rows (state and value) closer together vertically
if plotValueTimeSeries:
    fig.canvas.draw()  # force layout so get_position() returns real values
    _ts_pairs = [(ax_learn_3c, ax_est_3c), (ax_learn_3c_val, ax_est_3c_val)]
    _pos_state_l = ax_learn_3c.get_position()
    _pos_val_l   = ax_learn_3c_val.get_position()
    _gap = _pos_state_l.y0 - (_pos_val_l.y0 + _pos_val_l.height)  # gap between the two rows
    _shift = _gap * 0.4  # close by 40%
    for ax in (ax_learn_3c_val, ax_est_3c_val, ax_empty_3c_val):
        p = ax.get_position()
        ax.set_position([p.x0, p.y0 + _shift, p.width, p.height])

# Save
fig.savefig(save_dir + f"Figure3_vertical_room_for_ROIs_{model_name}.pdf", dpi=600, transparent=True, bbox_inches="tight")
plt.show()


#%% Plot Figure S2 with timeseries for the remaining ROIs

ROIs_fig_S2 = ['HPC', 'FFA', 'VISUALFUNC_HIGHORDER']

if timeseries_model_type == 'decoding':
    _ofc_df = pd.read_csv(base_dir + 'OFC_corr_results_active_trials_all_pairs_collapsed_sides_zscored.csv')
    subjects = _ofc_df['subject'].unique()

    decoding_state_learning_sup = running_window_decoding(
        ROIs_fig_S2, 'state_learning', 24, subjects)
    decoding_state_estimation_sup = running_window_decoding(
        ROIs_fig_S2, 'state_estimation', 18, subjects)

else:
    # Load learning and estimation CSVs for S2 ROIs
    _dfs_l_s2, _dfs_e_s2 = [], []
    for _roi in ROIs_fig_S2:
        try:
            _df_l = pd.read_csv(base_dir + _roi + '_corr_results_active_trials_all_pairs_collapsed_sides_zscored.csv')
            _df_l = add_active_value_columns(_df_l)
            _dfs_l_s2.append(_df_l)
        except FileNotFoundError:
            print(f"[S2] Missing learning CSV for {_roi}")
        try:
            _df_e = pd.read_csv(base_dir + _roi + '_corr_results_passive_trials_all_2_all_all_pairs_zscored.csv')
            _df_e['val_sim'] = 1 - _df_e['abs_val_diff'] / max(house_values.values())
            _dfs_e_s2.append(_df_e)
        except FileNotFoundError:
            print(f"[S2] Missing estimation CSV for {_roi}")

    corr_df_learning_s2   = pd.concat(_dfs_l_s2, ignore_index=True) if _dfs_l_s2 else pd.DataFrame()
    corr_df_estimation_s2 = pd.concat(_dfs_e_s2, ignore_index=True) if _dfs_e_s2 else pd.DataFrame()

    betas_learning_state_s2, _ = running_window_glm(
        corr_df_learning_s2,   ROIs_fig_S2, 'state',       2, 1, 'learning',   timeseries_model_type)
    betas_est_state_s2, _      = running_window_glm(
        corr_df_estimation_s2, ROIs_fig_S2, 'state',       2, 1, 'estimation', timeseries_model_type)
    if plotValueTimeSeries:
        betas_learning_value_s2, _ = running_window_glm(
            corr_df_learning_s2,   ROIs_fig_S2, 'state_value', 2, 1, 'learning',   timeseries_model_type)
        betas_est_value_s2, _      = running_window_glm(
            corr_df_estimation_s2, ROIs_fig_S2, 'state_value', 2, 1, 'estimation', timeseries_model_type)

# --- Figure S2 ---
if timeseries_model_type == 'decoding':
    n_rows_s2 = len(ROIs_fig_S2)
    fig = plt.figure(figsize=(6, n_rows_s2 * 1.5))
    gs = gridspec.GridSpec(n_rows_s2, 3, figure=fig,
                           width_ratios=[0.15, 1, 1], hspace=0.9, wspace=0.25)
    axes = []
    for i, roi in enumerate(ROIs_fig_S2):
        ax_roi = fig.add_subplot(gs[i, 0])
        ax_roi.axis("off")
        ax_roi.text(0.5, 0.5, roi_name_map.get(roi, roi),
                    ha='right', va='center', fontsize=10,
                    rotation=90, fontweight='bold')
        ax_learn = fig.add_subplot(gs[i, 1])
        ax_est   = fig.add_subplot(gs[i, 2])
        axes.append((ax_learn, ax_est))
        plot_running_window_styled_decoding(
            decoding_state_learning_sup[i],
            decoding_state_estimation_sup[i],
            width=5, colors=custom_colors, alpha_sig=0.05,
            bin_sizes=[24, 18], axes=[ax_learn, ax_est])

else:
    _rows_per_roi = 2 if plotValueTimeSeries else 1
    n_rows_s2 = len(ROIs_fig_S2) * _rows_per_roi
    fig = plt.figure(figsize=(6, n_rows_s2 * 1.4))
    gs = gridspec.GridSpec(n_rows_s2, 3, figure=fig,
                           width_ratios=[0.15, 1, 1], hspace=0.9, wspace=0.25)
    axes = []
    for i, roi in enumerate(ROIs_fig_S2):
        row_state = i * _rows_per_roi

        ax_roi = fig.add_subplot(gs[row_state, 0])
        ax_roi.axis("off")
        ax_roi.text(0.5, 0.5, roi_name_map.get(roi, roi),
                    ha='right', va='center', fontsize=10,
                    rotation=90, fontweight='bold')

        ax_learn_state = fig.add_subplot(gs[row_state, 1])
        ax_est_state   = fig.add_subplot(gs[row_state, 2])
        plot_running_window_styled(
            betas_learning_state_s2[betas_learning_state_s2['ROI'] == roi].copy(),
            betas_est_state_s2[betas_est_state_s2['ROI'] == roi].copy(),
            width=5, colors=custom_colors, alpha_sig=0.05,
            model_type=timeseries_model_type,
            axes=[ax_learn_state, ax_est_state])

        if plotValueTimeSeries:
            row_value = i * _rows_per_roi + 1
            ax_empty_val = fig.add_subplot(gs[row_value, 0])
            ax_empty_val.axis("off")
            ax_learn_val = fig.add_subplot(gs[row_value, 1])
            ax_est_val   = fig.add_subplot(gs[row_value, 2])
            plot_running_window_styled(
                betas_learning_value_s2[betas_learning_value_s2['ROI'] == roi].copy(),
                betas_est_value_s2[betas_est_value_s2['ROI'] == roi].copy(),
                width=5, colors=value_colors, alpha_sig=0.05,
                model_type=timeseries_model_type,
                axes=[ax_learn_val, ax_est_val],
                regressors_learning=['val_sim'],
                regressors_estimation=['val_sim'],
                ylabel='value similarity X block\nbeta')

        axes.append((ax_learn_state, ax_est_state))

# Column titles
axes[0][0].set_title("Learning trials", fontsize=10, fontweight='bold', pad=15)
axes[0][1].set_title("Estimation trials", fontsize=10, fontweight='bold', pad=15)

fig.savefig(save_dir + f"FigureS2_{timeseries_model_type}_{model_name}.pdf",
            dpi=600, transparent=True, bbox_inches="tight")
plt.show()