#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Control analysis: co-updates in the estimation-only blocks following learning blocks 
# (no feedback given betwen pre and post, control for unsupervised mechanisms).
#
# Author: Shany Grossman

# Control analysis for consolidation/unsupervised mechanisms underlying the non-local representation learning effect:
#   co-updates in the post-learning passive (estimation) blocks, where no feedback is given in between
#   the PRE and POST measurements.
#   included blocks: passive2 (end of day 1) + passive3 (beginning of day 2, before the remapping)
#   (passive1 is not used: it comes before any learning)
#
# Pseudo mini-blocks, built to resemble the learning mini-blocks:
#       each run has 64 house presentations, 8 per state;
#       mini-blocks are built separately within each run (never across runs)
#       Mini-block 1:
#             PRE  = going through the run from the start, the first 4 different states,
#                    2 from each context (as in learning: always 2 + 2); a state from a context
#                    that already has 2 states is skipped
#             POST = for each of those 4 states, its next presentation with the SAME irrelevant
#                    features (client + house) as at its PRE (as in learning, where the irrelevant
#                    features of a state were the same at PRE and POST)
#       Mini-block 2:
#            start right after the last PRE trial of mini-block 1, and do the same again
#           ... and so on (mini-blocks whose 4 states do not all have a POST are dropped)
#       Notes: - mini-blocks overlap in time (the POSTs of one mini-block come after the PREs of the
#                next ones), but no trial is used in more than one change vector (checked: each
#                state x irrelevant-feature combination is shown only 2x per run = one PRE + one POST)
#              - unlike in learning, the 4 states of a mini-block do not all share the same
#                irrelevant features (only each state with itself, PRE vs POST); requiring this
#                too leaves only ~4 mini-blocks per subject
#              - the PRE-POST gap is long (median ~33 trials, ~330 s vs ~106 s in learning),
#                because each state is shown with a given set of irrelevant features only 2x per run
#
# Analysis in each mini-block (as in the main co-update analysis):
#       change vector of a state = POST pattern - PRE pattern
#       co-update of two states  = Pearson correlation between their change vectors
#
# Summary measures:
#   context-dependent co-update = mean co-update (same context) - mean co-update (different context)
#                              (computed within each mini-block, then averaged across mini-blocks)
#   value-dependent co-update   = as in Figure 5 (c_crosscontext_coupdates_Figure5.py):
#                              cross-context pairs only; in each mini-block take the mean co-update at
#                              the LARGEST and at the SMALLEST |value difference| present (needs >= 2
#                              different levels); average each over mini-blocks; value-dependent = max - min
#   group level: paired t-tests (same vs diff; max vs min) + JZS Bayes factor BF01
#
# Outputs:
#   <pref>_control_pairs.csv          one row per state pair per mini-block per ROI per subject
#   <pref>_control_subject_level.csv  one row per subject per ROI (the summary measures)
#   <pref>_control_group_stats.csv    group statistics per ROI
#   <pref>_control.pdf                simple plot of the subject-level values
#   (<pref> = the run names joined + settings, e.g. 'passive2_passive3_irrelMatched_balanced')


import os
import json
import numpy as np
import pandas as pd
from scipy.stats import zscore, ttest_1samp
from scipy.integrate import quad
import matplotlib.pyplot as plt

# ==============
# SETTINGS 
# ==============
# Folder with the LSS beta patterns + event files (same default "universe" as the main analysis):
BASE = '/Users/grossman/Volumes/'          # on tardis use: '/home/mpib/grossman/'
DATA_DIR = BASE + 'STATE_FORMATION/MRI_DATA/MRI_REAL/ROI_timeseries_LSS_5mm_smoothing_dropout_30_physio_reg_out_updated/'
RUNS = ['passive2', 'passive3']   # passive blocks to use (e.g. ['passive2'] for passive2 only)
# In the learning mini-blocks, the irrelevant features (client + house) of a state were always
# the same at PRE and POST. True = do the same here: POST is the next presentation of the state
# with the SAME irrelevant features as its PRE. False = simply the next presentation.
MATCH_IRRELEVANT_FEATURES = True
# True = 2 states from each context in every mini-block (as in the learning mini-blocks)
# False = simply the next 4 different states
BALANCE_CONTEXTS = True
TAG = ('_'.join(RUNS) + ('_irrelMatched' if MATCH_IRRELEVANT_FEATURES else '')
       + ('_balanced' if BALANCE_CONTEXTS else ''))   # used in output names
OUTPUT_DIR = DATA_DIR + 'coupdate_' + TAG + '_control/'
N_STATES_PER_MINIBLOCK = 4  # as in the learning mini-blocks
ROIS = ['OFC']
ZSCORE_PER_RUN = True       # z-score each voxel across trials (as in the main analysis)
ZSCORE_COUPDATES = True     # z-score co-update values per subject x ROI across all pairs (as in the main analysis)

# Subjects: same selection as the main fMRI figures (c_contextdep_coupdate_Figure4.py)
import os
def _find_summary_files_behaviour():
    d = os.path.dirname(os.path.abspath(__file__))
    while not os.path.isdir(os.path.join(d, 'summary_files_behaviour')):
        if os.path.dirname(d) == d:
            raise FileNotFoundError('summary_files_behaviour/ not found above ' + __file__)
        d = os.path.dirname(d)
    return os.path.join(d, 'summary_files_behaviour') + os.sep
SUMMARY_DIR = _find_summary_files_behaviour()

LEARNERS_DICT = SUMMARY_DIR + 'learners_dict.json'

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# STEP 1: build the pseudo mini-blocks from the order of states in the run
# =============================================================================
def build_miniblocks(states, variants, n_states=N_STATES_PER_MINIBLOCK):

    #states : list with the state of every trial in the run, in order
    #         (e.g. ['b2', 'b4', 'b3', 'b2', ...])
    #variants: list with the irrelevant features of every trial, e.g. (0, 1) =
    #          (irrelevant client feature, irrelevant house feature)
    #returns: a list of mini-blocks; each mini-block is a list of (state, pre_trial, post_trial)
    
    miniblocks = []
    start = 0                                   # where the current mini-block starts looking
    while True:
        # collect the first n different states from 'start' on (these are the PRE trials),
        # (if BALANCE_CONTEXTS) at most n/2 = 2 from each context
        pre = {}                                # state -> trial number of its PRE presentation
        n_per_context = {}                      # how many PRE states we have from each context
        for trial in range(start, len(states)):
            state = states[trial]
            if state in pre:
                continue                        # already have this state
            context = state[0]                  # 'a2' -> context 'a'
            if BALANCE_CONTEXTS and n_per_context.get(context, 0) == n_states // 2:
                continue                        # already 2 states from this context
            n_per_context[context] = n_per_context.get(context, 0) + 1
            pre[state] = trial
            if len(pre) == n_states:
                break
        if len(pre) < n_states:
            break                               # not enough trials left -> stop

        # POST = the next presentation of each of these states (after its PRE)
        # (if MATCH_IRRELEVANT_FEATURES) with the same irrelevant features as its PRE
        miniblock = []
        for state, pre_trial in pre.items():
            later = [t for t in range(pre_trial + 1, len(states)) if states[t] == state]
            if MATCH_IRRELEVANT_FEATURES:
                # keep only later presentations with the same irrelevant features as the PRE
                later = [t for t in later if variants[t] == variants[pre_trial]]
            if len(later) == 0:
                break                           # no suitable POST for this state -> incomplete
            miniblock.append((state, pre_trial, later[0]))

        if len(miniblock) == n_states:          # keep only complete mini-blocks
            miniblocks.append(miniblock)
        start = max(pre.values()) + 1           # next mini-block starts after the last PRE trial
    return miniblocks


# =============================================================================
# STEP 2: subject list
# =============================================================================
with open(LEARNERS_DICT) as f:
    learners_dict = json.load(f)
subjects = sorted([s for s in learners_dict['full_and_partial']
                   if os.path.exists(DATA_DIR + s + '_LSS.npz')])
print(f'{len(subjects)} subjects')


# =============================================================================
# STEP 3: co-updates for every subject, ROI, mini-block and state pair
# =============================================================================
pair_rows = []
lags = []          # number of trials between PRE and POST, for every state in every mini-block
lags_sec = []      # the same in seconds

for sub in subjects:
    print(sub)
    data = np.load(DATA_DIR + sub + '_LSS.npz')          # LSS beta patterns
    events = pd.read_csv(DATA_DIR + sub + '.csv')        # event log

    for run in RUNS:
        # house presentations of this run, in the order they were shown
        houses = events[(events.runType == run) &
                        (events.event_type == 'single house onset')].reset_index(drop=True)
        states = houses['state'].tolist()
        if len(houses) == 0:
            print(f'  {sub}: no {run} data -- skipping this run')
            continue
        variants = list(zip(houses['irrelevant_client'], houses['irrelevant_house_passive']))
        miniblocks = build_miniblocks(states, variants)

        for roi in ROIS:
            key = roi + '_' + run + '_house_passive'        # rows = house presentations (same order)
            if key not in data:
                continue
            patterns = data[key]
            if patterns.shape[0] != len(houses):
                # sub-534 passive3: the event log lists the run twice (128 events, 64 patterns) --> run wasn't completed, skip it
                print(f'  {sub} {run}: {patterns.shape[0]} patterns but {len(houses)} events -- skipping this run')
                break

            if roi == ROIS[0]:     # PRE-POST gaps of the mini-blocks used (counted once per run)
                times = houses['time'].values
                lags += [post - pre for miniblock in miniblocks for _, pre, post in miniblock]
                lags_sec += [times[post] - times[pre] for miniblock in miniblocks for _, pre, post in miniblock]

            if ZSCORE_PER_RUN:
                patterns = zscore(patterns, axis=0, nan_policy='omit')

            for i_mb, miniblock in enumerate(miniblocks):
                # change vector of each state = POST - PRE
                change = {state: patterns[post] - patterns[pre] for state, pre, post in miniblock}
                lag = {state: post - pre for state, pre, post in miniblock}   # trials between PRE and POST
                trial_of = {state: (pre, post) for state, pre, post in miniblock}  # trial numbers of PRE and POST

                # co-update for every pair of states in this mini-block
                mb_states = [state for state, _, _ in miniblock]
                for i in range(len(mb_states)):
                    for j in range(i + 1, len(mb_states)):
                        s1, s2 = mb_states[i], mb_states[j]
                        v1 = houses.loc[houses.state == s1, 'phouse_reward'].iloc[0]   # value of state 1
                        v2 = houses.loc[houses.state == s2, 'phouse_reward'].iloc[0]   # value of state 2
                        pair_rows.append(dict(
                            subject=sub, ROI=roi, run=run, miniblock=i_mb,
                            state_1=s1, state_2=s2,
                            same_context=int(s1[0] == s2[0]),      # 'a2' -> context 'a'
                            same_house_type=int(s1[1] == s2[1]),   # 'a2' and 'b2' -> same house type
                            value_1=v1, value_2=v2,
                            value_diff=abs(v1 - v2),
                            lag_1=lag[s1], lag_2=lag[s2],
                            pre_1=trial_of[s1][0], post_1=trial_of[s1][1],
                            pre_2=trial_of[s2][0], post_2=trial_of[s2][1],
                            coupdate=np.corrcoef(change[s1], change[s2])[0, 1] ))

pairs = pd.DataFrame(pair_rows)

# z-score the co-update values within each subject and ROI, across all pairs
# (same step as in c_contextdep_coupdate_Figure4.py: subtract the mean, divide by the SD)
if ZSCORE_COUPDATES:
    pairs['coupdate'] = (pairs.groupby(['subject', 'ROI'])['coupdate']
                         .transform(lambda x: (x - x.mean()) / x.std()))
pairs.to_csv(OUTPUT_DIR + TAG + '_control_pairs.csv', index=False)


# =============================================================================
# STEP 4: summary measures per subject and ROI
# =============================================================================
subject_rows = []
for (sub, roi), df_sub in pairs.groupby(['subject', 'ROI']):
    same_per_mb, diff_per_mb, max_vd_per_mb, min_vd_per_mb = [], [], [], []

    for (run, i_mb), df_mb in df_sub.groupby(['run', 'miniblock']):
        same = df_mb[df_mb.same_context == 1]
        diff = df_mb[df_mb.same_context == 0]
        # context-dependent co-update: only mini-blocks that have BOTH same- and cross-context
        # pairs (always the case with BALANCE_CONTEXTS = True, as in the learning mini-blocks)
        if len(same) > 0 and len(diff) > 0:
            same_per_mb.append(same.coupdate.mean())
            diff_per_mb.append(diff.coupdate.mean())
        # value-dependent co-update (as in Figure 5): cross-context pairs,
        # mean co-update at the largest and at the smallest |value difference| in this mini-block
        vd_pairs = diff
        level_means = vd_pairs.groupby('value_diff').coupdate.mean()   # one mean per |value diff| level
        if len(level_means) >= 2:                                      # needs >= 2 different levels
            max_vd_per_mb.append(level_means.iloc[-1])                 # largest |value difference|
            min_vd_per_mb.append(level_means.iloc[0])                  # smallest |value difference|

    corr_same = np.mean(same_per_mb) if same_per_mb else np.nan
    corr_diff = np.mean(diff_per_mb) if diff_per_mb else np.nan
    subject_rows.append(dict(
        subject=sub, ROI=roi,
        n_miniblocks=df_sub.groupby(['run', 'miniblock']).ngroups,
        n_miniblocks_for_context_dep=len(same_per_mb),   # mini-blocks with both pair types
        n_miniblocks_for_value_dep=len(max_vd_per_mb),
        corr_same_context=corr_same,
        corr_diff_context=corr_diff,
        context_dep_coupdate=corr_same - corr_diff,
        corr_max_value_diff=np.mean(max_vd_per_mb) if max_vd_per_mb else np.nan,
        corr_min_value_diff=np.mean(min_vd_per_mb) if min_vd_per_mb else np.nan,
        value_dep_coupdate=(np.mean(max_vd_per_mb) - np.mean(min_vd_per_mb)) if max_vd_per_mb else np.nan ))

subject_level = pd.DataFrame(subject_rows)
subject_level.to_csv(OUTPUT_DIR + TAG + '_control_subject_level.csv', index=False)


# =============================================================================
# STEP 5: group statistics (paired t-tests + Bayes factor)
# =============================================================================

def bayes_factor_01(t, n, r=0.707):
    """
    Bayes factor for a one-sample t-test: how much more likely the data are under
    H0 (no effect) than under H1 (some effect). BF01 > 3 = moderate evidence for no effect,
    BF01 > 10 = strong evidence for no effect.

    This is the standard "JZS" Bayes factor (Rouder et al., 2009, Psychon Bull Rev), the same one
    used by JASP and by pingouin.bayesfactor_ttest. It only needs the t-value and the sample size:
      - under H0 the effect size is exactly 0
      - under H1 the effect size can take different values; we expect small effects more than
        large ones (a Cauchy distribution with width r = 0.707, the default in JASP/pingouin)
    The Bayes factor compares how well each hypothesis predicts the observed t-value.
    """
    df = n - 1
    # how likely the observed t is if there is no effect (H0)
    likelihood_h0 = (1 + t**2 / df) ** (-(df + 1) / 2)
    # how likely the observed t is under H1: average over all possible effect sizes,
    # weighted by the prior (written as an integral over the variable g, as in Rouder et al.)
    def integrand(g):
        return ((1 + n * g * r**2) ** -0.5
                * (1 + t**2 / ((1 + n * g * r**2) * df)) ** (-(df + 1) / 2)
                * (2 * np.pi) ** -0.5 * g ** -1.5 * np.exp(-1 / (2 * g)))
    likelihood_h1 = quad(integrand, 0, np.inf)[0]
    return likelihood_h0 / likelihood_h1      # BF01 (its inverse, BF10, is the evidence FOR an effect)


# context_dep_coupdate (same - diff) and value_dep_coupdate (max - min |value diff|) are tested
# against 0, which is identical to the paired t-tests (same vs diff, max vs min) of Figures 4 and 5
measures = ['context_dep_coupdate', 'value_dep_coupdate', 'corr_same_context', 'corr_diff_context',
            'corr_max_value_diff', 'corr_min_value_diff']
stat_rows = []
for roi in ROIS:
    df_roi = subject_level[subject_level.ROI == roi]
    for measure in measures:
        x = df_roi[measure].dropna()
        if len(x) < 3:
            continue
        t, p = ttest_1samp(x, 0)
        stat_rows.append(dict(ROI=roi, measure=measure, N=len(x), mean=x.mean(),
                              sd=x.std(ddof=1), d=x.mean() / x.std(ddof=1), t=t, p=p,
                              BF01=bayes_factor_01(t, len(x))))
group_stats = pd.DataFrame(stat_rows)
group_stats.to_csv(OUTPUT_DIR + TAG + '_control_group_stats.csv', index=False)

print('\nHow many mini-blocks per subject (median):', subject_level.n_miniblocks.median())
print('Trials between PRE and POST (median, range):', np.median(lags), min(lags), '-', max(lags))
print('Seconds between PRE and POST (median, IQR): %.0f (%.0f-%.0f)'
      % (np.median(lags_sec), np.percentile(lags_sec, 25), np.percentile(lags_sec, 75)))
print('Mini-blocks with both same- and cross-context pairs (median):', subject_level.n_miniblocks_for_context_dep.median())
print('Mini-blocks usable for value-dependence (median):', subject_level.n_miniblocks_for_value_dep.median())
print('\nGroup statistics (differences tested against 0 = paired t-tests; BF01 = evidence for no effect):')
print(group_stats.round(3).to_string(index=False))


# =============================================================================
# STEP 6: simple plot, one dot per subject, black line = group mean
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
for ax, measure, label in zip(axes, ['context_dep_coupdate', 'value_dep_coupdate'],
                              ['Context-dependent co-update\n(same - different context)',
                               'Value-dependent co-update\n(max - min |value diff.|, cross-context)']):
    for i_roi, roi in enumerate(ROIS):
        x = subject_level.loc[subject_level.ROI == roi, measure].dropna()
        ax.scatter(np.full(len(x), i_roi) + np.random.uniform(-0.15, 0.15, len(x)), x,
                   s=8, color='grey', alpha=0.6)
        ax.plot([i_roi - 0.25, i_roi + 0.25], [x.mean(), x.mean()], color='k', lw=2)
    ax.axhline(0, ls='--', color='k', lw=0.8)
    ax.set_xticks(range(len(ROIS)))
    ax.set_xticklabels(ROIS, rotation=30, fontsize=7)
    ax.set_ylabel(label, fontsize=8)
    ax.set_title(' + '.join(RUNS) + ' (no feedback)', fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
fig.savefig(OUTPUT_DIR + TAG + '_control.pdf', bbox_inches='tight')
print('\nSaved results to', OUTPUT_DIR)
