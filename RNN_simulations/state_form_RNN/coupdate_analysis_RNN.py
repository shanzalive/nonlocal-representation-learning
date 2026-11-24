#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Co-update analysis of performance-matched neural networks (Fig. 7c-h, Fig. S6) and of the networks
# trained with Oja's rule (Fig. S7).
# Same analysis as in the fMRI data: per learning segment, the change in the hidden-layer pattern
# of each state, and the correlation of these changes between states (co-update).
#  - Context-dependent co-updates (same vs different context) and house-match co-updates
#    (cross-context pairs with the same vs a different house), as in Figs. 4-5.
#  - Co-update vs generalization (as in Fig. 6): each segment is classified by the network's
#    accuracy on the probe items before vs after it: generalized (accuracy goes up), not
#    generalized (goes down, or stays at 0%) or saturated (stays at 100%). The context-dependent
#    co-update is then compared between these classes (Fig. 7d, g).
# Run once with doEmpiricalSampling = True (states sampled as in the fMRI experiment, Fig. 7c-e)
# and once with doEmpiricalSampling = False (dense sampling, Fig. 7f-h); then draw the figures
# with plot_Figure7_RNN.py (Fig. 7, S6) or plot_FigureS_Oja_rule.py (Fig. S7, simulations = ['oja']).
# Run from inside RNN_simulations/state_form_RNN/ (the network fits are found via a relative path).
# Input:  network activity saved by fit_RNN_to_participants.py, summary_files_behaviour/
# Output: one results file (.npy) per simulation and sampling scheme, e.g.
#         acc_curve_mse_empirical.npy, acc_curve_mse_dense.npy
#
# Author: Shany Grossman

import numpy as np
from stateformation_get_input_output import get_input_output_stateformation, input2state
from scipy.linalg import orthogonal_procrustes
import os
import json
from run_RNN_helper_funcs import rewardfun
import warnings

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

warnings.filterwarnings('ignore', message='Mean of empty slice', category=RuntimeWarning)   # mini blocks without any probes                                                                                          
warnings.filterwarnings('ignore', message='invalid value encountered in scalar divide', category=RuntimeWarning)  # mini blocks without the relevant value levels     

simulations =  ['acc_curve_mse']
# Per simulation: [fitted to participants?, data path, n learning trials]
simulations_dict = {'acc_curve_mse': [True, '../state_form_RNN_outputs/NN_activations/gradientDescentPerfromanceMatched/grid_search_fits/acc_curve_mse_w15_fc_gain_rnn_gain_rl_temp_1_greedy_choice_10_seeds_9_grid_points/', 264],
                    'oja': [False, '../state_form_RNN_outputs/NN_activations/localLearningRules/heldOutSeed/oja/', 264]}  # networks trained with Oja's rule (Fig. S7)

# Base dir where saved activation files are 
_MAC_ROOT     = '/Users/grossman/Volumes/STATE_FORMATION'
_CLUSTER_ROOT = '/home/mpib/grossman/STATE_FORMATION'
_ROOT = _MAC_ROOT if os.path.isdir(_MAC_ROOT) else _CLUSTER_ROOT

# This script's own outputs are outside the code folder, and are not currently in the repo
# under ../state_form_RNN_outputs/NN_activations/ 

output_basedir = _ROOT + '/RNN_simulations/state_form_RNN_outputs/coupdateAnalysis/'

# Participants: full + partial learners from learners_dict.json (as in the fMRI scripts)
_ldict_path       = SUMMARY_DIR + 'learners_dict.json'
with open(_ldict_path) as _f:
    _learners_dict = json.load(_f)
included_subjects = [s.replace('-', '') for s in sorted(_learners_dict['full_and_partial'])]
print(f'Including {len(included_subjects)} subjects (full_and_partial).')

# All results use one network per participant: the held-out seed (the 11th seed, not used to fit the
# network to the participant's accuracy).
#%%
for simulation in simulations:

    csvpath = SUMMARY_DIR
   
    _sim_entry = simulations_dict[simulation]
    Shany_sim, data_dir, n_trials = _sim_entry[:3]
    # Participants' curricula and the estimation inputs (identical for every simulation)
    data_dir_shany_sim = simulations_dict['acc_curve_mse'][1]

    if Shany_sim:
        sim_included_subjects = [s for s in included_subjects
                                  if os.path.isfile(os.path.join(data_dir, s + '_fitted_params.npy'))]
        _missing_subs = sorted(set(included_subjects) - set(sim_included_subjects))
        if _missing_subs:
            print(f'[{simulation}] skipping {len(_missing_subs)} subject(s) with no fit in {data_dir}: {_missing_subs}')
    else:
        sim_included_subjects = included_subjects
    if not Shany_sim:
      # one .npz file per participant: the subject list comes from the file names (ignoring
      # macOS '._' files)
      localRule_subjects = np.array(sorted(
          f[:-4] for f in os.listdir(data_dir)
          if f.endswith('.npz') and not f.startswith('._')))
      sim_included_subjects = [s for s in included_subjects if s in set(localRule_subjects)]
      _missing_subs = sorted(set(included_subjects) - set(sim_included_subjects))
      if _missing_subs:
          print(f'[{simulation}] skipping {len(_missing_subs)} subject(s) not found in {data_dir}: {_missing_subs}')

    layers = ['RNN_hidden1','hidden2']
    reward_weights = np.array([-1,2,0,0,-1,1,-3,3,0, 0])
    # Sampling strategy: True = same miniblock positions as experiment; False = every k trials (dense)
    doEmpiricalSampling = False
    # Empirical sampling uses the experiment's own estimation trials and probe thresholds, with all
    # 4 irrelevant-feature variants of each probed state. States shown in the interleaved learning
    # trials are excluded from the co-update pairs (both sampling schemes).
    k = 1  # step size for dense sampling (doEmpiricalSampling=False only); Figure 7 and supplementary figures use k = 1

    # Include the last empirical learning segment. Needs network fits that saved a final estimation
    # sweep (index 264); False for the paper results.
    INCLUDE_LAST_EMPIRICAL_MINIBLOCK = False
    # learning-trial rows to load/build: one extra when the final miniblock is kept
    _n_sim_trials = n_trials + 1 if INCLUDE_LAST_EMPIRICAL_MINIBLOCK else n_trials

    # Reference set for the mean-centering + Procrustes alignment:
    #   'probe_set'  : estimated from the items entering the co-update analysis (nf_idx; used in the paper)
    #   'all_states' : estimated from all estimation inputs (the same for both sampling schemes)
    alignmentReferenceSet = 'probe_set'
    _align_suffix = '' if alignmentReferenceSet == 'probe_set' else '_alignAll'

    # If False, skip the Procrustes rotation entirely
    doProcrustesAlignment = True
    _proc_suffix = '' if doProcrustesAlignment else '_noProc'

    # one output folder per simulation, with the empirical and dense results
    _sim_branch = 'gradientDescentPerfromanceMatched' if Shany_sim else 'localLearningRules'
    if Shany_sim:
        _save_subdir = 'heldOutSeed'
        save_dir = output_basedir + _sim_branch + '/' + simulation + '/' + _save_subdir
    else:
        save_dir = output_basedir + _sim_branch + '/' + simulation + '/'
    _scheme_suffix = 'empirical' if doEmpiricalSampling else 'dense'
    # dense results with k != 1 get their own file name (the paper uses k = 1)
    if (not doEmpiricalSampling) and k != 1:
        _scheme_suffix += f'_denseStep{k}'


    os.makedirs(save_dir, exist_ok=True)
     

    coupdate_sample_empirical_design = [[i + offset for i in pair]
                                        for offset in (24 * k for k in range(11))
                                        for pair in ([0, 4], [20, 24])] #22x2 
    
    # without the final estimation sweep (INCLUDE_LAST_EMPIRICAL_MINIBLOCK) the last segment has
    # no 'after' measurement, so it is dropped
    if not INCLUDE_LAST_EMPIRICAL_MINIBLOCK:
        coupdate_sample_empirical_design = coupdate_sample_empirical_design[:-1]
    
    subjects = sim_included_subjects

    results = {}
    
    #%% helper func to compute learning accuracy in the Oja simulation
    def compute_learning_trials_acc_localRule_sim(estimation_inputs,y_csub, yt_csub,learning_input_left, learning_input_right):
        estimation_inputs=estimation_inputs[:,:10].copy()
        tmp = []
        correct_resp = []
        for i, input_left, input_right in zip(range(len(learning_input_left)), learning_input_left,learning_input_right):
            idx_left = np.where(np.all(estimation_inputs == input_left, axis=1))[0]
            idx_right = np.where(np.all(estimation_inputs == input_right, axis=1))[0]
            y_left = y_csub[i,idx_left]
            y_right = y_csub[i,idx_right]
            yt_left = yt_csub[i,idx_left]
            yt_right = yt_csub[i,idx_right]
            
            if y_right>y_left:
                tmp.append('righ')
            else:
                tmp.append('left')
                
            if yt_right>yt_left:
                correct_resp.append('righ')
            else:
                correct_resp.append('left')
        is_correct = (np.array(correct_resp) == np.array(tmp)).astype(int)
        return tmp, is_correct
    
    #%% Helper: Fisher-z mean for averaging correlation values
    def _fz_mean(vals):
        """Average correlations via Fisher z-transform to avoid downward bias."""
        arr = np.array(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            return np.nan
        return float(np.tanh(np.mean(np.arctanh(np.clip(arr, -0.9999, 0.9999)))))

    def _fz_mean_axis0(buf):
        """Fisher z mean across axis 0 (seeds) of a buffer array."""
        return np.tanh(np.nanmean(np.arctanh(np.clip(buf, -0.9999, 0.9999)), axis=0))

    #%% Start main analysis
    for ilayer, layer in enumerate(layers):
        # Init empty arrays to aggregate results
        n_subjects = len(subjects)
        acc_change_all                = np.full((n_subjects, n_trials), np.nan)
        acc_pre_all                   = np.full((n_subjects, n_trials), np.nan)
        context_dep_coup_all          = np.full((n_subjects, n_trials), np.nan)
        same_context_coup_all         = np.full((n_subjects, n_trials), np.nan)
        diff_context_coup_all         = np.full((n_subjects, n_trials), np.nan)
        same_irrel_context_coup_all   = np.full((n_subjects, n_trials), np.nan)
        diff_irrel_context_coup_all   = np.full((n_subjects, n_trials), np.nan)
        context_irrel_dep_coup_all    = np.full((n_subjects, n_trials), np.nan)
        cross_ctx_same_house_coup_all = np.full((n_subjects, n_trials), np.nan)
        cross_ctx_diff_house_coup_all = np.full((n_subjects, n_trials), np.nan)
        cross_ctx_house_dep_coup_all  = np.full((n_subjects, n_trials), np.nan)
        acc_learning                  = np.full((n_subjects),np.nan)
        acc_learning_single_trials    = np.full((n_subjects, n_trials),np.nan)
        # Per participant: mean co-update over the generalized / not generalized / saturated
        # learning segments
        ctx_dep_coup_gen_mean           = np.full(n_subjects, np.nan)
        ctx_dep_coup_no_gen_mean        = np.full(n_subjects, np.nan)
        ctx_dep_coup_sat_mean           = np.full(n_subjects, np.nan)
        same_context_coup_gen_mean      = np.full(n_subjects, np.nan)
        same_context_coup_no_gen_mean   = np.full(n_subjects, np.nan)
        diff_context_coup_gen_mean      = np.full(n_subjects, np.nan)
        diff_context_coup_no_gen_mean   = np.full(n_subjects, np.nan)
        # one network (the held-out seed) per participant
        _n_seeds_perseed = 1
        # per-trial arrays of the held-out seed (subject x seed x trial), so that the learning
        # segments can be classified in plot_Figure7_RNN.py (empirical sampling)
        ctx_dep_coup_perseed_all      = np.full((n_subjects, _n_seeds_perseed, n_trials), np.nan)
        same_context_coup_perseed_all = np.full((n_subjects, _n_seeds_perseed, n_trials), np.nan)
        diff_context_coup_perseed_all = np.full((n_subjects, _n_seeds_perseed, n_trials), np.nan)
        # accuracy before and after each learning segment
        acc_pre_perseed_all      = np.full((n_subjects, _n_seeds_perseed, n_trials), np.nan)
        acc_post_perseed_all     = np.full((n_subjects, _n_seeds_perseed, n_trials), np.nan)
        # house-match effect on the cross-context co-updates, per class
        cross_ctx_house_dep_gen_mean    = np.full(n_subjects, np.nan)
        cross_ctx_house_dep_no_gen_mean = np.full(n_subjects, np.nan)
        
        # Oja simulation: H_rec/H_fc/Y/X are loaded from one .npz file per participant
        # (accuracy is computed via compute_learning_trials_acc_localRule_sim).
        
        for s_idx, subject in enumerate(subjects):

            # check that the network data exist
            if Shany_sim:
                _primary = data_dir_shany_sim + subject + '/activations_passive_full_input_1.npy'
            else:
                _primary = None  # Oja simulation: .npz files are checked below
            if _primary is not None and not os.path.isfile(_primary):
                print(f'  [MISSING] {subject}: {_primary} not found — skipping.')
                continue

            print(f'Processing {subject}')

            # layer size
            if layer == 'RNN_hidden1':
                n_units = 64
            elif layer == 'hidden2':
                n_units = 32

            # load RNN simulation
            if Shany_sim:
                estimation_inputs_raw = np.load(data_dir_shany_sim + subject +  '/activations_passive_full_input_1.npy')
                estimation_inputs_raw = estimation_inputs_raw[200,:,:,:].copy()
                merged = np.empty((16, 64))
                merged[0:4] = estimation_inputs_raw[0, 0:4]
                merged[4:]  = estimation_inputs_raw[1, 4:]
                estimation_inputs = merged.T.copy()

                n, d = estimation_inputs.shape
                estimation_inputs_extended = np.zeros((n, 2, d))
                estimation_inputs_extended[:, 0, :4] = estimation_inputs[:, :4]
                estimation_inputs_extended[:, 1, 4:] = estimation_inputs[:, 4:]

                estimation_input_state = [input2state(s[[0,1,4,5,6,7,10,11,12,13]])
                                          for s in estimation_inputs]

                keep_idx = np.array([i for i, st in enumerate(estimation_input_state) if '_l' in st])
                estimation_inputs = estimation_inputs[keep_idx, :] # 32 x 16
                estimation_inputs_extended = estimation_inputs_extended[keep_idx,:,:] # 32 x 2 x 16

                # held-out seed of this participant's fitted network
                _seed_dirs_sub = [data_dir + subject + os.sep]

            else: # Oja simulation

                if subject not in set(localRule_subjects):
                    print(f'  [MISSING in Oja simulation] {subject} — skipping (row stays NaN).')
                    continue

                # held-out seed of this participant's Oja network
                _npz_sub = os.path.join(data_dir, subject + '.npz')
                x_sub = np.load(_npz_sub)['X']

                tmp = np.full((32, 16), 0)
                tmp2 = np.full((32, 2, 16), 0)
                for i_input in range(32):
                    tmp[i_input,:4] = x_sub[i_input,0,:4]
                    tmp[i_input,4:10] = x_sub[i_input,1,4:10]

                    tmp2[i_input,0,:4] = x_sub[i_input,0,:4]
                    tmp2[i_input,1, 4:10] = x_sub[i_input,1,4:10]
                estimation_inputs = tmp.copy()
                estimation_inputs_extended = tmp2.copy()

                # H_rec/H_fc/Y are loaded in the loop below
                _seed_dirs_sub = [_npz_sub]


            # buffers per network seed (here one: the held-out seed)
            _n_seeds_sub     = len(_seed_dirs_sub)
            _buf_same        = np.full((_n_seeds_sub, n_trials), np.nan)
            _buf_diff        = np.full((_n_seeds_sub, n_trials), np.nan)
            _buf_ctx_dep     = np.full((_n_seeds_sub, n_trials), np.nan)
            _buf_same_irrel  = np.full((_n_seeds_sub, n_trials), np.nan)
            _buf_diff_irrel  = np.full((_n_seeds_sub, n_trials), np.nan)
            _buf_irrel_dep   = np.full((_n_seeds_sub, n_trials), np.nan)
            _buf_cross_ctx_same_house = np.full((_n_seeds_sub, n_trials), np.nan)
            _buf_cross_ctx_diff_house = np.full((_n_seeds_sub, n_trials), np.nan)
            _buf_acc_pre      = np.full((_n_seeds_sub, n_trials), np.nan)
            _buf_acc_post     = np.full((_n_seeds_sub, n_trials), np.nan)
            _buf_acc_chg      = np.full((_n_seeds_sub, n_trials), np.nan)
            _buf_acc_learning = np.full((_n_seeds_sub, n_trials), np.nan)
            _buf_gen_flag             = np.full((_n_seeds_sub, n_trials), np.nan)  # 1=gen, 0=no_gen, 2=saturated
            _buf_coup_gen_mean        = np.full(_n_seeds_sub, np.nan)
            _buf_coup_no_gen_mean     = np.full(_n_seeds_sub, np.nan)
            _buf_coup_sat_mean        = np.full(_n_seeds_sub, np.nan)
            _buf_same_gen_mean    = np.full(_n_seeds_sub, np.nan)
            _buf_same_no_gen_mean = np.full(_n_seeds_sub, np.nan)
            _buf_diff_gen_mean    = np.full(_n_seeds_sub, np.nan)
            _buf_diff_no_gen_mean = np.full(_n_seeds_sub, np.nan)
            _buf_house_dep_gen_mean   = np.full(_n_seeds_sub, np.nan)
            _buf_house_dep_no_gen_mean= np.full(_n_seeds_sub, np.nan)

            # loop over network seeds (here one: the held-out seed)
            for _i_seed, _c_seed_dir in enumerate(_seed_dirs_sub):

                # Load h_csub / y_csub for this seed
                if Shany_sim:
                    h_csub = np.load(_c_seed_dir + 'activations_passive_full_' + layer + '.npy')
                    h_csub = h_csub.swapaxes(-1, -2)
                    h_csub = h_csub[:_n_sim_trials, 1, keep_idx, :n_units//2]
                    y_csub = np.load(_c_seed_dir + 'activations_passive_full_output.npy')
                    y_csub = y_csub[:_n_sim_trials, -1, 0, keep_idx].copy()
                    # skip a seed whose activations became numerically degenerate (NaN)
                    _nan_frac = max(np.isnan(h_csub).mean(), np.isnan(y_csub).mean())
                    if _nan_frac > 0.01:
                        print(f'  [{subject}] seed {_i_seed} ({_c_seed_dir}): {_nan_frac:.1%} '
                              f'NaN in activations -- skipping this seed (numerically degenerate)')
                        continue
                    yt_csub = np.zeros((_n_sim_trials, len(keep_idx)))
                else:
                    # Oja simulation: H_rec/H_fc/Y from the participant's .npz. Accuracy is derived
                    # via compute_learning_trials_acc_localRule_sim below.
                    _npz = np.load(_c_seed_dir)
                    h_csub = _npz['H_rec'] if layer == 'RNN_hidden1' else _npz['H_fc']
                    y_csub = _npz['Y']
                    yt_csub = np.full((_n_sim_trials, 32), np.nan)
                correct_csub = np.full((_n_sim_trials, 32), np.nan)

                # Compute true values; then probe every (trial, state) cell with a
                # random threshold drawn equally from {0.3, 0.5, 0.7}
                for i in range(_n_sim_trials):
                    for b in range(32):
                        yt_csub[i, b] = rewardfun(estimation_inputs_extended[b, :, :], 0,
                                                  reward_weights, 2, 3, 2, 16)
                total_cells = n_trials * 32
                thresholds  = np.array([0.3, 0.5, 0.7])
                thr_values  = np.resize(thresholds, total_cells)  # equal thirds, no leftover
                np.random.seed(42)
                np.random.shuffle(thr_values)
                thr_grid = thr_values.reshape(n_trials, 32)
                # Only with INCLUDE_LAST_EMPIRICAL_MINIBLOCK = True: one extra row (the final estimation
                # sweep) is drawn separately and added at the end, so the thresholds of all other trials
                # stay the same as without it.
                if _n_sim_trials > n_trials:
                    _thr_extra = np.random.RandomState(12345).choice(thresholds, size=(_n_sim_trials - n_trials, 32))
                    thr_grid = np.vstack([thr_grid, _thr_extra])
                for i in range(_n_sim_trials):
                    for b in range(32):
                        thr = thr_grid[i, b]
                        correct_csub[i, b] = int(
                            np.sign(yt_csub[i, b] - thr) == np.sign(y_csub[i, b] - thr))

                # Fill in learning inputs
                learning_inputs = np.load(data_dir_shany_sim + subject + '/activations_input_1.npy')[:264,:,:]
                _correct_dir = _c_seed_dir if Shany_sim else data_dir_shany_sim + subject + os.sep
                is_correct = np.load(_correct_dir + 'correct.npy')

                learning_input_left  = np.empty((n_trials, 10))
                learning_input_right = np.empty((n_trials, 10))

                learning_input_left[:, :4]  = learning_inputs[:, 0, :4]
                learning_input_right[:, :4] = learning_inputs[:, 0, :4]
                learning_input_left[:, 4:]  = learning_inputs[:, 1, 4:10]
                learning_input_right[:, 4:] = learning_inputs[:, 1, 10:16]

                if not Shany_sim:
                    _, is_correct = compute_learning_trials_acc_localRule_sim(estimation_inputs,y_csub, yt_csub,learning_input_left, learning_input_right)


                _buf_acc_learning[_i_seed, :] = is_correct[:n_trials]*100

                if doEmpiricalSampling:

                    # Sample at the same place as in experiment:
                    sampling_trials = coupdate_sample_empirical_design


                    # Per sampled learning segment: the probed estimation items (all 4 irrelevant-feature
                    # variants of each probed state) and the accuracy before / after the segment
                    empirical_passive_input_all_irrel_variants_idx_csub = np.zeros((len(sampling_trials), 4*4))
                    empirical_passive_acc_pre_all_irrel_variants_csub = np.zeros((len(sampling_trials)))
                    empirical_passive_acc_post_all_irrel_variants_csub = np.zeros((len(sampling_trials)))

                    inputs, outputs, isPassive, states, blocks, passive_type, responses, correct_responses, probed_thresholds = \
                        get_input_output_stateformation(subject, csvpath = csvpath, isRecurrent=True, withPassive=True)

                    passive_inputs = inputs[isPassive==1,:,:10]
                    passive_inputs_flat = np.zeros((passive_inputs.shape[0], 10))
                    passive_inputs_flat[:,:4]=passive_inputs[:,0,:4]
                    passive_inputs_flat[:,4:]=passive_inputs[:,1,4:]
                    passive_probed_thresholds = probed_thresholds[isPassive==1]
                    states_passive = [s[:-2] for s in states[isPassive==1]]
                    state_learning = states[isPassive==0]

                    # Remove passive inputs which were presented in the learning segment (1 in each late miniblock):
                    presented_idx = []
                    for ib, b in enumerate(np.sort(np.unique(blocks))[:-1]):
                        late_mini_block_idx = ib*2+1
                        tmp = state_learning[sampling_trials[late_mini_block_idx][0]:sampling_trials[late_mini_block_idx][1]]
                        states_learning_cminiblock = [s[:2] for s in tmp]

                        mask = ((blocks[isPassive==1] == b) &
                            (np.char.find(passive_type[isPassive==1].astype(str), 'late') != -1) &
                            (np.isin(states_passive, states_learning_cminiblock)))

                        presented_idx.append(np.where(mask)[0])

                    presented_idx = np.concatenate(presented_idx)
                    mask_nf = np.ones(passive_inputs.shape[0], dtype=bool)
                    mask_nf[presented_idx] = False
                    passive_inputs_flat_clean = passive_inputs_flat[mask_nf, :]
                    passive_probed_thresholds_clean = passive_probed_thresholds[mask_nf]

                    # Fill indices of passive inputs per mini block, and pre/post acc/mse:
                    for idx_sample, pair in enumerate(sampling_trials):
                        start_pre, end_pre = idx_sample*8, idx_sample*8+4
                        start_post, end_post = end_pre, end_pre+4

                        passive_inputs_csample_pre = passive_inputs_flat_clean[start_pre:end_pre,:]
                        passive_inputs_csample_post = passive_inputs_flat_clean[start_post:end_post,:]

                        # sub520 has one block fewer; skip segments outside its data
                        if pair[0] >= sum(isPassive==0):
                            continue

                        est_cropped = estimation_inputs[:, :10]
                        matches_pre = np.all(est_cropped[:, None, :] == passive_inputs_csample_pre[None, :, :],axis=2)
                        matches_post = np.all(est_cropped[:, None, :] == passive_inputs_csample_post[None, :, :],axis=2)

                        match_pre_idx = [np.where(matches_pre[:, j])[0][0]
                            for j in range(matches_pre.shape[1])
                            if np.any(matches_pre[:, j])]
                        match_pre_idx = np.array(match_pre_idx)

                        match_post_idx_real = np.array([np.where(matches_post[:, j])[0][0]
                            for j in range(matches_post.shape[1])
                            if np.any(matches_post[:, j])])
                        # The post half of an estimation phase probes the same 4 items as the pre half,
                        # in a different order: keep all arrays in pre order and align the post
                        # thresholds to their items.
                        _perm = np.array([(np.where(match_post_idx_real == _it)[0][0]
                                           if np.any(match_post_idx_real == _it) else -1)
                                          for _it in match_pre_idx])
                        _thr_post_trialorder = passive_probed_thresholds_clean[start_post:end_post].astype(float)
                        _thr_post_aligned = np.full(len(match_pre_idx), np.nan)
                        _ok = _perm >= 0
                        _thr_post_aligned[_ok] = _thr_post_trialorder[_perm[_ok]]
                        match_post_idx = match_pre_idx.copy()

                        assert(bool((np.sort(match_pre_idx)==np.sort(match_post_idx)).all()))

                        cols = [0, 1, 4, 5, 6, 7]
                        presented_passive = est_cropped[match_pre_idx,:]
                        state_matches = np.all(
                            presented_passive[:, None, cols] == est_cropped[None, :, cols],
                            axis=2)

                        presented_passive = est_cropped[match_post_idx,:]
                        state_matches_post = np.all(
                            presented_passive[:, None, cols] == est_cropped[None, :, cols],
                            axis=2)

                        match_pre_idx_all_irrel_variants = [np.where(row)[0].tolist() for row in state_matches]
                        match_pre_idx_all_irrel_variants_post = [np.where(row)[0].tolist() for row in state_matches_post]
                        empirical_passive_input_all_irrel_variants_idx_csub[idx_sample,:] = np.sort(np.array(match_pre_idx_all_irrel_variants).flatten())

                        # Accuracy before the segment: all 4 irrelevant-feature variants of each probed state
                        predict_y_csmaple_pre_all_irrel_variants = y_csub[pair[0],match_pre_idx_all_irrel_variants].flatten()
                        true_y_csub_pre_all_irrel_variants = yt_csub[pair[0],match_pre_idx_all_irrel_variants].flatten()
                        passive_probes_pre_all_irrel_variants = np.repeat(passive_probed_thresholds_clean[start_pre:end_pre]/100,4)

                        # Accuracy after the segment
                        predict_y_csmaple_post_all_irrel_variants = y_csub[pair[1],match_pre_idx_all_irrel_variants_post].flatten()
                        true_y_csub_post_all_irrel_variants = yt_csub[pair[1],match_pre_idx_all_irrel_variants_post].flatten()
                        passive_probes_post_all_irrel_variants = np.repeat(_thr_post_aligned/100,4)   # item-aligned

                        # each variant is scored separately against its state's probe threshold
                        valid = ~np.isnan(passive_probes_pre_all_irrel_variants.astype(float))
                        correct_pre_all_irrel_variants = (np.sign(predict_y_csmaple_pre_all_irrel_variants[valid] - passive_probes_pre_all_irrel_variants[valid].astype(float))==np.sign(true_y_csub_pre_all_irrel_variants[valid] - passive_probes_pre_all_irrel_variants[valid].astype(float)))
                        empirical_passive_acc_pre_all_irrel_variants_csub[idx_sample] = correct_pre_all_irrel_variants.mean()

                        valid = ~np.isnan(passive_probes_post_all_irrel_variants.astype(float))
                        correct_post_all_irrel_variants = (np.sign(predict_y_csmaple_post_all_irrel_variants[valid] - passive_probes_post_all_irrel_variants[valid].astype(float))==np.sign(true_y_csub_post_all_irrel_variants[valid] - passive_probes_post_all_irrel_variants[valid].astype(float)))
                        empirical_passive_acc_post_all_irrel_variants_csub[idx_sample] = correct_post_all_irrel_variants.mean()

                else:  # not empirical sampling
                    i = np.arange(0, n_trials - k, 1)   # one window per trial
                    sampling_trials = np.column_stack((i, i + k))

                # loop over learning segments
                for t_idx, pair in enumerate(sampling_trials):
                    start, end = pair[0], pair[1]
                    presented_input_l = learning_input_left[start:end,:]
                    presented_input_r = learning_input_right[start:end,:]
                    combined = np.vstack((presented_input_l, presented_input_r))
                    presented_inputs = np.unique(combined, axis=0)

                    est_cropped = estimation_inputs[:, :10]

                    cols = [0, 1, 4, 5, 6, 7]
                    matches = np.all(
                        est_cropped[:, None, cols] == presented_inputs[None, :, cols],
                        axis=2)
                    nf_idx = np.where(~np.any(matches, axis=1))[0]

                    if doEmpiricalSampling:
                        nf_idx = empirical_passive_input_all_irrel_variants_idx_csub[t_idx,:].astype(int)

                    R_pre  = h_csub[start,:,:]
                    R_post = h_csub[end,:,:]
                    
                    # items that define the reference frame for centering and rotation (see alignmentReferenceSet)
                    align_idx = (np.arange(R_pre.shape[0])
                                 if alignmentReferenceSet == 'all_states' else nf_idx)

                    # Remove global translation from both representations
                    mean_pre = R_pre[align_idx].mean(axis=0)
                    mean_post = R_post[align_idx].mean(axis=0)
                    R_pre_c  = R_pre  - mean_pre
                    R_post_c = R_post - mean_post


                    # Q maps pre → post frame (skipped when doProcrustesAlignment=False)
                    if doProcrustesAlignment:
                        Q, _ = orthogonal_procrustes(R_pre_c[align_idx, :], R_post_c[align_idx, :])
                        R_pre_rotated = (R_pre_c @ Q)
                    else:
                        R_pre_rotated = R_pre_c
                    
                    # Change = actual post minus pre-in-post-frame
                    change_vecs = R_post_c - R_pre_rotated


                    same_tmp, diff_tmp = [], []
                    same_irrel_tmp, diff_irrel_tmp = [], []
                    cross_ctx_same_house_tmp, cross_ctx_diff_house_tmp = [], []

                    for i, g in enumerate(nf_idx):
                        for l in nf_idx[i+1:]:
                            # only pairs of different states (a state's irrelevant-feature variants are not paired)
                            if np.array_equal(estimation_inputs[g, [0,1,4,5,6,7]], estimation_inputs[l, [0,1,4,5,6,7]]):
                                continue
                            r = np.corrcoef(change_vecs[g], change_vecs[l])[0,1]

                            # Aggregate for same/diff RELEVANT context
                            if np.array_equal(estimation_inputs[g,:2], estimation_inputs[l,:2]):
                                same_tmp.append(r)
                            else:
                                diff_tmp.append(r)

                            # Aggregate for same/diff IRRELEVANT context
                            if np.array_equal(estimation_inputs[g,2:4], estimation_inputs[l,2:4]):
                                same_irrel_tmp.append(r)
                            else:
                                diff_irrel_tmp.append(r)

                            # Cross-context pairs split by house type (estimation_inputs[g, 4:8])
                            if not np.array_equal(estimation_inputs[g, :2], estimation_inputs[l, :2]):
                                if np.array_equal(estimation_inputs[g, 4:8], estimation_inputs[l, 4:8]):
                                    cross_ctx_same_house_tmp.append(r)
                                else:
                                    cross_ctx_diff_house_tmp.append(r)

                    # Fisher-z mean for co-update correlations
                    same_mean           = _fz_mean(same_tmp)
                    diff_mean           = _fz_mean(diff_tmp)
                    same_irrel_mean     = _fz_mean(same_irrel_tmp)
                    diff_irrel_mean     = _fz_mean(diff_irrel_tmp)
                    cross_ctx_same_house_mean = _fz_mean(cross_ctx_same_house_tmp)
                    cross_ctx_diff_house_mean = _fz_mean(cross_ctx_diff_house_tmp)

                    # store the values of this segment
                    _buf_same[_i_seed, t_idx]        = same_mean
                    _buf_diff[_i_seed, t_idx]        = diff_mean
                    _buf_ctx_dep[_i_seed, t_idx]     = same_mean - diff_mean
                    _buf_same_irrel[_i_seed, t_idx]  = same_irrel_mean
                    _buf_diff_irrel[_i_seed, t_idx]  = diff_irrel_mean
                    _buf_irrel_dep[_i_seed, t_idx]   = same_irrel_mean - diff_irrel_mean
                    _buf_cross_ctx_same_house[_i_seed, t_idx] = cross_ctx_same_house_mean
                    _buf_cross_ctx_diff_house[_i_seed, t_idx] = cross_ctx_diff_house_mean

                    # Accuracy before and after the segment.
                    # Dense: every (trial, state) cell has its own pseudorandom probe threshold
                    # (thr_grid, 30/50/70, fixed seed), so pre and post are scored independently.
                    # Empirical: the experiment's own probes (all 4 irrelevant-feature variants).
                    if doEmpiricalSampling:
                        acc_pre  = empirical_passive_acc_pre_all_irrel_variants_csub[t_idx]
                        acc_post = empirical_passive_acc_post_all_irrel_variants_csub[t_idx]
                    else:
                        acc_pre  = float(np.nanmean(correct_csub[start, nf_idx].astype(bool)))
                        acc_post = float(np.nanmean(correct_csub[end, nf_idx].astype(bool)))

                    _buf_acc_pre[_i_seed, t_idx]  = acc_pre
                    _buf_acc_post[_i_seed, t_idx] = acc_post
                    _buf_acc_chg[_i_seed, t_idx]  = acc_post - acc_pre
                    # Generalization class of the segment (same rule as the fMRI analysis, Fig. 6):
                    # 1 = generalized (accuracy goes up), 0 = not generalized (goes down, or stays
                    # at 0%), 2 = saturated (stays at 100%); other segments stay unclassified (NaN)
                    _acc_chg = acc_post - acc_pre
                    if _acc_chg > 0:
                        _buf_gen_flag[_i_seed, t_idx] = 1.0
                    elif _acc_chg < 0 or (_acc_chg == 0 and acc_pre == 0):
                        _buf_gen_flag[_i_seed, t_idx] = 0.0
                    elif _acc_chg == 0 and acc_pre == 1:
                        _buf_gen_flag[_i_seed, t_idx] = 2.0

                # mean co-update over the gen / no_gen / saturated segments
                _gen_mb    = (_buf_gen_flag[_i_seed, :] == 1.0)
                _no_gen_mb = (_buf_gen_flag[_i_seed, :] == 0.0)
                _sat_mb    = (_buf_gen_flag[_i_seed, :] == 2.0)
                if _sat_mb.any():
                    _buf_coup_sat_mean[_i_seed] = np.nanmean(_buf_ctx_dep[_i_seed, _sat_mb])
                if _gen_mb.any():
                    _buf_coup_gen_mean[_i_seed]      = np.nanmean(_buf_ctx_dep[_i_seed, _gen_mb])
                    _buf_same_gen_mean[_i_seed] = np.nanmean(_buf_same[_i_seed, _gen_mb])
                    _buf_diff_gen_mean[_i_seed] = np.nanmean(_buf_diff[_i_seed, _gen_mb])
                    _buf_house_dep_gen_mean[_i_seed] = np.nanmean(
                        _buf_cross_ctx_same_house[_i_seed, _gen_mb] -
                        _buf_cross_ctx_diff_house[_i_seed, _gen_mb])
                if _no_gen_mb.any():
                    _buf_coup_no_gen_mean[_i_seed]      = np.nanmean(_buf_ctx_dep[_i_seed, _no_gen_mb])
                    _buf_same_no_gen_mean[_i_seed] = np.nanmean(_buf_same[_i_seed, _no_gen_mb])
                    _buf_diff_no_gen_mean[_i_seed] = np.nanmean(_buf_diff[_i_seed, _no_gen_mb])
                    _buf_house_dep_no_gen_mean[_i_seed] = np.nanmean(
                        _buf_cross_ctx_same_house[_i_seed, _no_gen_mb] -
                        _buf_cross_ctx_diff_house[_i_seed, _no_gen_mb])

            # per-participant values (means over the seed axis; with one seed, that seed's values)
            same_context_coup_all[s_idx, :]        = _fz_mean_axis0(_buf_same)
            diff_context_coup_all[s_idx, :]        = _fz_mean_axis0(_buf_diff)
            context_dep_coup_all[s_idx, :]         = np.nanmean(_buf_ctx_dep,    axis=0)
            same_irrel_context_coup_all[s_idx, :]  = _fz_mean_axis0(_buf_same_irrel)
            diff_irrel_context_coup_all[s_idx, :]  = _fz_mean_axis0(_buf_diff_irrel)
            context_irrel_dep_coup_all[s_idx, :]   = np.nanmean(_buf_irrel_dep,  axis=0)
            cross_ctx_same_house_coup_all[s_idx, :] = _fz_mean_axis0(_buf_cross_ctx_same_house)
            cross_ctx_diff_house_coup_all[s_idx, :] = _fz_mean_axis0(_buf_cross_ctx_diff_house)
            cross_ctx_house_dep_coup_all[s_idx, :]  = (cross_ctx_same_house_coup_all[s_idx, :] -
                                                        cross_ctx_diff_house_coup_all[s_idx, :])
            acc_pre_all[s_idx, :]                  = np.nanmean(_buf_acc_pre,    axis=0)
            acc_change_all[s_idx, :]               = np.nanmean(_buf_acc_chg,    axis=0)
            acc_learning_single_trials[s_idx, :]   = np.nanmean(_buf_acc_learning, axis=0)
            acc_learning[s_idx]                    = np.nanmean(acc_learning_single_trials[s_idx, :])
            ctx_dep_coup_gen_mean[s_idx]           = np.nanmean(_buf_coup_gen_mean)
            ctx_dep_coup_no_gen_mean[s_idx]        = np.nanmean(_buf_coup_no_gen_mean)
            same_context_coup_gen_mean[s_idx]      = np.nanmean(_buf_same_gen_mean)
            same_context_coup_no_gen_mean[s_idx]   = np.nanmean(_buf_same_no_gen_mean)
            diff_context_coup_gen_mean[s_idx]      = np.nanmean(_buf_diff_gen_mean)
            diff_context_coup_no_gen_mean[s_idx]   = np.nanmean(_buf_diff_no_gen_mean)
            # per-trial arrays of the held-out seed
            ctx_dep_coup_perseed_all[s_idx, :_buf_ctx_dep.shape[0], :] = _buf_ctx_dep
            same_context_coup_perseed_all[s_idx, :_buf_same.shape[0], :] = _buf_same
            diff_context_coup_perseed_all[s_idx, :_buf_diff.shape[0], :] = _buf_diff
            acc_pre_perseed_all[s_idx, :_buf_acc_pre.shape[0], :] = _buf_acc_pre
            acc_post_perseed_all[s_idx, :_buf_acc_post.shape[0], :] = _buf_acc_post
            # NaN when this participant had no saturated segment
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                ctx_dep_coup_sat_mean[s_idx]       = np.nanmean(_buf_coup_sat_mean)
            cross_ctx_house_dep_gen_mean[s_idx]    = np.nanmean(_buf_house_dep_gen_mean)
            cross_ctx_house_dep_no_gen_mean[s_idx] = np.nanmean(_buf_house_dep_no_gen_mean)


        # results of this layer
        results[layer] = {
        "acc_change_all": acc_change_all,
        "acc_pre_all": acc_pre_all,
        "context_dep_coup_all": context_dep_coup_all,
        "same_context_coup_all": same_context_coup_all,
        "diff_context_coup_all": diff_context_coup_all,
        "same_irrel_context_coup_all": same_irrel_context_coup_all,
        "diff_irrel_context_coup_all": diff_irrel_context_coup_all,
        "context_irrel_dep_coup_all": context_irrel_dep_coup_all,
        "cross_ctx_same_house_coup_all": cross_ctx_same_house_coup_all,
        "cross_ctx_diff_house_coup_all": cross_ctx_diff_house_coup_all,
        "cross_ctx_house_dep_coup_all":  cross_ctx_house_dep_coup_all,
        "acc_learning_single_trials":acc_learning_single_trials,
        "acc_learning": acc_learning,
        "ctx_dep_coup_gen_mean": ctx_dep_coup_gen_mean,
        "ctx_dep_coup_no_gen_mean": ctx_dep_coup_no_gen_mean,
        "ctx_dep_coup_sat_mean": ctx_dep_coup_sat_mean,
        "same_context_coup_gen_mean": same_context_coup_gen_mean,
        "same_context_coup_no_gen_mean": same_context_coup_no_gen_mean,
        "diff_context_coup_gen_mean": diff_context_coup_gen_mean,
        "diff_context_coup_no_gen_mean": diff_context_coup_no_gen_mean,
        # per-trial arrays of the held-out seed (subject x seed x trial)
        "ctx_dep_coup_perseed_all": ctx_dep_coup_perseed_all,
        "same_context_coup_perseed_all": same_context_coup_perseed_all,
        "diff_context_coup_perseed_all": diff_context_coup_perseed_all,
        "acc_pre_perseed_all": acc_pre_perseed_all,
        "acc_post_perseed_all": acc_post_perseed_all,
        "cross_ctx_house_dep_gen_mean": cross_ctx_house_dep_gen_mean,
        "cross_ctx_house_dep_no_gen_mean": cross_ctx_house_dep_no_gen_mean}

    # Save the dict
    np.save(f"/{save_dir}/{simulation}{_align_suffix}{_proc_suffix}_{_scheme_suffix}.npy", results, allow_pickle=True)
