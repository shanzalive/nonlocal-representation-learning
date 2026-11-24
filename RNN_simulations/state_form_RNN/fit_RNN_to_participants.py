#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Fits the networks to one participant's curriculum 
#
# Grid search over learning rate and weight gains: for each setting, networks (10 seeds) are trained
# on the participant's curriculum, and the setting whose accuracy curve is closest to the
# participant's is kept:
#     MSE = mean_t( roll_mean(is_correct_rnn, w)[t] - roll_mean(is_correct_ppt, w)[t] )^2
# where w = ACC_SMOOTH_WINDOW (default 15). The final network is then trained with a held-out seed
# and its activity is saved for the co-update analysis.
# Usage: python fit_RNN_to_participants.py sub503
# Run for all participants with run_gridSearch_fit_NNs_acc_curve_tardis.sh.
#
# Author: Shany Grossman

import sys
import os
import gc
import itertools
import multiprocessing as mp
import numpy as np
import pandas as pd

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


os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Rolling window for smoothing accuracy curves before computing MSE
ACC_SMOOTH_WINDOW = 15

# Set to True to re-run all 10 training seeds at best params and save activations
SAVE_TRAIN_SEEDS = False


# If True and a fit already exists for this subject (fitted_params.npy present), skip the
# grid search and only (re)compute the held-out seed for the existing best_params.
REUSE_EXISTING_FIT = True

# RNG stream that generates the 10 training seeds used for the grid search and the held-out seed.
RNG_SEED = 42


def _is_valid(path):
    # True only if the file exists and is not the dummy ('none'-filled) result that fit_fun
    # returns when its subprocess crashes
    if not os.path.isfile(path):
        return False
    try:
        arr = np.load(path, allow_pickle=True)
    except Exception:
        return False
    return not (arr.dtype.kind in 'US' and np.all(arr == 'none'))



def _fit_worker(params, seed, is_final, save_dir, subj, result_queue):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    import gc
    import numpy as np

    try:
        from RNN_simulation import (RNN_simulation_v3_one_hot_coupled_weights_single_item)

        if is_final:
            storePassiveActivation                   = True
            storePassiveActivation_empiricalSampling = True
            storeActivations                         = True
            doActiveTrialManipulationMass            = False
        else:
            storePassiveActivation                   = False
            storePassiveActivation_empiricalSampling = False
            storeActivations                         = False
            doActiveTrialManipulationMass            = False

        rnn_resp, ppt_resp, choices_p = RNN_simulation_v3_one_hot_coupled_weights_single_item(
            activation_dir_name=save_dir,
            alternate_rmap=False,
            storePassiveActivation=storePassiveActivation,
            storePassiveActivation_empiricalSampling=storePassiveActivation_empiricalSampling,
            subj=subj,
            lr=params[0],
            temp=params[3],
            is_fitting_opt=True,
            input_client_type_only=False,
            doActiveTrialManipulationMass=doActiveTrialManipulationMass,
            storeActivations=storeActivations,
            same_reward_range=False,
            fc_gain=params[1],
            rnn_gain=params[2],
            seed=seed,
            add_final_passive_sweep=True)

        result_queue.put((rnn_resp, ppt_resp, choices_p))

    except Exception as e:
        print(f"[worker] Error with params {params}, seed {seed}: {e}")
        dummy_len = 264
        result_queue.put((
            np.array(['none'] * dummy_len),
            np.array(['none'] * dummy_len),
            np.full((dummy_len, 2), np.nan),
        ))
    finally:
        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
        except Exception:
            pass
        gc.collect()


def fit_fun(params, seed, is_final, save_dir, subj):
    #initialize a fresh process for each model run so TF memory is fully released on exit
    ctx = mp.get_context('spawn')
    q   = ctx.Queue()
    p   = ctx.Process(
        target=_fit_worker,
        args=(params, seed, is_final, save_dir, subj, q),
    )
    p.start()
    p.join()

    if not q.empty():
        return q.get()

    dummy_len = 264
    return (
        np.array(['none'] * dummy_len),
        np.array(['none'] * dummy_len),
        np.full((dummy_len, 2), np.nan))


def compute_acc_mse(rnn_resp, ppt_resp, correct_side_int, ppt_is_correct, window=ACC_SMOOTH_WINDOW):
    # MSE between rolling accuracy curves of RNN and participant.
    # correct_side_int : (n_trials,) int, 1 if right option is correct, 0 if left
    #                    aligned with rnn_resp / ppt_resp (includes 'none' rows)
    # ppt_is_correct   : (n_trials,) float, participant's is_correct (0/1, 'none'=0)
    #                    aligned with rnn_resp / ppt_resp
    # window           : rolling-mean half-window

    n = len(ppt_resp)
    rnn_resp = rnn_resp[:n]

    # valid = non-'none' trials (timeout rows excluded from both curves)
    valid = rnn_resp != 'none'
    if valid.sum() == 0:
        return np.inf

    correct_side = correct_side_int[:n][valid]
    rnn_int      = (rnn_resp[valid] != 'left').astype(int)
    is_correct_rnn = (rnn_int == correct_side).astype(float)
    is_correct_ppt = ppt_is_correct[:n][valid].astype(float)

    rnn_curve = (pd.Series(is_correct_rnn)
                   .rolling(window, center=False, min_periods=1).mean().values)
    ppt_curve = (pd.Series(is_correct_ppt)
                   .rolling(window, center=False, min_periods=1).mean().values)

    return float(np.mean((rnn_curve - ppt_curve) ** 2))


def save_checkpoint(checkpoint_file, results):
    tmp_file = checkpoint_file + '.tmp'
    np.save(tmp_file, results, allow_pickle=True)
    os.replace(tmp_file + '.npy', checkpoint_file)


def load_checkpoint(checkpoint_file):
    if not os.path.exists(checkpoint_file):
        return [], set()
    try:
        results     = list(np.load(checkpoint_file, allow_pickle=True))
        done_params = {tuple(np.round(r[1], 10)) for r in results}
        print(f"Resuming from checkpoint — {len(results)} combos already done.")
        return results, done_params
    except Exception as e:
        print(f"Warning: could not load checkpoint ({e}), starting fresh.")
        return [], set()


if __name__ == '__main__':

    subj = sys.argv[1]
    print(f"{subj} ...")

    save_dir = ('../state_form_RNN_outputs/NN_activations/gradientDescentPerfromanceMatched/'
                'grid_search_fits/acc_curve_mse_w15_fc_gain_rnn_gain_rl_temp_1_greedy_choice_10_seeds_9_grid_points/')
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_file = os.path.join(save_dir, f"{subj}_grid_search_checkpoint.npy")

    rng   = np.random.default_rng(seed=RNG_SEED)
    seeds = rng.integers(low=0, high=2**32, size=11)  # seeds[:10] = fit, seeds[-1] = final

    learning_rates = np.linspace(0.01, 0.15, 9)
    rnn_gains      = np.linspace(0.01, 3,    9)
    fc_gains       = np.linspace(0.01, 3,    9)
    temperatures   = np.array([1])

    # Pre-compute participant accuracy curve from the behavioral CSV 
    behav_csv = os.path.join(SUMMARY_DIR, 'learning_trials_all_subjects.csv')
    df_behav  = pd.read_csv(behav_csv)

    # CSV subject label may or may not have hyphen: normalise to match the CSV
    subj_csv = subj if subj in df_behav['subject'].values else subj.replace('sub-', 'sub')
    sub_df   = df_behav[df_behav['subject'] == subj_csv].reset_index(drop=True)
    if len(sub_df) == 0:
        raise RuntimeError(f"Subject {subj} not found in {behav_csv}")

    # Correct side: 1 if right option has higher value, 0 if left has higher value
    correct_side_int = (sub_df['r_state_val'].values > sub_df['l_state_val'].values).astype(int)
    ppt_is_correct   = sub_df['is_correct'].values.astype(float)
    # 'none' trials have is_correct=0 in the CSV; they will be filtered out in compute_acc_mse

    print(f"Loaded CSV: {len(sub_df)} trials, "
          f"ppt mean acc (valid) = {ppt_is_correct[sub_df['response'].values != 'none'].mean():.3f}")

    # Grid search 
    _fitted_params_path = os.path.join(save_dir, f"{subj}_fitted_params.npy")
    if REUSE_EXISTING_FIT and _is_valid(_fitted_params_path):
        best_params = np.load(_fitted_params_path, allow_pickle=True)
        _mse_path = os.path.join(save_dir, f"{subj}_acc_mse_at_fitted_params.npy")
        best_mse = float(np.load(_mse_path, allow_pickle=True)) if os.path.isfile(_mse_path) else np.nan
        print(f"Reusing existing fit (grid search skipped): best_params={best_params}, best_mse={best_mse:.5f}")
    else:
        results, done_params = load_checkpoint(checkpoint_file)

        all_combos = list(itertools.product(learning_rates, fc_gains, rnn_gains, temperatures))
        n_combos   = len(all_combos)

        for i_combo, (lr, fc_gain, rnn_gain, temp) in enumerate(all_combos):

            params = np.array([lr, fc_gain, rnn_gain, temp])
            key    = tuple(np.round(params, 10))

            if key in done_params:
                continue

            print(f"Combo {i_combo+1}/{n_combos} — lr={lr:.3f}, fc={fc_gain:.3f}, rnn={rnn_gain:.3f}, temp={temp:.2f}")

            mse_vals = []
            try:
                for seed in seeds[:-1]:
                    rnn_resp, ppt_resp, choices_p = fit_fun(params, seed, is_final=False,
                                                             save_dir=save_dir, subj=subj)
                    try:
                        mse = compute_acc_mse(rnn_resp, ppt_resp, correct_side_int, ppt_is_correct)
                        if np.isfinite(mse):
                            mse_vals.append(mse)
                    except Exception as e:
                        print(f"  MSE computation failed for seed {seed}: {e}")
                    finally:
                        del rnn_resp, ppt_resp, choices_p
                        gc.collect()

                if mse_vals:
                    mean_mse = float(np.mean(mse_vals))
                    std_mse  = float(np.std(mse_vals))
                    print(f"  Mean MSE: {mean_mse:.5f} ± {std_mse:.5f}, Params: {params}")
                else:
                    mean_mse = np.inf
                    std_mse  = np.nan
                    print("  All seeds failed — skipping combo.")

                results.append((mean_mse, params.copy()))
                done_params.add(key)
                save_checkpoint(checkpoint_file, results)

            except Exception as e:
                print(f"  Outer exception for params {params}: {e}")
            finally:
                gc.collect()

        # Select best params
        valid_results = [(mse, p) for mse, p in results if np.isfinite(mse)]
        if not valid_results:
            raise RuntimeError("No valid results found — all combos failed.")

        best_mse, best_params = min(valid_results, key=lambda x: x[0])
        print(f"\nBest MSE: {best_mse:.5f}, Best params: {best_params}")

    # Held-out seed: the final network, trained with a seed not used in the grid search 

    _rnn_path = os.path.join(save_dir, f"{subj}_rnn_resp_fit.npy")
    _ppt_path = os.path.join(save_dir, f"{subj}_ppt_resp_fit.npy")
    _rich_path = os.path.join(save_dir, subj, 'activations_passive_full_RNN_hidden1.npy')
    if REUSE_EXISTING_FIT and _is_valid(_rnn_path) and _is_valid(_ppt_path) and os.path.isfile(_rich_path):
        print("  Held-out seed already computed — skipping.")
    else:
        rnn_resp_fit, ppt_resp_fit, choices_p_fit = fit_fun(best_params, seeds[-1], is_final=True,
                                                             save_dir=save_dir, subj=subj)
        np.save(_rnn_path,                                                      rnn_resp_fit,  allow_pickle=True)
        np.save(_ppt_path,                                                      ppt_resp_fit,  allow_pickle=True)
        np.save(os.path.join(save_dir, f"{subj}_fitted_params.npy"),            best_params,   allow_pickle=True)
        np.save(os.path.join(save_dir, f"{subj}_acc_mse_at_fitted_params.npy"), best_mse,      allow_pickle=True)

    # Save activations for all 10 training seeds at best params
    if SAVE_TRAIN_SEEDS:
        print(f"\n--- Saving activations for all 10 training seeds at best params ---")
        base_dir         = save_dir.rstrip('/').rstrip('\\')
        train_seeds_base = base_dir + '_all_train_seeds'
        for i_seed, seed in enumerate(seeds[:-1]):
            seed_dir = os.path.join(train_seeds_base, f'seed_{i_seed:02d}') + os.sep
            os.makedirs(seed_dir, exist_ok=True)
            print(f"  Training seed {i_seed+1}/10 (seed={seed}) ...")
            rnn_r, ppt_r, _ = fit_fun(best_params, seed, is_final=True, save_dir=seed_dir, subj=subj)
            np.save(os.path.join(seed_dir, f"{subj}_rnn_resp_fit.npy"), rnn_r, allow_pickle=True)
            np.save(os.path.join(seed_dir, f"{subj}_ppt_resp_fit.npy"), ppt_r, allow_pickle=True)

    print("Done.")
