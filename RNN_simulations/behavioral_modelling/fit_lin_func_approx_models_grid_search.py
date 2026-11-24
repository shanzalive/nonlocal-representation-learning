#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Fits one behavioural model to one participant (Fig. 2).
#
# Grid search over learning rate and softmax temperature; keeps the parameters with the lowest
# negative log-likelihood of the participant's choices, then saves the fit at those parameters.
# Initial weights are drawn once per model (SD = 0.25) from a fixed random seed and
# used for both the grid search and the final fit.
#
# Flags (edit below):
#   UPDATE_RULE : 'model'       - model updates weights on its own softmax choice
#                 'participant' - model updates weights on the participant's actual choice (paper)
#
# Output folder, e.g.:
#   full_face_grid_search_1seed_participant_tempOpt_softmax_lr_grid  (Figure 2)
#
# Usage:
#     python fit_lin_func_approx_models_grid_search.py <subj> <model>
#     e.g. python fit_lin_func_approx_models_grid_search.py sub503 context+house
# Run for all participants with run_fit_behavioral_models_tardis.sh.
#
# Author: Shany Grossman

import sys
import os
import numpy as np
import behavioral_modelling

# Flags 
UPDATE_RULE   = 'participant'  # 'model' or 'participant'

subj = sys.argv[1]
m    = sys.argv[2]   # 'full', 'context', 'context+house'

save_dir = f'../behavioral_modelling_outputs/{m}_grid_search_1seed_{UPDATE_RULE}_tempOpt_softmax_lr_grid/'
os.makedirs(save_dir, exist_ok=True)
print(f'{subj}  model={m}  update_rule={UPDATE_RULE}')
print(f'Saving to: {save_dir}')

# Seed 
seed = np.random.default_rng(seed=42).integers(low=0, high=2**32)  # seed for the initial weights

# Parameter grid 
learning_rates = np.linspace(0.1, 1, 9)
temperatures   = np.linspace(1, 20, 9)


# NLL helper 
def compute_nll(choices_p, ppt_resp):
    null_idx  = np.where(ppt_resp == 1000)[0]
    ppt_clean = np.delete(ppt_resp,  null_idx)
    cp_clean  = np.delete(choices_p, null_idx, axis=0)
    eps       = 1e-8
    picked    = cp_clean[np.arange(len(ppt_clean)), ppt_clean.astype(int)]
    picked    = np.clip(picked, eps, 1)
    return -np.sum(np.log(picked))


#  Weight init helper 
_N_STATES = {'full': 32, 'context': 16, 'context+house': 8, 'full_with_face': 64, 'full_face': 64}
W_INIT_SD = 0.25   # same initialization scale for all models 

def make_w_init(seed):
    return np.random.default_rng(int(seed)).normal(0.0, W_INIT_SD, _N_STATES[m])

# Grid search 
results = []

for i_lr, lr in enumerate(learning_rates):
    print(f'  lr {i_lr+1}/{len(learning_rates)} ({lr:.4f})')
    for temp in temperatures:
        _, ppt_resp, choices_p, _, _ = behavioral_modelling.behavioral_modelling(
            m, subj=subj, lr=lr, temp=temp, w_init=make_w_init(seed),
            update_rule=UPDATE_RULE)
        choices_p = choices_p[:len(ppt_resp), :]
        nll = compute_nll(choices_p, ppt_resp)
        print(f'    temp={temp:5.1f}  NLL={nll:.3f}')
        results.append((nll, (lr, temp)))

# Select best parameters
best_nll, best_params = min(results, key=lambda x: x[0])
lr_best, temp_best = best_params
print(f'\nBest: lr={lr_best:.4f}, temp={temp_best:.2f}, mean_NLL={best_nll:.3f}')

#  Final fit at the best parameters 

model_resp_fit, ppt_resp_fit, choices_p_fit, correct_fit, output_vals_fit = behavioral_modelling.behavioral_modelling(
    m, subj=subj, lr=lr_best, temp=temp_best, w_init=make_w_init(seed),
    update_rule=UPDATE_RULE)
choices_p_fit    = choices_p_fit[:len(ppt_resp_fit), :]
output_vals_fit  = output_vals_fit[:len(ppt_resp_fit), :]
nll_fit = compute_nll(choices_p_fit, ppt_resp_fit)
print(f'NLL at fitted parameters: {nll_fit:.3f}')

# Save 
np.save(save_dir + f'{subj}_fitted_params.npy',        np.array([lr_best, temp_best]))
np.save(save_dir + f'{subj}_nll_at_fitted_params.npy', best_nll)
np.save(save_dir + f'{subj}_choices_p.npy',            choices_p_fit)
np.save(save_dir + f'{subj}_output_vals.npy',          output_vals_fit)
np.save(save_dir + f'{subj}_ppt_resp_fit.npy',         ppt_resp_fit)
np.save(save_dir + f'{subj}_model_resp_fit.npy',        model_resp_fit)
np.save(save_dir + f'{subj}_correct_response.npy',     correct_fit[:len(ppt_resp_fit)])
print(f'Saved to {save_dir}')
