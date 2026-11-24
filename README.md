# A signature of deep credit assignment in the human orbitofrontal cortex — code

Code accompanying the manuscript

> Grossman, S., Van Rossem, L., Hedrich, N., Saxe, A. & Schuck, N. W.
> *A signature of deep credit assignment in the human orbitofrontal cortex.*

This repository contains the **fMRI analyses**, the **behavioural modelling** and the
**neural network simulations**.

## Repository structure

```
summary_files_behaviour/            participant-level behavioural summary files (inputs to all analyses)
MRI_DATA/MRI_REAL/code/
  1a_bids_conversion_state_formation/   DICOM -> BIDS (heudiconv)
  1c_physio_processing/                 RETROICOR regressors (TAPAS PhysIO)
  1d_behavior/                          task logs -> behavioural csv, learner classification
  2_mriqc/  3_fmriprep/                 quality control and preprocessing
  5_GLM/                                single-trial (LSS) GLMs and visual localizer GLM
  6_multivariate_analyses/              ROI patterns, RSA-GLM, co-update analyses, figures (Figs. 1a, 3-6, S1-S4)
RNN_simulations/
  behavioral_modelling/             linear function-approximation models of choice (Fig. 2)
  state_form_RNN/                   performance-matched recurrent networks (Fig. 7, Figs. S5–S7)
```

The folder layout mirrors the project folder on our computing cluster; steps without code used
in this paper (1b: BIDS validation, 4) are not included. Every script that reads
the behavioural files finds `summary_files_behaviour/` by searching upwards from its own
location, so the scripts can be run from anywhere inside this repository.

## Data: `summary_files_behaviour/`

| File | Content |
|---|---|
| `State_Formation_behaviour.csv` | trial-by-trial task data of all participants (pseudonymised IDs); created by `1d_behavior/convert_logs_to_csv.py` |
| `learning_trials_all_subjects.csv` | learning trials only, compact format used for fitting the networks (created by `RNN_simulations/state_form_RNN/extract_behaviour_compact_csv.py`) |
| `learners_dict.json` | participants classified as full / partial / non-learners (created by `1d_behavior/plot_accuracy.py`); sub-501 (old version of the estimation task) and sub-511 (motion outlier) are already excluded here, and all analyses use the full + partial learners (N = 51) |

The fMRI data are not part of this repository.

## fMRI pipeline

All paths below are relative to `MRI_DATA/MRI_REAL/code/`.

| Step | Scripts (in order) |
|---|---|
| BIDS conversion | `1a_.../heudiconv_state_formation_tardis.sh` with the heuristic `heudiconv_state_formation_nononorm_nogre.py` (participant-specific heuristics `..._subXXX.py` for sessions with scanner exceptions); `state_formation_bids_extrainfo.sh/.py` |
| Physiological noise | `1c_physio_processing/PhysioTardis.sh` runs `Physio_updated.m` |
| Behaviour | `1d_behavior/convert_logs_to_csv.py`, then `1d_behavior/plot_accuracy.py` |
| QC, preprocessing | `2_mriqc/*.sh`, `3_fmriprep/run_fmri_prep_state_formation_tardis.sh` |
| Single-trial betas (LSS) | `5_GLM/runGLM_nilearn_Tardis.sh` runs `runGLM_nilearn.py` per participant (learning/estimation runs; visual localizer runs for the FFA/VTC ROIs) |
| Visual localizer group maps | `5_GLM/runGLM_nilearn.py` on the localizer runs in MNI space (`isNative = False`), then `5_GLM/runGLM_2ndLevel_nilearn.py` (second-level GLM; surface projection with FreeSurfer 7.4.1) |
| ROI masks and patterns | `6_.../a_extract_ROI_signals_Tardis.sh` runs `a_extract_ROI_signals.py` (dropout threshold, smoothing within ROI); `plot_ROI_masks.py` and `plot_dropuout_rates.py` for the visual inspection and dropout rates of the OFC/HPC masks |

## Figures and the scripts that produce them

| Figure | Scripts (in order) |
|---|---|
| **Fig. 1a** schematic | `6_.../plot_Figure1A.py` |
| **Fig. 2** behavioural modelling | `behavioral_modelling/run_fit_behavioral_models_tardis.sh` runs `fit_lin_func_approx_models_grid_search.py` (model defined in `behavioral_modelling.py`) per participant and model (`full_face` = perceptual, `context+house` = compressed); then `plot_Figure2_behavioral_modelling.py` |
| **Fig. 3, Fig. S1** latent state encoding (RSA-GLM) | `6_.../b_RSA_GLM_task_active_trials_all_block_pairs.py` (learning trials) and `b_RSA_GLM_task_passive_trials_all_block_pairs.py` (estimation trials); then `b_RSA_GLM_state_encoding_Figure3.py` |
| **Figs. 4–6, S2, S4** non-local co-updates in the brain | `6_.../c_coupdate_analysis_Tardis.sh` runs `c_coupdate_analysis.py` per participant; `c_2nd_level_coupdate_analysis.py` collects all participants; then `c_contextdep_coupdate_Figure4.py` (Fig. 4, Fig. S2a), `c_crosscontext_coupdates_Figure5.py` (Fig. 5, Fig. S4), `c_coupdate_vs_behavior_Figure6.py` (Fig. 6, Fig. S2b) |
| Control: estimation-only blocks | `6_.../c_coupdate_passive_control.py` |
| **Fig. S3** robustness analysis | `5_GLM/runGLM_nilearn.py` (CompCor variant, see the settings at the top) and `6_.../a_extract_ROI_signals.py` with `runMultiverse = True`; `c_coupdate_analysis.py` and `c_2nd_level_coupdate_analysis.py` with `doMultiverse = True`; then `f_plot_OFC_multiverse.py` |
| **Fig. 7a** network | `state_form_RNN/RNN_simulation.py` (network, training on each participant's curriculum; helpers `run_RNN_helper_funcs.py`, `stateformation_get_input_output.py`) |
| **Fig. 7b, Fig. S5** performance-matched networks | `state_form_RNN/run_gridSearch_fit_NNs_acc_curve_tardis.sh` runs `fit_RNN_to_participants.py` per participant (grid search over learning rate and weight gains, 10 seeds; final simulation from a held-out seed); then `plot_fitted_RNNs.py` |
| **Fig. 7c–h, Fig. S6** non-local co-updates in the networks | `state_form_RNN/coupdate_analysis_RNN.py`: run once with `doEmpiricalSampling = True` (empirical sampling, Fig. 7c–e) and once with `doEmpiricalSampling = False` (dense sampling, Fig. 7f–h); then `plot_Figure7_RNN.py` draws the figures and prints the statistics |
| **Fig. S7** networks trained with Oja's rule | `state_form_RNN/coupdate_analysis_RNN.py` with `simulations = ['oja']`, then `plot_FigureS_Oja_rule.py` |

Notes
- The networks trained with Oja's rule were simulated with code by L. Van Rossem, which is not
  included in this repository; the co-update analysis and plotting of those networks are.
- `*_tardis.sh` / `*_Tardis.sh` scripts submit one SLURM job per participant on our cluster (MPIB Tardis);
  the Python scripts can also be run directly, e.g. `python fit_RNN_to_participants.py sub503`.
- Run the network scripts from inside `RNN_simulations/state_form_RNN/`; they find the network fits via a relative path.
- Scripts contain absolute paths of our setup for data, model outputs and figures
  (`base_dir`, `save_dir`, `data_dir`, `_MAC_ROOT` / `_CLUSTER_ROOT`, `/home/mpib/...`); adjust these to your own folders.

## Requirements

Python 3.11, two environments:
- `requirements.txt`: behavioural modelling and neural network simulations (TensorFlow/Keras 2.15).
- `requirements_fmri.txt`: fMRI analyses (nilearn 0.11).

BIDS conversion, preprocessing and physiological noise modelling use heudiconv, MRIQC, fMRIPrep
and the TAPAS PhysIO toolbox (MATLAB R2021b; TAPAS release 6.0.1 = PhysIO R2022a v8.1.0,
https://github.com/translationalneuromodeling/tapas, with SPM12 version 7771). These toolboxes are
not included in this repository; download them and add them to the MATLAB path.

## License

MIT, see `LICENSE`.
