# Code for study about non-local representation learning in human Orbitofrontal cortex and Neural nets

## Contents

```
summary_files_behaviour/       behavioural data of all participants (pseudonymised)
MRI_DATA/MRI_REAL/code/        fMRI pipeline and analyses
  1a, 1c, 1d                   BIDS conversion, physiological noise regressors, behaviour
  2_mriqc, 3_fmriprep          quality control and preprocessing
  5_GLM                        single-trial GLMs and visual localizer
  6_multivariate_analyses      ROI patterns, RSA, co-update analyses and figures
RNN_simulations/
  behavioral_modelling         models of participants' choices
  state_form_RNN               recurrent networks fitted to participants' learning
```

Scripts with `Figure` in their name draw the figures.

The fMRI data and the simulation outputs are not included yet. They will be made
available, and the code will be documented in full, upon publication.

## Running the code

- Scripts that read the behavioural data find `summary_files_behaviour/` themselves, so
  they can be run from anywhere inside the repository.
- Run the network scripts from inside `RNN_simulations/state_form_RNN/`.
- `*_tardis.sh` scripts submit one job per participant on our cluster; the Python scripts
  can also be run directly.

## Requirements

Python 3.11. `requirements.txt` for the behavioural modelling and network simulations,
`requirements_fmri.txt` for the fMRI analyses. Preprocessing uses heudiconv, MRIQC,
fMRIPrep and the TAPAS PhysIO toolbox (MATLAB), which are not included.

## License

MIT, see `LICENSE`.
