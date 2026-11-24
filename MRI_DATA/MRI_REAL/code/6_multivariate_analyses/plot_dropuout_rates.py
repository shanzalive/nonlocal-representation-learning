#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Signal dropout rates of the ROIs (fraction of voxels removed by the 30% dropout threshold),
# for the participants in the paper (N = 51).
#
# Input:  <sub>_dropout.npy files from a_extract_ROI_signals.py
#
# Author: Shany Grossman

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import json

# === Set your data folder here ===
# Main pipeline (30% dropout threshold, physio-regressed), as in the paper.
BASE_DIR = "/Users/grossman/Volumes/STATE_FORMATION/MRI_DATA/MRI_REAL/"
data_folder = BASE_DIR + "ROI_timeseries_LSS_5mm_smoothing_dropout_30_physio_reg_out_updated/"

import os
def _find_summary_files_behaviour():
    d = os.path.dirname(os.path.abspath(__file__))
    while not os.path.isdir(os.path.join(d, 'summary_files_behaviour')):
        if os.path.dirname(d) == d:
            raise FileNotFoundError('summary_files_behaviour/ not found above ' + __file__)
        d = os.path.dirname(d)
    return os.path.join(d, 'summary_files_behaviour') + os.sep
SUMMARY_DIR = _find_summary_files_behaviour()

with open(SUMMARY_DIR + 'learners_dict.json') as f:
    _learners_dict = json.load(f)
included_subjects = sorted(_learners_dict['full_and_partial'])

# === Load dropout .npy files for the included subjects only ===
ofc_values = []
hpc_values = []
subjects = []

for subject_id in included_subjects:
    file_path = os.path.join(data_folder, subject_id + '_dropout.npy')
    if not os.path.exists(file_path):
        print(f"WARNING: no dropout file for {subject_id}, skipping")
        continue
    array = np.load(file_path)
    ofc_values.append(array[0] * 100)  # Convert to percentage
    hpc_values.append(array[1] * 100)
    subjects.append(subject_id)

#%% === Plot ===
fig, axs = plt.subplots(2, 1, figsize=(max(10, len(subjects) * 0.5), 8), constrained_layout=True)

# Plot OFC dropout
axs[0].bar(subjects, ofc_values, color='darkblue')
axs[0].set_title("OFC dropout (%)", fontsize=18)
axs[0].set_ylabel("Dropout (%)",fontsize=12)
axs[0].tick_params(axis='x', rotation=35, labelsize=12)
axs[0].tick_params(axis='y', labelsize=18)

# Plot HPC dropout
axs[1].bar(subjects, hpc_values, color='darkblue')
axs[1].set_title("HPC dropout (%)", fontsize=18)
axs[1].set_ylabel("Dropout (%)",fontsize=12)
axs[1].tick_params(axis='x', rotation=35,labelsize=12)
axs[1].tick_params(axis='y', labelsize=18)

plt.show()
