#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Writes the learning trials of all participants in a compact format used to fit the networks.
#
# Input:  summary_files_behaviour/State_Formation_behaviour.csv
# Output: summary_files_behaviour/learning_trials_all_subjects.csv
#
# Author: Shany Grossman

import numpy as np
import pandas as pd
from stateformation_get_input_output import get_input_output_stateformation
import os

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

csvpath = SUMMARY_DIR


# Participants written to the csv: the full and partial learners (N = 51).
# The rows of each participant are computed separately (get_input_output_stateformation).
subjects = ['sub503', 'sub505', 'sub506', 'sub507', 'sub508', 'sub509',
            'sub510',
            'sub512', 'sub513', 'sub514', 'sub516', 'sub517',
            'sub518', 'sub519', 'sub520', 'sub521', 'sub522', 'sub523',
            'sub524', 'sub526', 'sub527', 'sub528', 'sub529',
            'sub530', 'sub531', 'sub532', 'sub533', 'sub534', 'sub536',
            'sub537', 'sub538', 'sub540', 'sub541', 'sub542', 'sub544',
            'sub545', 'sub546', 'sub547', 'sub548', 'sub549', 'sub552',
            'sub553', 'sub554', 'sub555', 'sub557', 'sub558', 'sub559',
            'sub560', 'sub561', 'sub563','sub564']


rows = []
    
for subject in subjects:
    
    inputs, outputs, isPassive, states, blocks, passive_type, responses, correct_responses, probed_thresholds = \
        get_input_output_stateformation(subject, csvpath = csvpath, isRecurrent=True, withPassive=False)
        
    #
    print(subject)
    l_state = [s[:2] for s in states]
    r_state = [s[2:] for s in states]
    l_state_val = [output[0] for output in outputs]
    r_state_val = [output[1] for output in outputs]
    responses = responses
    is_correct = np.array(responses==correct_responses).astype(int)
    
    n_trials = len(responses)
    
    for i in range(n_trials):
        rows.append({
            "subject": subject,
            "l_state": l_state[i],
            "r_state": r_state[i],
            "l_state_val": l_state_val[i],
            "r_state_val": r_state_val[i],
            "response": responses[i],
            "is_correct": is_correct[i],
#            "isNewSolution": isNewSolution[i]
        })


df = pd.DataFrame(rows)
df.to_csv(csvpath + "learning_trials_all_subjects.csv", index=False)