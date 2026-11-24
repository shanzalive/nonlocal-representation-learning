#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Builds the ROI masks and extracts the ROI activity patterns for one participant.
#
# - ROI masks: FreeSurfer anatomical ROIs (e.g. OFC, HPC) and localizer-based
#   functional ROIs (FFA, VTC); voxels with signal dropout are removed (threshold 30%).
# - Patterns: single-trial LSS betas inside each ROI, smoothed within the ROI (5 mm).
# - Also writes a log csv with the task events of every trial.
# Input:  LSS beta maps from 5_GLM/runGLM_nilearn.py, fMRIPrep/FreeSurfer outputs
# Output: ROI_timeseries_LSS_5mm_smoothing_dropout_30_physio_reg_out_updated/
#         (<sub>_LSS.npz with the patterns, <sub>.csv with the events, ROI masks, dropout rates)
# runMultiverse = True --> instead runs the robustness variants of Fig. S3 (OFC only).
# Run: a_extract_ROI_signals_Tardis.sh (one job per participant)
#
# Author: Shany Grossman

import os
import glob
import numpy as np
import nilearn
from nilearn import image
from nilearn.interfaces.fmriprep import load_confounds
import nibabel as nib
from nilearn.masking import apply_mask
from nilearn.masking import unmask
from nilearn.image import smooth_img
import matplotlib.pyplot as plt
import pandas as pd
import sys

#%% PARMAS and FLAGS - to update manually before running
subject = sys.argv[1]
subject = subject.replace('sub5','sub-5')

# roi_mode='left_right' restricts extraction to OFC_L/OFC_R/HPC_L/HPC_R and saves
# to a separate *_LSS_left_right.npz sibling file, leaving the default *_LSS.npz
# (used by the day-1 draft analyses) completely untouched.
roi_mode = sys.argv[2] if len(sys.argv) > 2 else 'full'

data_type = 'LSS' # altenratives: 'raw','LSS'
log_only = False #if TURE, no need to save data but only dataframe with log info
allROIs = True # if true use all functional roi + ALL ANTOMICAL ROI from FRESSEURFER; if Flase: only our predefined rois

# If True: restrict extraction to OFC only and run every combination in
# MULTIVERSE_UNIVERSES (below) for this subject, each saved to its own
# output dir. The main pipeline (runMultiverse=False) is not affected; see
# run_multiverse_extraction() and the early-exit branch below.
runMultiverse = False # False = main pipeline. True = robustness variants (Fig. S3) only.

#%% PATHS
base_dir = '/home/mpib/grossman/STATE_FORMATION/MRI_DATA/MRI_REAL/'
derviatives_path = base_dir + 'derivatives/'
logs_path = base_dir + 'LOGs/'
physio_path = base_dir + 'PHYSIO_processed/'
LSS_dir = base_dir + 'nilearn_outputs/vol_maps/LSS_physio_regressed_out_updated/' #'LSS_physio_compcorr/'
if data_type== 'LSS':
    output_time_series = base_dir + 'ROI_timeseries_LSS_5mm_smoothing_dropout_30_physio_reg_out_updated/'
elif data_type== 'raw':
    output_time_series = base_dir + 'ROI_timeseries_zscore_5mm_smoothing/'
func_roi_dir = base_dir + 'nilearn_outputs/vol_maps/visloc_categories_physio_regressed_out_updated/' #here are single subject contrast maps from visloc to use for ROI definition

for ipath, path in enumerate([output_time_series]):
    if not os.path.exists(path):
        # Create the folder
        os.makedirs(path)

#%% MULTIVERSE UNIVERSES (only used when runMultiverse=True)
# Each entry: which GLM output to read LSS betas from, the smoothing kernel (fwhm), the dropout
# threshold multiplier (None = no threshold) and where to save. dir names match the UNIVERSES
# list in f_plot_OFC_multiverse.py.
MULTIVERSE_UNIVERSES = [
    # Robustness variants plotted in Fig. S3 (f_plot_OFC_multiverse.py). The main pipeline
    # (physio regression, 5 mm smoothing, dropout threshold 30%) is the runMultiverse=False path.
    # CompCor: LSS betas from runGLM_nilearn.py with add_physio_regressors=False, add_compcor=True,
    # compcor_strategy='anat_separated' (5 components each from the WM and CSF masks).
    dict(name='compcor', kind='LSS',
         LSS_dir=base_dir + 'nilearn_outputs/vol_maps/LSS_compcorr/',
         fwhm=5, dropout_th=0.3,
         output_dir=base_dir + 'ROI_timeseries_LSS_5mm_smoothing_dropout_30_compcorr/'),
    dict(name='dropout50', kind='LSS',
         LSS_dir=base_dir + 'nilearn_outputs/vol_maps/LSS_physio_regressed_out_updated/',
         fwhm=5, dropout_th=0.5,
         output_dir=base_dir + 'ROI_timeseries_LSS_5mm_smoothing_dropout_50_physio_reg_out_updated/'),
    dict(name='dropout80', kind='LSS',
         LSS_dir=base_dir + 'nilearn_outputs/vol_maps/LSS_physio_regressed_out_updated/',
         fwhm=5, dropout_th=0.8,
         output_dir=base_dir + 'ROI_timeseries_LSS_5mm_smoothing_dropout_80_physio_reg_out_updated/'),
    dict(name='no_dropout', kind='LSS',
         LSS_dir=base_dir + 'nilearn_outputs/vol_maps/LSS_physio_regressed_out_updated/',
         fwhm=5, dropout_th=None,
         output_dir=base_dir + 'ROI_timeseries_LSS_5mm_smoothing_physio_reg_out_updated/'),
    dict(name='smoothing_3mm', kind='LSS',
         LSS_dir=base_dir + 'nilearn_outputs/vol_maps/LSS_physio_regressed_out_updated/',
         fwhm=3, dropout_th=0.3,
         output_dir=base_dir + 'ROI_timeseries_LSS_3mm_smoothing_dropout_30_physio_reg_out_updated/'),
]

#%% ROIs
roi_dict = {
    "OFC":[2014,1014,2012,1012,2019,1019],
    "HPC": [17,53],
    "OFC_L": [1014,1012,1019],
    "OFC_R": [2014,2012,2019],
    "HPC_L": [17],
    "HPC_R": [53],
    "3&4VENTRICLES": [4,14],
    "FFA":                  [1009,2009,1007,2007,1011,2011,1016,2016,1013,2013],
    "PPA":                  [1009,2009,1007,2007,1011,2011,1016,2016,1013,2013],
    "LO":                   [1009,2009,1007,2007,1011,2011,1016,2016,1013,2013],
    "VISUALFUNC_HIGHORDER": [1009,2009,1007,2007,1011,2011,1016,2016,1013,2013],          
    "bankssts": [1001,2001],
    "caudalanteriorcingulate": [1002,2002],
    "caudalmiddlefrontal": [1003,2003],
    "corpuscallosum": [1004,2004],
    "cuneus": [1005,2005],
    "entorhinal": [1006,2006],
    "fusiform": [1007,2007],
    "inferiorparietal": [1008,2008],
    "inferiortemporal": [1009,2009],
    "isthmuscingulate": [1010,2010],
    "lateraloccipital": [1011,2011],
    "lateralorbitofrontal": [1012,2012],
    "lingual": [1013,2013],
    "medialorbitofrontal": [1014,2014],
    "middletemporal": [1015,2015],
    "parahippocampal": [1016,2016],
    "parsopercularis": [1018,2018],
    "parsorbitalis": [1019,2019],
    "parstriangularis": [1020,2020],
    "pericalcarine": [1021,2021],
    "postcentral": [1022,2022],
    "posteriorcingulate": [1023,2023],
    "precentral":[1024,2024],
    "precuneus": [1025,2025],
    "rostralanteriorcingulate": [1026,2026],
    "rostralmiddlefrontal": [1027,2027],
    "superiorfrontal": [1028,2028],
    "superiorparietal": [1029,2029],
    "superiortemporal": [1030,2030],
    "supramarginal": [1031,2031],
    "frontalpole": [1032,2032],
    "temporalpole": [1033,2033],
    "transversetemporal": [1034,2034],
    "insula": [1035,2035]}

func_roi_contrast_dict = { 
            "FFA":              'faces>scenes',
            "PPA":              'faces>scenes',
            "LO":               'objects>scrambled_objects',
            "VISUALFUNC":       'all_visual',
            "VISUALFUNC_HIGHORDER": 'all_cat>scrambled_objects'}
if roi_mode == 'left_right':
    ROIs = ['OFC_L', 'OFC_R', 'HPC_L', 'HPC_R']
elif allROIs:
    ROIs = list(roi_dict.keys())
else:
    ROIs = ['FFA', 'OFC' 'HPC', 'VISUALFUNC_HIGHORDER']

TR=1.75 #in sec

#%% DEFINE SPECIAL CASES (e.g. run to exclude)
runs2exclude = [
    'sub-509_ses-3_task-sf_run-03',  # active14
    '_ses-3_task-sf_run-08',
    '_ses-3_task-sf_run-09'] #'sub-509_ses3_run3_active14', #I accidently entered sub number  510 for this run, run4/active 15 will in fact be the first block with the mapping change]

#%% DEFINE RUN TYPES DICT
def sesrun2runType(sub,ses,run):
    s = "ses"+ses+"-run"+run
    conversiondict = {
      "ses1-run1":["passive","1"],
      "ses1-run2":["active","1"],
      "ses1-run3":["active","2"],
      "ses1-run4":["active","3"],
      "ses1-run5":["active","4"],
      "ses1-run6":["active","5"],
      "ses1-run7":["active","6"],
      "ses2-run1":["active","7"],
      "ses2-run2":["active","8"],
      "ses2-run3":["active","9"],
      "ses2-run4":["active","10"],
      "ses2-run5":["active","11"],
      "ses2-run6":["passive","2"],
      "ses3-run1":["passive","3"],
      "ses3-run2":["active","13"],
      "ses3-run3":["active","14"],
      "ses3-run4":["active","15"],
      "ses3-run5":["active","16"],
      "ses3-run6":["active","17"],
      "ses3-run7":["passive","4"],
      "ses3-run8":["visloc_categories","none"],
      "ses3-run9":["visloc_features","none"],
    }
        
    return conversiondict[s]

#%% DEFINE STATE DICTS 
def convert_input2single_house_state(client_img,phouse_img):
    c_f1 = client_img.replace('-1','0')[7]
    h_f1, h_f2 =  phouse_img.replace('-1','0')[6],  phouse_img.replace('-1','0')[8]
    s=[]
    for i,f in enumerate([c_f1, h_f1, h_f2]):
        if f=='0':
            s.append(0)
            s.append(1)
        elif f=='1':
            s.append(1)
            s.append(0)
    s=str(s)
    s=s.replace(' ', '')
    
    passive_trial_type_dict ={'[1,0,0,1,0,1]':'a1',
                              '[1,0,0,1,1,0]':'a2',
                              '[1,0,1,0,0,1]':'a3',
                              '[1,0,1,0,1,0]':'a4',
                              '[0,1,0,1,0,1]':'b1',
                              '[0,1,0,1,1,0]':'b2',
                              '[0,1,1,0,0,1]':'b3',
                              '[0,1,1,0,1,0]':'b4'}
    return passive_trial_type_dict.get(s)

def convert_input2two_house_state(client_img,lhouse_img,rhouse_img):
    c_f1 = client_img.replace('-1','0')[7]
    lh_f1, lh_f2 = lhouse_img.replace('-1','0')[6],   lhouse_img.replace('-1','0')[8]
    rh_f1, rh_f2 = rhouse_img.replace('-1','0')[6], rhouse_img.replace('-1','0')[8]
    s=[]
    for i,f in enumerate([c_f1, lh_f1, lh_f2, rh_f1, rh_f2]):
        if f=='0':
            s.append(0)
            s.append(1)
        elif f=='1':
            s.append(1)
            s.append(0)
    s=str(s) 
    s=s.replace(' ', '')
    
    active_trial_type_dict = {'[1,0,0,1,0,1,0,1,1,0]':'a1a2',
                              '[1,0,0,1,1,0,0,1,0,1]':'a2a1',
                              '[1,0,0,1,0,1,1,0,0,1]':'a1a3',
                              '[1,0,1,0,0,1,0,1,0,1]':'a3a1',
                              '[1,0,0,1,0,1,1,0,1,0]':'a1a4',
                              '[1,0,1,0,1,0,0,1,0,1]':'a4a1',
                              '[1,0,0,1,1,0,1,0,0,1]':'a2a3',
                              '[1,0,1,0,0,1,0,1,1,0]':'a3a2',
                              '[1,0,0,1,1,0,1,0,1,0]':'a2a4',
                              '[1,0,1,0,1,0,0,1,1,0]':'a4a2',
                              '[1,0,1,0,0,1,1,0,1,0]':'a3a4',
                              '[1,0,1,0,1,0,1,0,0,1]':'a4a3',
                              '[0,1,0,1,0,1,0,1,1,0]':'b1b2',
                              '[0,1,0,1,1,0,0,1,0,1]':'b2b1',
                              '[0,1,0,1,0,1,1,0,0,1]':'b1b3',
                              '[0,1,1,0,0,1,0,1,0,1]':'b3b1',
                              '[0,1,0,1,0,1,1,0,1,0]':'b1b4',
                              '[0,1,1,0,1,0,0,1,0,1]':'b4b1',
                              '[0,1,0,1,1,0,1,0,0,1]':'b2b3',
                              '[0,1,1,0,0,1,0,1,1,0]':'b3b2',
                              '[0,1,0,1,1,0,1,0,1,0]':'b2b4',
                              '[0,1,1,0,1,0,0,1,1,0]':'b4b2',
                              '[0,1,1,0,0,1,1,0,1,0]':'b3b4',
                              '[0,1,1,0,1,0,1,0,0,1]':'b4b3'}
    return active_trial_type_dict.get(s)


#%% Function which masks beta maps and patials smooths them:
def mask_and_smooth_data(data,ROI_mask,fwhm):
    
    # Load data into a nifti fomrat
    affine = np.eye(4)  # 
    beta_img = nib.Nifti1Image(data, affine)
    mask_img = nib.Nifti1Image(ROI_mask.astype(np.uint8), affine)
    
    # Apply mask 
    masked_beta = apply_mask(beta_img, mask_img)  # shape: (n_trials, n_voxels)
    
    # Convert masked data back to 4d image and apply saptial smoothing
    unmasked_img = unmask(masked_beta, mask_img)  # shape: (75, 97, 78, n_trials)
    smoothed_img = smooth_img(unmasked_img, fwhm=fwhm)  # FWHM in mm
    
    # Convet back to 2D
    masked_smoothed_data = apply_mask(smoothed_img, mask_img)  # shape: (n_trials, n_voxels)
    
    return masked_smoothed_data

def droupout_brain_mask(func_dirs): #,drop_out_th=0.8):


    # loading the brain mask of the second run as an initial template
    brain_mask_cur_run_img =  nilearn.image.load_img(func_dirs[0].replace('_space-T1w_desc-preproc_bold.nii.gz', '_space-T1w_desc-brain_mask.nii.gz'),
                                                     wildcards=True, 
                                                     dtype=None)
    
    brain_mask = image.get_data(brain_mask_cur_run_img)
    
    #loop over all runs and exclude voxels that are outside of the mask in at least one of the runs (final mask is the intersection of all brain masks)
    vox_mean = np.zeros(len(func_dirs))
    vox_mean[:]=np.nan
    for i_func, func in enumerate(func_dirs):

            brain_mask_cur_run_img =  nilearn.image.load_img(func.replace('_space-T1w_desc-preproc_bold.nii.gz', '_space-T1w_desc-brain_mask.nii.gz'), 
                                                             wildcards=True, 
                                                             dtype=None)
            
            brain_mask_cur_run = image.get_data(brain_mask_cur_run_img)
            
            #load actual 4D run to detect dropouts:
            cur_run_raw_4D_img = nilearn.image.load_img(func, 
                                                        wildcards=True,
                                                        dtype=None)
           
            #remove voxels with drop out signal
            crun_global_mean = np.mean(nilearn.masking.apply_mask(cur_run_raw_4D_img,nib.Nifti1Image(brain_mask_cur_run,brain_mask_cur_run_img.affine), 
                                                             dtype='f', 
                                                             smoothing_fwhm=None, 
                                                             ensure_finite=True)) #output is np array time points X ROI voxels
            
            vox_mean[i_func] = crun_global_mean
            brain_mask = brain_mask & brain_mask_cur_run #& valid_vox
            
    global_mean = np.nanmean(vox_mean)     
    return(brain_mask, global_mean)

#%% MULTIVERSE EXTRACTION (OFC only). Self-contained, does not share state
# with the code below it. Builds the events log and OFC anatomical mask once
# (both are identical across universes), then for each universe in
# MULTIVERSE_UNIVERSES re-applies that universe's dropout threshold and
# either re-loads/smooths the relevant LSS beta maps, or re-cleans the raw
# BOLD (with physio regressors) and smooths, saving each universe to its own
# output_dir in the same {subject}_LSS.npz / {subject}.npz + {subject}.csv +
# {subject}_dropout.npy layout the rest of the pipeline already expects.
def build_events_log(subject, func_dirs):
    #   Ports the event-parsing loop (identical for every universe, purely a
    #   function of the psychopy logs and TR timing, independent of ROI/smoothing/
    #   physio/dropout choices) so it only has to run once per subject."""
    columns = ['subject', 'ses', 'run', 'runType',
               'time', 'closest_tr', 'closest_tr_2', 'event_type',
               'trial_type', 'state', 'irrelevant_client', 'irrlevant_house_left',
               'irrelevant_house_right', 'irrelevant_house_passive', 'primary_house_passive', 'secondary_house_passive',
               'reward', 'lhouse_reward', 'rhouse_reward', 'phouse_reward',
               'correct_resp', 'resp', 'is_correct', 'is_probed_passive', 'probe_th',
               'client_img', 'phouse_img', 'lhouse_img', 'rhouse_img']
    mri_log_df = pd.DataFrame(columns=columns)
    runTypes = []

    for func in func_dirs:
        ses = func[func.find('ses') + 4:func.find('ses') + 5]
        run = func[func.find('run') + 5:func.find('run') + 6]
        runType, nRunType = sesrun2runType(subject, ses, run)
        if 'visloc' in runType:
            continue

        func_img = nilearn.image.load_img(func, wildcards=True, dtype=None)
        func_4D = image.get_data(func_img)
        n_voxels = func_4D.shape[0] * func_4D.shape[1] * func_4D.shape[2]
        func_2D = func_4D.reshape(n_voxels, -1).T
        n_samples = func_2D.shape[0]
        tr_times = np.arange(0, n_samples * TR, TR)

        runTypes.append(runType + nRunType)

        log = pd.read_csv(logs_path + subject.replace('sub-5', 'sub5') + '/' + subject[-3:] + '_' + ses + '_' + run + '_' + runType + nRunType + '_log.csv')
        for i, time, event_type, is_passive in zip(range(log.shape[0]), np.array(log.time), np.array(log.eventType), np.array(log.is_passive)):

            distances = np.abs(tr_times - (time + 5))
            closest_2_trs = distances.argsort()[:2]
            closest_tr = closest_2_trs[0]
            closest_tr_2 = closest_2_trs[1]

            if is_passive:
                if subject == 'sub-501' or subject == 'sub-502':
                    lhouse_img, rhouse_img, reward, lhouse_reward, rhouse_reward, probe_th = 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'
                    if event_type == 'client onset':
                        client_img, phouse_img, phouse_reward = log.iloc[i].client_img, log.iloc[i + 1].passive_house_img, log.iloc[i + 1].passive_house_reward
                        is_probed_passive = 1
                        probe_th = 50
                        correct_resp = 'less' if probe_th > phouse_reward else 'more'
                        resp = log.iloc[i + 2].response
                        is_correct = 1 if correct_resp == resp else 0
                    elif event_type == 'single house onset':
                        client_img, phouse_img, phouse_reward = log.iloc[i - 1].client_img, log.iloc[i].passive_house_img, log.iloc[i].passive_house_reward
                        is_probed_passive = 1
                        probe_th = 50
                        correct_resp = 'less' if probe_th > phouse_reward else 'more'
                        resp = log.iloc[i + 1].response
                        is_correct = 1 if correct_resp == resp else 0
                    elif event_type == 'response':
                        client_img, phouse_img, phouse_reward = log.iloc[i - 2].client_img, log.iloc[i - 1].passive_house_img, log.iloc[i - 1].passive_house_reward
                        is_probed_passive = 1
                        probe_th = 50
                        correct_resp = 'less' if probe_th > phouse_reward else 'more'
                        resp = log.iloc[i].response
                        is_correct = 1 if correct_resp == resp else 0
                    else:
                        client_img, phouse_img, phouse_reward, is_probed_passive, state = 'NA', 'NA', 'NA', 'NA', 'NA'
                        correct_resp, resp, is_correct, probe_th = 'NA', 'NA', 'NA', 'NA'
                else:
                    lhouse_img, rhouse_img, reward, lhouse_reward, rhouse_reward, probe_th = 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'
                    if event_type == 'client onset':
                        client_img, phouse_img, phouse_reward = log.iloc[i].client_img, log.iloc[i + 1].passive_house_img, log.iloc[i + 1].passive_house_reward
                        if (log.shape[0] < i + 2) and (log.iloc[i + 2].eventType == 'passive probe'):
                            is_probed_passive = 1
                        else:
                            is_probed_passive = 0
                        if is_probed_passive:
                            probe_th = log.iloc[i + 2].probe_th
                            correct_resp = 'less' if log.iloc[i + 2].probe_th > phouse_reward else 'more'
                            resp = log.iloc[i + 3].response
                            is_correct = 1 if correct_resp == resp else 0
                        else:
                            correct_resp, resp, is_correct = 'NA', 'NA', 'NA'
                    elif event_type == 'single house onset':
                        client_img, phouse_img, phouse_reward = log.iloc[i - 1].client_img, log.iloc[i].passive_house_img, log.iloc[i].passive_house_reward
                        if (log.shape[0] < i + 1) and (log.iloc[i + 1].eventType == 'passive probe'):
                            is_probed_passive = 1
                        else:
                            is_probed_passive = 0
                        if is_probed_passive:
                            probe_th = log.iloc[i + 1].probe_th
                            correct_resp = 'less' if log.iloc[i + 1].probe_th > phouse_reward else 'more'
                            resp = log.iloc[i + 2].response
                            is_correct = 1 if correct_resp == resp else 0
                        else:
                            correct_resp, resp, is_correct = 'NA', 'NA', 'NA'
                    elif event_type == 'response':
                        client_img, phouse_img, phouse_reward = log.iloc[i - 3].client_img, log.iloc[i - 2].passive_house_img, log.iloc[i - 2].passive_house_reward
                        is_probed_passive = 1
                        probe_th = log.iloc[i - 1].probe_th
                        correct_resp = 'less' if log.iloc[i - 1].probe_th > phouse_reward else 'more'
                        resp = log.iloc[i].response
                        is_correct = 1 if correct_resp == resp else 0
                    else:
                        client_img, phouse_img, phouse_reward, is_probed_passive, state = 'NA', 'NA', 'NA', 'NA', 'NA'
                        correct_resp, resp, is_correct, probe_th = 'NA', 'NA', 'NA', 'NA'
            else:
                phouse_img, phouse_reward, is_probed_passive, probe_th = 'NA', 'NA', 'NA', 'NA'
                if event_type == 'client onset':
                    client_img, lhouse_img, rhouse_img = log.iloc[i].client_img, log.iloc[i + 1].lhouse_img, log.iloc[i + 1].rhouse_img
                    reward, lhouse_reward, rhouse_reward = log.iloc[i + 3].reward, log.iloc[i + 1].lhouse_reward, log.iloc[i + 1].rhouse_reward
                    correct_resp, resp = log.iloc[i + 1].correctResp, log.iloc[i + 2].response
                    is_correct = 1 if correct_resp == resp else 0
                elif event_type == 'house onset':
                    client_img, lhouse_img, rhouse_img = log.iloc[i - 1].client_img, log.iloc[i].lhouse_img, log.iloc[i].rhouse_img
                    reward, lhouse_reward, rhouse_reward = log.iloc[i + 2].reward, log.iloc[i].lhouse_reward, log.iloc[i].rhouse_reward
                    correct_resp, resp = log.iloc[i].correctResp, log.iloc[i + 1].response
                    is_correct = 1 if correct_resp == resp else 0
                elif event_type == 'response':
                    client_img, lhouse_img, rhouse_img = log.iloc[i - 2].client_img, log.iloc[i - 1].lhouse_img, log.iloc[i - 1].rhouse_img
                    reward, lhouse_reward, rhouse_reward = log.iloc[i + 1].reward, log.iloc[i - 1].lhouse_reward, log.iloc[i - 1].rhouse_reward
                    correct_resp, resp = log.iloc[i - 1].correctResp, log.iloc[i].response
                    is_correct = 1 if correct_resp == resp else 0
                elif event_type == 'reward':
                    client_img, lhouse_img, rhouse_img = log.iloc[i - 3].client_img, log.iloc[i - 2].lhouse_img, log.iloc[i - 2].rhouse_img
                    reward, lhouse_reward, rhouse_reward = log.iloc[i].reward, log.iloc[i - 2].lhouse_reward, log.iloc[i - 2].rhouse_reward
                    correct_resp, resp = log.iloc[i - 2].correctResp, log.iloc[i - 1].response
                    is_correct = 1 if correct_resp == resp else 0
                else:
                    client_img, lhouse_img, rhouse_img, state = 'NA', 'NA', 'NA', 'NA'
                    reward, lhouse_reward, rhouse_reward = 'NA', 'NA', 'NA'
                    correct_resp, resp, is_correct = 'NA', 'NA', 'NA'

            if event_type == 'passive probe':
                client_img, phouse_img, phouse_reward = log.iloc[i - 2].client_img, log.iloc[i - 1].passive_house_img, log.iloc[i - 1].passive_house_reward
                is_probed_passive = 1
                is_passive = 1
                probe_th = log.iloc[i].probe_th
                correct_resp = 'less' if log.iloc[i].probe_th > phouse_reward else 'more'
                resp = log.iloc[i + 1].response
                is_correct = 1 if correct_resp == resp else 0

            if 'prompt' not in event_type:
                trial_type = 'passive' if is_passive == 1 else 'active'
                state = convert_input2single_house_state(client_img, phouse_img) if is_passive == 1 else convert_input2two_house_state(client_img, lhouse_img, rhouse_img)
            else:
                trial_type = 'NA'

            irrelevant_client = 'NA' if client_img == 'NA' else client_img.replace('-1', '0')[9]
            irrlevant_house_left = 'NA' if lhouse_img == 'NA' else lhouse_img.replace('-1', '0')[10]
            irrelevant_house_right = 'NA' if rhouse_img == 'NA' else rhouse_img.replace('-1', '0')[10]
            irrelevant_house_passive = 'NA' if phouse_img == 'NA' else phouse_img.replace('-1', '0')[10]
            primary_house_passive = 'NA' if phouse_img == 'NA' else phouse_img.replace('-1', '0')[8]
            secondary_house_passive = 'NA' if phouse_img == 'NA' else phouse_img.replace('-1', '0')[6]

            new_row = pd.DataFrame([[subject, ses, run, runType + nRunType,
                                     time, closest_tr, closest_tr_2, event_type,
                                     trial_type, state, irrelevant_client, irrlevant_house_left,
                                     irrelevant_house_right, irrelevant_house_passive, primary_house_passive, secondary_house_passive,
                                     reward, lhouse_reward, rhouse_reward, phouse_reward,
                                     correct_resp, resp, is_correct, is_probed_passive, probe_th,
                                     client_img, phouse_img, lhouse_img, rhouse_img]],
                                   columns=mri_log_df.columns)
            mri_log_df = pd.concat([mri_log_df, new_row], ignore_index=True)

    return mri_log_df, runTypes


def build_roi_mask(roi_name, dropout_th, fs_aparcaseg, brain_mask, global_mean, func_dirs):
    """Anatomical + functional (dropout-thresholded) mask for the given ROI. dropout_th=None
    skips the dropout threshold entirely."""
    idxs = roi_dict[roi_name]
    anat_mask = np.isin(fs_aparcaseg[0], idxs) & brain_mask
    n_vox_anat = anat_mask.sum()

    mask = anat_mask.copy()
    for ifunc in range(len(fs_aparcaseg)):
        mask = mask & np.isin(fs_aparcaseg[ifunc], idxs)
        if dropout_th is not None:
            func_img = nilearn.image.load_img(func_dirs[ifunc], wildcards=True, dtype=None)
            func_4D = image.get_data(func_img)
            func_3D = np.mean(func_4D, axis=3)
            mask = mask & (func_3D > dropout_th * global_mean)

    mask = np.array(brain_mask & mask)
    dropout_ratio = 1 - (mask.sum() / n_vox_anat) if n_vox_anat > 0 else np.nan
    return mask, dropout_ratio


def build_ofc_mask(dropout_th, fs_aparcaseg, brain_mask, global_mean, func_dirs):
    """OFC anatomical + functional (dropout-thresholded) mask. dropout_th=None
    skips the dropout threshold entirely."""
    return build_roi_mask('OFC', dropout_th, fs_aparcaseg, brain_mask, global_mean, func_dirs)


# Extra ROIs to extract alongside OFC for the raw_bold universe only (RSA sanity-check
# request, not part of the OFC co-update multiverse itself). VISUALFUNC_HIGHORDER is VTC
# (matches the anatomical label set used for the FFA/VTC group maps in runGLM_2ndLevel_nilearn.py).
RAW_EXTRA_ROIS = ['HPC', 'FFA', 'VISUALFUNC_HIGHORDER']


def run_multiverse_extraction(subject, func_dirs, fs_aparcaseg, brain_mask, global_mean, mri_log_df, runTypes):
    for universe in MULTIVERSE_UNIVERSES:
        print(f"sub-{subject}: multiverse universe '{universe['name']}'")
        u_output_dir = universe['output_dir']
        if not os.path.exists(u_output_dir):
            os.makedirs(u_output_dir)

        mask, dropout_ratio = build_ofc_mask(universe['dropout_th'], fs_aparcaseg, brain_mask, global_mean, func_dirs)
        total_voxels = np.sum(mask.astype(np.int64))
        if total_voxels == 0:
            print(f"sub-{subject} universe {universe['name']}: no voxels found in OFC")
            continue

        # Raw universe only: also build masks for HPC/FFA/VTC so the raw-BOLD RSA sanity
        # check has these ROIs too. Every other universe here is OFC-multiverse-specific.
        extra_roi_masks = {}
        if universe['kind'] == 'raw':
            for extra_roi in RAW_EXTRA_ROIS:
                extra_mask, _ = build_roi_mask(extra_roi, universe['dropout_th'], fs_aparcaseg, brain_mask, global_mean, func_dirs)
                if np.sum(extra_mask.astype(np.int64)) == 0:
                    print(f"sub-{subject} universe {universe['name']}: no voxels found in {extra_roi}")
                    continue
                extra_roi_masks[extra_roi] = extra_mask

        u_ROIs_time_series, u_run_names = [], []
        for ifunc, func in enumerate(func_dirs):
            ses = func[func.find('ses') + 4:func.find('ses') + 5]
            run = func[func.find('run') + 5:func.find('run') + 6]
            runType, nRunType = sesrun2runType(subject, ses, run)
            if 'visloc' in runType:
                continue

            if universe['kind'] == 'LSS':
                lss_path = universe['LSS_dir'] + subject + '_' + runType + nRunType + '_LSS.npz'
                if not os.path.exists(lss_path):
                    continue
                LSS_dict = np.load(lss_path, allow_pickle=True)
                for key in LSS_dict.keys():
                    masked_smoothed_data = mask_and_smooth_data(LSS_dict[key], mask, universe['fwhm'])
                    u_ROIs_time_series.append(masked_smoothed_data)
                    u_run_names.append('OFC_' + runType + nRunType + '_' + key)
                del LSS_dict

            else:  # kind == 'raw', with physio regression added
                func_img = nilearn.image.load_img(func, wildcards=True, dtype=None)
                func_4D = image.get_data(func_img)
                n_voxels = func_4D.shape[0] * func_4D.shape[1] * func_4D.shape[2]
                func_2D = func_4D.reshape(n_voxels, -1).T

                confounds, inc_TRs = load_confounds(func, strategy=('motion', 'wm_csf'),
                                                     motion='full', wm_csf='basic', demean=False)
                confounds = confounds.copy()

                if subject == 'sub-549' and ses == '2' and run == '2':
                    print('Skipping physio for sub-549 ses2 run2 since physio has an extra TR')
                else:
                    pattern = f"SUB{subject[-3:]}/SUB{subject[-3:]}_0{ses}/RUN-0{run}_*_multiple_regressors.txt"
                    physio_files = glob.glob(os.path.join(physio_path, pattern))
                    if physio_files:
                        physio_df = pd.read_csv(physio_files[0], sep="\t", header=None)
                        physio_df = physio_df.dropna(axis=1, how='all')
                        if confounds.shape[0] != physio_df.shape[0]:
                            print(f"WARNING: physio length mismatch for {subject} ses{ses} run{run} "
                                  f"({physio_df.shape[0]} rows vs {confounds.shape[0]} expected) - skipping physio for this run")
                        else:
                            confounds = pd.concat([confounds.reset_index(drop=True), physio_df.reset_index(drop=True)], axis=1)
                    else:
                        print("No matching physio file found for " + subject + " ses" + ses + " run " + run)

                func_2D_clean = nilearn.signal.clean(func_2D, runs=None, detrend=True,
                                                      standardize='zscore_sample', sample_mask=None,
                                                      confounds=confounds, standardize_confounds=True,
                                                      filter='butterworth', low_pass=None, high_pass=0.01, t_r=TR)
                func_4D_clean = func_2D_clean.T.reshape(func_4D.shape)

                # Mask BEFORE smoothing (mirrors mask_and_smooth_data's unmask->smooth->remask,
                # already used for every LSS universe) so the smoothing kernel only ever blends
                # real OFC signal with zero-padding at mask edges, not with neighboring tissue
                # (OFC borders the sinuses, so the signal just outside it is often contaminated).
                time_series = mask_and_smooth_data(func_4D_clean, mask, universe['fwhm'])
                u_ROIs_time_series.append(time_series)
                u_run_names.append('OFC_' + runType + nRunType)

                for extra_roi, extra_mask in extra_roi_masks.items():
                    extra_time_series = mask_and_smooth_data(func_4D_clean, extra_mask, universe['fwhm'])
                    u_ROIs_time_series.append(extra_time_series)
                    u_run_names.append(extra_roi + '_' + runType + nRunType)

        data_dict = {name: data for name, data in zip(u_run_names, u_ROIs_time_series)}
        npz_name = subject + ("_LSS.npz" if universe['kind'] == 'LSS' else ".npz")
        np.savez(u_output_dir + npz_name, **data_dict, allow_pickle=True)
        mri_log_df.to_csv(u_output_dir + subject + '.csv', index=False)
        np.save(u_output_dir + subject + '_dropout.npy', np.array([dropout_ratio]))

#%% LOOP OVER RUNS OF CURRENT SUBJECT
 
# Get all func paths of this subject
func_dirs_raw = np.sort(glob.glob("%s%s/*/func/*space-T1w_desc-preproc_bold.nii.gz"%(derviatives_path,subject)))

# Remove runs that are in runs2exclude:
func_dirs = [s for s in func_dirs_raw if not any(sub in s for sub in runs2exclude)]  

#remove run that were stopped spontanously and take the second run instead:
if subject=='sub-507':
    substrings_to_remove = ['sub-507_ses-1_task-sf_run-03_space']
elif subject=='sub-539':
    substrings_to_remove = ['sub-539_ses-2_task-sf_run-02_space']
elif subject=='sub-542' :
    substrings_to_remove = ['sub-542_ses-2_task-sf_run-02_space']
elif subject=='sub-544' :
    substrings_to_remove = ['sub-544_ses-1_task-sf_run-03_space']
elif subject=='sub-553' :
    substrings_to_remove =  ['sub-553_ses-1_task-sf_run-05_space','sub-553_ses-3_task-sf_run-05_space' ]
elif subject=='sub-554' :
    substrings_to_remove = ['sub-554_ses-1_task-sf_run-07_space']
else:
    substrings_to_remove = []

func_dirs = [path for path in func_dirs if not any(sub in path for sub in substrings_to_remove)]
    
ROIs_time_series, run_names =[], []
# Genrate Events PD which gives TR,sec,eventType for all event
# Initialize an empty dataframe with specified columns to store events timing 
columns = ['subject', 'ses', 'run', 'runType', 
           'time', 'closest_tr', 'closest_tr_2', 'event_type',
           'trial_type','state', 'irrelevant_client', 'irrlevant_house_left', 
           'irrelevant_house_right', 'irrelevant_house_passive','primary_house_passive','secondary_house_passive',
           'reward', 'lhouse_reward', 'rhouse_reward', 'phouse_reward',
           'correct_resp', 'resp', 'is_correct','is_probed_passive', 'probe_th',
           'client_img', 'phouse_img',  'lhouse_img', 'rhouse_img'] 
mri_log_df = pd.DataFrame(columns=columns)

brain_mask, global_mean = droupout_brain_mask(func_dirs) #,drop_out_th=0.8)


# Load freesurfer aparcaseg atlas in native space from all func runs
fs_aparcaseg=[]
for func in func_dirs:
    index = func.find('desc')
    fs_aparcaseg_img = nilearn.image.load_img(func[:index+4] + '-aparcaseg_dseg.nii.gz', wildcards=True, dtype=None)
    fs_aparcaseg.append(image.get_data(fs_aparcaseg_img))

if runMultiverse:
    multiverse_mri_log_df, multiverse_runTypes = build_events_log(subject, func_dirs)
    run_multiverse_extraction(subject, func_dirs, fs_aparcaseg, brain_mask, global_mean,
                               multiverse_mri_log_df, multiverse_runTypes)
    sys.exit(0)

# Define ROI masks based on the union of brain_mask and relevant freesurfer labels across all runs
ROI_masks,dropout_ratio = [], []
# Assume the first functional image defines space for the output
ref_img = nilearn.image.load_img(func_dirs[0])
affine = ref_img.affine
header = ref_img.header

for i, ROI in enumerate(ROIs):
    isFuncROI = True if ROI in list(func_roi_contrast_dict.keys()) else False 
    idxs = roi_dict[ROI] 
    # 1. Save anatomical ROI mask (before dropout threshold)
    anat_mask = np.isin(fs_aparcaseg[0], idxs) & brain_mask
    n_vox_anat = anat_mask.sum()
    if 'OFC' in ROI or 'HPC' in ROI:
        anat_mask_img = nib.Nifti1Image(anat_mask.astype(np.uint8), affine, header)
        nib.save(anat_mask_img, os.path.join(output_time_series, f"{subject}_{ROI}_anat_mask.nii.gz"))

    # 2. Apply dropout threshold per functional image
    mask = anat_mask.copy()
    for ifunc in range(len(fs_aparcaseg)):
        mask = mask & np.isin(fs_aparcaseg[ifunc], idxs)
        if 'OFC' in ROI or 'HPC' in ROI:  # apply dropout threshold
            func_img = nilearn.image.load_img(func_dirs[ifunc], wildcards=True, dtype=None)
            func_4D = image.get_data(func_img)
            func_3D = np.mean(func_4D, axis=3)
            mask = mask & (func_3D > 0.3 * global_mean)
   
    # 3. Final mask (after dropout and brain_mask)
    mask = np.array(brain_mask & mask)
    if 'OFC' in ROI or 'HPC' in ROI:
        # Save the final functional mask
        func_mask_img = nib.Nifti1Image(mask.astype(np.uint8), affine, header)
        nib.save(func_mask_img, os.path.join(output_time_series, f"{subject}_{ROI}_func_mask.nii.gz"))
    
    # 4. Compute dropout ratio
    dropout_ratio.append(1- (mask.sum()/n_vox_anat))

    if isFuncROI: #functional ROI based on visloc 
        #load t map for relevant contrast:
        tmap_filename = func_roi_dir + subject + '_' + func_roi_contrast_dict[ROI] + '_t_map.nii.gz'
        tmap_img = nilearn.image.load_img(tmap_filename,wildcards=True, dtype=None)
        tmap = np.squeeze(image.get_data(tmap_img))
        assert(tmap.shape == mask.shape)

        # Binarize map based on desired ROI and threshold
        if  ROI=='VISUALFUNC':
            mask_func = tmap>2.33
        if  ROI=='VISUALFUNC_HIGHORDER':
            mask_func = tmap>2.33
        elif  ROI=='FFA':
            mask_func = tmap>2.33
        elif  ROI=='PPA':
            mask_func = tmap<-2.33
        elif  ROI=='LO':
            mask_func = tmap>2.33
            
        mask = mask & mask_func
        mask = mask.astype('int')


    total_voxels = np.sum(mask.astype(np.int64))
    if total_voxels == 0:
        print('sub-'+subject + ' roi ' + ROI + ': no voxels found')
    else:
        print('sub-'+subject + ' roi ' + ROI +  ' : ' + str(total_voxels) + ' voxels found')

    ROI_masks.append(mask)
    del mask
         
# Load functional data of each run and extract time series (VOXELS X TRs) from each ROI + timing dict   
runTypes=[] 
for ifunc,func in enumerate(func_dirs):
    
    # Extract session and run from func name and get runType and nRunType (i.e. active1-17, passive1-4)
    ses = func[func.find('ses')+4:func.find('ses')+5]
    run = func[func.find('run')+5:func.find('run')+6]
    print(subject + ' ses' + ses + ' run' + run)
    runType, nRunType = sesrun2runType(subject,ses,run)
    
    # No need to extract time series for visual localizer runs:
    if 'visloc' in runType:
        continue
    
    # If LSS - check if LSS beta maps were extracted for this run:
    if (data_type=='LSS' and not os.path.exists(LSS_dir + subject + '_' + runType + nRunType +'_LSS.npz')) :
        continue
    
    runTypes.append(runType+nRunType)

    # Load 4D data of current run 
    func_img = nilearn.image.load_img(func, wildcards=True, dtype=None)
    affine = func_img.affine
    func_4D = image.get_data(func_img)
    n_voxels = func_4D.shape[0] * func_4D.shape[1] * func_4D.shape[2]
    func_2D = func_4D.reshape(n_voxels, -1).T  # Shape: (t, n_voxels)
    
    if data_type=='LSS':
        LSS_dict = np.load(LSS_dir + subject + '_' + runType + nRunType +'_LSS.npz', allow_pickle=True)

    # Generate event dataframe from log file for later analyses 
    log = pd.read_csv(logs_path + subject.replace('sub-5', 'sub5') + '/' + subject[-3:] + '_' + ses + '_' + run + '_' + runType + nRunType + '_log.csv') # load csv log on current run
    for i, time, event_type, is_passive in zip(range(log.shape[0]), np.array(log.time), np.array(log.eventType), np.array(log.is_passive)):
        
        n_samples = func_2D.shape[0]
        tr_times = np.arange(0,n_samples*TR,TR)
        distances = np.abs(tr_times - (time+5))
        closest_2_trs = distances.argsort()[:2]
        closest_tr = closest_2_trs[0]
        closest_tr_2 = closest_2_trs[1]

        if is_passive:
            if subject=='sub-501' or subject=='sub-502': #old passive task - alway presese more or less than 50 
                lhouse_img, rhouse_img,  reward, lhouse_reward, rhouse_reward, probe_th = 'NA', 'NA','NA', 'NA', 'NA','NA'
    
                if event_type == 'client onset':
                    client_img,  phouse_img, phouse_reward = log.iloc[i].client_img, log.iloc[i+1].passive_house_img, log.iloc[i+1].passive_house_reward
                    is_probed_passive = 1 
                    probe_th = 50
                    correct_resp = 'less' if probe_th > phouse_reward else 'more'
                    resp = log.iloc[i+2].response
                    is_correct = 1 if correct_resp == resp  else 0


                elif event_type == 'single house onset':
                    client_img, phouse_img, phouse_reward = log.iloc[i-1].client_img, log.iloc[i].passive_house_img, log.iloc[i].passive_house_reward
                    is_probed_passive = 1 
                    probe_th = 50
                    correct_resp = 'less' if probe_th > phouse_reward else 'more'
                    resp = log.iloc[i+1].response
                    is_correct = 1 if correct_resp == resp  else 0


                elif event_type == 'response':
                    client_img, phouse_img, phouse_reward = log.iloc[i-2].client_img, log.iloc[i-1].passive_house_img, log.iloc[i-1].passive_house_reward
                    is_probed_passive = 1
                    probe_th = 50
                    correct_resp = 'less' if probe_th > phouse_reward else 'more'
                    resp = log.iloc[i].response
                    is_correct = 1 if correct_resp == resp  else 0
     
                else:
                    client_img, phouse_img, phouse_reward, is_probed_passive, state = 'NA', 'NA', 'NA', 'NA', 'NA'
                    correct_resp, resp, is_correct, probe_th = 'NA', 'NA', 'NA', 'NA'
                
            else:
                lhouse_img, rhouse_img,  reward, lhouse_reward, rhouse_reward, probe_th = 'NA', 'NA','NA', 'NA', 'NA','NA'
    
                if event_type == 'client onset':
                    client_img,  phouse_img, phouse_reward = log.iloc[i].client_img, log.iloc[i+1].passive_house_img, log.iloc[i+1].passive_house_reward
                    if (log.shape[0]<i+2) and (log.iloc[i+2].eventType == 'passive probe'):
                        is_probed_passive = 1 
                    else:
                        is_probed_passive = 0
                        
                    if is_probed_passive:
                        probe_th = log.iloc[i+2].probe_th
                        correct_resp = 'less' if log.iloc[i+2].probe_th > phouse_reward else 'more'
                        resp = log.iloc[i+3].response
                        is_correct = 1 if correct_resp == resp  else 0
                    else: 
                        correct_resp, resp, is_correct = 'NA', 'NA', 'NA'
                        
                elif event_type == 'single house onset':
                    client_img, phouse_img, phouse_reward = log.iloc[i-1].client_img, log.iloc[i].passive_house_img, log.iloc[i].passive_house_reward
                    if (log.shape[0]<i+1) and (log.iloc[i+1].eventType == 'passive probe'):
                        is_probed_passive = 1
                    else:
                        is_probed_passive = 0
                        
                    if is_probed_passive:
                        probe_th = log.iloc[i+1].probe_th
                        correct_resp = 'less' if log.iloc[i+1].probe_th > phouse_reward else 'more'
                        resp = log.iloc[i+2].response
                        is_correct = 1 if correct_resp == resp  else 0
                    else: 
                        correct_resp, resp, is_correct = 'NA', 'NA', 'NA'
                    
                elif event_type == 'response':
                    client_img, phouse_img, phouse_reward = log.iloc[i-3].client_img, log.iloc[i-2].passive_house_img, log.iloc[i-2].passive_house_reward
                    is_probed_passive = 1
                    probe_th = log.iloc[i-1].probe_th
                    correct_resp = 'less' if log.iloc[i-1].probe_th > phouse_reward else 'more'
                    resp = log.iloc[i].response
                    is_correct = 1 if correct_resp == resp  else 0
     
                else:
                    client_img, phouse_img, phouse_reward, is_probed_passive, state = 'NA', 'NA', 'NA', 'NA', 'NA'
                    correct_resp, resp, is_correct, probe_th = 'NA', 'NA', 'NA', 'NA'
                
        else: #active trial 
            phouse_img, phouse_reward, is_probed_passive, probe_th = 'NA', 'NA', 'NA', 'NA'
  
            if event_type == 'client onset':
                client_img, lhouse_img, rhouse_img = log.iloc[i].client_img, log.iloc[i+1].lhouse_img, log.iloc[i+1].rhouse_img
                reward, lhouse_reward, rhouse_reward = log.iloc[i+3].reward, log.iloc[i+1].lhouse_reward, log.iloc[i+1].rhouse_reward 
                correct_resp, resp = log.iloc[i+1].correctResp, log.iloc[i+2].response 
                is_correct = 1 if correct_resp == resp  else 0
                
            elif event_type == 'house onset':
                client_img, lhouse_img, rhouse_img = log.iloc[i-1].client_img, log.iloc[i].lhouse_img, log.iloc[i].rhouse_img
                reward, lhouse_reward, rhouse_reward = log.iloc[i+2].reward, log.iloc[i].lhouse_reward, log.iloc[i].rhouse_reward 
                correct_resp, resp = log.iloc[i].correctResp, log.iloc[i+1].response 
                is_correct = 1 if correct_resp == resp  else 0
                
            elif event_type == 'response':
                client_img, lhouse_img, rhouse_img = log.iloc[i-2].client_img, log.iloc[i-1].lhouse_img, log.iloc[i-1].rhouse_img
                reward, lhouse_reward, rhouse_reward = log.iloc[i+1].reward, log.iloc[i-1].lhouse_reward, log.iloc[i-1].rhouse_reward 
                correct_resp, resp = log.iloc[i-1].correctResp, log.iloc[i].response 
                is_correct = 1 if correct_resp == resp  else 0

            elif event_type == 'reward':
                client_img, lhouse_img, rhouse_img = log.iloc[i-3].client_img, log.iloc[i-2].lhouse_img, log.iloc[i-2].rhouse_img
                reward, lhouse_reward, rhouse_reward = log.iloc[i].reward, log.iloc[i-2].lhouse_reward, log.iloc[i-2].rhouse_reward 
                correct_resp, resp = log.iloc[i-2].correctResp, log.iloc[i-1].response 
                is_correct = 1 if correct_resp == resp  else 0
                
            else:
                client_img, lhouse_img, rhouse_img, state = 'NA', 'NA', 'NA', 'NA'
                reward, lhouse_reward, rhouse_reward = 'NA', 'NA', 'NA'
                correct_resp, resp, is_correct = 'NA', 'NA', 'NA'  
                
        if event_type == 'passive probe':
            client_img, phouse_img, phouse_reward = log.iloc[i-2].client_img, log.iloc[i-1].passive_house_img, log.iloc[i-1].passive_house_reward
            is_probed_passive = 1 
            is_passive=1
            probe_th = log.iloc[i].probe_th
            correct_resp = 'less' if log.iloc[i].probe_th > phouse_reward else 'more'
            resp = log.iloc[i+1].response
            is_correct = 1 if correct_resp == resp  else 0
        
        if 'prompt' not in event_type:
            trial_type = 'passive' if is_passive == 1 else 'active'
            state = convert_input2single_house_state(client_img, phouse_img) if is_passive ==1 else  convert_input2two_house_state(client_img, lhouse_img, rhouse_img)
        else:
            trial_type = 'NA'

        irrelevant_client = 'NA' if client_img == 'NA' else client_img.replace('-1','0')[9]
        irrlevant_house_left = 'NA' if lhouse_img == 'NA' else lhouse_img.replace('-1','0')[10]
        irrelevant_house_right = 'NA' if rhouse_img == 'NA' else rhouse_img.replace('-1','0')[10]
        irrelevant_house_passive = 'NA' if phouse_img == 'NA' else phouse_img.replace('-1','0')[10]
        primary_house_passive = 'NA' if phouse_img == 'NA' else phouse_img.replace('-1','0')[8]
        secondary_house_passive = 'NA' if phouse_img == 'NA' else phouse_img.replace('-1','0')[6]

        new_row = pd.DataFrame([[subject, ses, run, runType+nRunType, 
                                 time, closest_tr, closest_tr_2, event_type,
                                 trial_type, state, irrelevant_client, irrlevant_house_left,
                                 irrelevant_house_right,irrelevant_house_passive, primary_house_passive, secondary_house_passive,
                                 reward, lhouse_reward, rhouse_reward, phouse_reward,
                                 correct_resp,resp,is_correct,is_probed_passive, probe_th,
                                 client_img, phouse_img, lhouse_img, rhouse_img]],
                               columns=mri_log_df.columns)

        mri_log_df = pd.concat([mri_log_df, new_row], ignore_index=True)

    # PREPROCESSING - for raw data only (not LSS beta maps)
    if not log_only:
        if data_type =='raw':
            # Load fmriprep confounds 
            confounds, inc_TRs = load_confounds(func, strategy=('motion','wm_csf'), # 'high_pass', 
                                       motion='full', 
                                       wm_csf='basic', 
                                       demean=False) #
                             
            # Detrending, highpass, regress out confounfs                        
            func_2D_clean = nilearn.signal.clean(func_2D,
                                                runs=None, 
                                                detrend=True, 
                                                standardize='zscore_sample', #False, psc, zscore_sample TO DO or change to %singal change
                                                sample_mask=None, 
                                                confounds=confounds, 
                                                standardize_confounds=True, 
                                                filter='butterworth', 
                                                low_pass=None, 
                                                high_pass=0.01, 
                                                t_r=TR,)
            func_4D_clean = func_2D_clean.T.reshape(func_4D.shape)
    
            # Spatial smoothing - perhaps better before cleaning??
            
            # Create a Nifti image from the numpy array
            nifti_img = nib.Nifti1Image(func_4D_clean, affine)
            
            # Apply smoothing with Nilearn (choose your FWHM value)
            fwhm = 5  # Full-width at half maximum (in mm)
            smoothed_img = image.smooth_img(nifti_img, fwhm=fwhm)
            
            # Convert back to a Numpy array if needed
            func_4D_clean_smooth = smoothed_img.get_fdata()
            func_2D_clean_smooth = func_4D_clean_smooth.reshape(n_voxels, -1).T  # Shape: (t, n_voxels)
            

        # Loop over ROIs and extract time series arrays VOXELS X TRs
        for iROI, ROI in enumerate(ROIs):
            if np.sum(ROI_masks[iROI]) > 0:#if there are voxels in this mask
            
                # Flatten mask
                ROI_mask_flattened = ROI_masks[iROI].reshape(n_voxels,-1).T  # Flatten mask to match voxel indexing ---> VOXELS X 1
                ROI_mask_idx = np.where(ROI_mask_flattened==1)[1]
            
                # Extract time series for voxels in the ROI
                if data_type=='LSS':


                    for i, key in enumerate(LSS_dict.keys()):
                        fwhm = 5
                        masked_smoothed_data = mask_and_smooth_data(LSS_dict[key], ROI_masks[iROI], fwhm) # shape: (n_trials, n_voxels)

                        ROIs_time_series.append(masked_smoothed_data)
                        run_names.append(ROI+ '_' + runTypes[-1] + '_' + key)

                else: #raw
                    time_series = func_2D_clean_smooth[:, ROI_mask_idx]  # Shape: (timepoints, n_voxels_in_ROI)
                    print(f"sub-{subject} ROI {ROI}: Extracted time series with shape {time_series.shape}")
                    ROIs_time_series.append(time_series)
                    run_names.append(ROI+ '_' + runTypes[-1])
        try:     
            del LSS_dict,func_4D_clean, func_2D_clean, func_4D, func_2D, func_img, time_series, func_2D_clean_smooth, func_4D_clean_smooth
        except NameError:
            pass
        
if not log_only:

    # Create the data dictionary
    data_dict = {name: data for name, data in zip(run_names, ROIs_time_series)}

    # Save the dictionary using np.savez
    npz_suffix = '_left_right' if roi_mode == 'left_right' else ''
    if data_type=='LSS':
        np.savez(output_time_series + subject + "_LSS" + npz_suffix + ".npz", **data_dict, allow_pickle=True)

    else:
        np.savez(output_time_series + subject + npz_suffix + ".npz", **data_dict, allow_pickle=True)

# Save the corresponding DF specifiying events and TRs. It is identical regardless of
# ROI selection, so only write it in 'full' mode to avoid touching the file the
# day-1 draft analyses depend on. Same for the dropout ratio array, whose entries
# are positionally tied to the 'full' ROIs list.
if roi_mode == 'full':
    mri_log_df.to_csv(output_time_series + subject + '.csv', index=False)
    np.save(output_time_series + subject + '_dropout.npy', np.array(dropout_ratio))
else:
    np.save(output_time_series + subject + '_dropout_left_right.npy', np.array(dropout_ratio))
    
