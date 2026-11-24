#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# First-level GLMs (nilearn) for one participant.
#
# Main use: single-trial beta maps with the least-squares-separate (LSS) approach,
# for the learning and estimation runs, and the visual localizer GLM
# (used to define the FFA / VTC ROIs).
# Input:  fMRIPrep outputs, events from the task logs, physiological regressors (PhysIO)
# Output: nilearn_outputs/vol_maps/<GLM folder>/<sub>_<run type>_LSS.npz (one beta map per trial)
# Settings at the top choose the noise regressors: the default is the main pipeline;
# the CompCor variant is the robustness analysis of Fig. S3.
# Run: runGLM_nilearn_Tardis.sh (one job per participant), or python runGLM_nilearn.py sub-5XX
#
# Author: Shany Grossman



#%% import
import sys
import os
import glob
import numpy as np
import pandas as pd
from nilearn import image
import nibabel as nib 
from nilearn.glm.first_level import FirstLevelModel
from nilearn.plotting import plot_design_matrix
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.glm.contrasts import compute_fixed_effects
from nilearn import plotting
from nilearn.plotting import plot_stat_map, plot_surf_stat_map
from nilearn.image import mean_img   
from nilearn.image import concat_imgs
from nilearn.glm.second_level import SecondLevelModel
from nilearn.datasets import fetch_surf_fsaverage,load_fsaverage,load_fsaverage_data
from nilearn.surface import load_surf_data, SurfaceImage, PolyData, PolyMesh
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from nilearn.interfaces.fmriprep import load_confounds
from scipy.stats import mode
from nilearn.image import concat_imgs, new_img_like
from scipy.linalg import sqrtm, inv


subj = sys.argv[1][:3] + '-' + sys.argv[1][3:]
print(subj + '...')


#%% Set paths:
base_dir = '/home/mpib/grossman/STATE_FORMATION/MRI_DATA/MRI_REAL/'
derviatives_path = base_dir + 'derivatives/'
physio_path = base_dir + 'PHYSIO_processed/'

#%% Carry analysis on surface/volume?
isVolume = True # If false - analysis will be carried on surface
isNative = True # If false, analyses will be carried in stadnard
                 # common space(FSAverage/MNI for surface and volume, repspectively).
                 # All analyses in the paper use native space (True), except the localizer
                 # group maps (runGLM_2ndLevel_nilearn.py), which use the MNI-space maps (False).
GLM_model = 'LSS' # 'LSS' (single-trial betas) or 'standard'
run_types_2_include =['passive', 'active'] #['visloc_categories'] #'  #['passive', 'active', 'visloc_categories', 'visloc_features]
sessions=[1, 2, 3]
# Default pipeline (main text): physiological noise (RETROICOR) regressed out, no CompCor.
#   -> nilearn_outputs/vol_maps/LSS_physio_regressed_out_updated/
#   The FFA/VTC localizer maps used for ROI definition come from the same noise settings with
#   GLM_model = 'standard' and run_types_2_include = ['visloc_categories']
#   -> nilearn_outputs/vol_maps/visloc_categories_physio_regressed_out_updated/ (one t map per
#   contrast, read by a_extract_ROI_signals.py).
# Robustness variant (Fig. S3, "CompCor physiological noise correction"):
#   add_physio_regressors = False, add_compcor = True, compcor_strategy = 'anat_separated'
#   (5 aCompCor components each from the WM and CSF masks) -> nilearn_outputs/vol_maps/LSS_compcorr/
add_physio_regressors = True
add_compcor = False
compcor_strategy = 'anat_separated' # only used when add_compcor = True
save_resid = False

#%% GLM specific params
hrf_model = "spm" #"glover" #"spm" #check! double gamma bmight be better

#%%
logs_path = base_dir + 'LOGs/'


if add_physio_regressors and add_compcor:
    suf_str1 = '_physio_compcorr'
elif add_physio_regressors:
    suf_str1 = '_physio_regressed_out_updated'
elif add_compcor:
    # anat_separated (already-run universe) keeps the original '_compcorr' name untouched;
    # anat_combined (this run) gets its own suffix so it lands in a separate directory.
    suf_str1 = '_compcorr_combined' if compcor_strategy == 'anat_combined' else '_compcorr'
else:
    suf_str1 = ''
    
if save_resid:
    suf_str2 = '_with_resids'
else:
    suf_str2 = ''
    
output_dir_figures = base_dir + 'nilearn_outputs/figures/' + GLM_model + suf_str1 + suf_str2 + '/'
output_dir_surf = base_dir + 'nilearn_outputs/surf_maps/' + GLM_model + suf_str1 + suf_str2 + '/'
output_dir_vol = base_dir + 'nilearn_outputs/vol_maps/' + GLM_model+ suf_str1 + suf_str2 + '/'    

for ipath, path in enumerate([output_dir_figures,output_dir_surf,output_dir_vol]):
    os.makedirs(path, exist_ok=True)


#%% Set subjects list
subjects = [subj] #,'sub-509'] # 'sub-510'] # 


#%% From here on no need to adapt anything
# Define conditions of interest:
# Client passive
# Client active
# House actve (pair)
# House passive (single)
# Reward 
# Response active
# Response passive
# Passive probe

conds_active_blocks = sorted(['client_active', 'client_passive', 'house_active', 'house_passive','reward', 'response active', 'response passive', 'passive_probe'])
contrasts_active_blocks = sorted(['face_active>houses_active', 
                           'face_passive>house_passive', 
                           'client_active', 'client_passive', 'house_active', 'house_passive','reward', 'response active', 'response passive', 'passive_probe'])

conds_active_blocks_day2 = sorted(['client_active','house_active','reward', 'response active'])
contrasts_active_blocks_day2 = sorted(['face_active>houses_active','client_active', 'house_active', 'reward', 'response active'])

conds_passive_blocks = sorted(['client_passive', 'house_passive','response passive', 'passive_probe'])
contrasts_passive_blocks = sorted(['client_passive','house_passive', 'response passive', 'passive_probe','face_passive>house_passive' ])

# Basic visual localizer with 5 categories - until 513 including
conds_vis_loc1_v1 = sorted(['object', 'scrambled_object', 'face', 'face_profile', 'face_glasses', 'body', 'house', 'scene'])
contrasts_vis_loc1_v1 = sorted(['object', 'scrambled_object', 'face', 'face_profile', 'face_glasses', 'body', 'house', 'scene',
                         'all_visual',
                         'faces',
                         'scenes',
                         'faces>scenes',
                         'objects>scrambled_objects',
                         'all_cat>scrambled_objects'])                         

# Changed version from 514 onwards:
conds_vis_loc1_v2 = sorted(['house', 'car', 'scrambled', 'adult', 'instrument', 'corridor', 'pattern'])
contrasts_vis_loc1_v2 = sorted(['house', 'car', 'scrambled', 'adult', 'instrument', 'corridor', 'pattern',
                         'all_visual',
                         'faces',
                         'scenes',
                         'faces>scenes',
                         'objects>scrambled_objects',
                         'all_cat>scrambled_objects'])

contrasts2save = sorted(['all_visual', 'faces>scenes','objects>scrambled_objects',  'all_cat>scrambled_objects'])
contrasts2save = sorted([
    # 'all_visual', 'objects>scrambled_objects',  # unused for the current FFA/VTC group-map task
    'faces>scenes', 'all_cat>scrambled_objects'])

block_types = [['passive'] + ['active']*6,
               ['active']*5 + ['passive'],
               ['passive'] + ['active']*5 + ['passive'] + ['vis_loc1'] + ['vis_loc2']]

#%% Define helper functions:
      
def flatten(xss):
    return [x for xs in xss for x in xs]

def index_containing_substring(my_list, substrings):
    if isinstance(substrings, str):
        substrings = [substrings]  # Convert single substring to a list
    
    indices = []
    for i, s in enumerate(my_list):
        if any(sub in s for sub in substrings):
            indices.append(i)
    
    return indices

def index_not_containing_substring(my_list, substrings):
    if isinstance(substrings, str):
        substrings = [substrings]  # Convert single substring to a list

    return [i for i, item in enumerate(my_list) if not any(sub in item for sub in substrings)]

def sesrun2runType(sub,ses,run): #TO DO - add excpetions 
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
      "ses3-run8":["visloc_categories",""],
      "ses3-run9":["visloc_features",""],
      "ses302-run1":["active","16"],
      "ses302-run2":["active","17"],
      "ses302-run3":["passive","4"],
      "ses302-run4":["visloc_categories",""],
      "ses302-run5":["visloc_features",""],
    }
        
    return conversiondict[s][0] + conversiondict[s][1] 

def get_contrast_val(contrast,conds,X):
    contrast_val = [[0]*(X.shape[-1])] #now edit entries according to specific contrasts; +1 is for the constant regresoor
    
    if contrast in conds: #single condition > baseline
        for i,x in enumerate(conds):
            contrast_val[0][i]=1 if contrast==x else 0
        
    else: #define specific contrasts
        
        if contrast=="face_active>houses_active":
            faces_conds = index_containing_substring(conds, 'client_active')
            for i in faces_conds:
                contrast_val[0][i]=1 
            
            house_conds = index_containing_substring(conds, 'house_active')
            for i in house_conds:
                contrast_val[0][i]=-1
                
        elif contrast=="face_passive>house_passive":
            faces_conds = index_containing_substring(conds, 'client_passive')
            for i in faces_conds:
                contrast_val[0][i]=1 
            
            house_conds = index_containing_substring(conds, 'house_passive')
            for i in house_conds:
                contrast_val[0][i]=-1
                
        elif  contrast=="all_visual":
            for i in range(len(conds)):
                contrast_val[0][i]=1
                
        elif  contrast=="faces":
            faces_conds = index_containing_substring(conds, ['face', 'adult'])
            for i in faces_conds:
                contrast_val[0][i]=1 
                
        elif  contrast=="scenes":
            house_conds = index_containing_substring(conds, ['house', 'scene', 'corridor'])
            for i in house_conds:
                contrast_val[0][i]=-1
                
        elif  contrast=="faces>scenes":
            faces_conds = index_containing_substring(conds, ['face', 'adult'])
            for i in faces_conds:
                contrast_val[0][i]=1/len(faces_conds) 
            
            house_conds = index_containing_substring(conds, ['house', 'scene', 'corridor'])
            for i in house_conds:
                contrast_val[0][i]=-1/len(house_conds) 
                
        elif  contrast=="objects>scrambled_objects":
            conds_suf = [s.split('_')[0] for s in conds]
            object_conds = index_containing_substring(conds_suf, ['object', 'car', 'instrument'])
            for i in object_conds:
                contrast_val[0][i]=1/len(object_conds) 
            
            scrambled_conds = index_containing_substring(conds_suf, ['scrambled'])
            for i in scrambled_conds:
                contrast_val[0][i]=-1/len(scrambled_conds) 
                
        elif  contrast=="all_cat>scrambled_objects":
            conds_suf = [s.split('_')[0] for s in conds]
            scrambled_conds = index_containing_substring(conds_suf, ['scrambled'])
            non_scrambled_conds = index_not_containing_substring(conds_suf, ['scrambled'])
            for i in scrambled_conds:
                contrast_val[0][i]=-1/len(scrambled_conds) 
            for i in non_scrambled_conds:
                contrast_val[0][i]=1/len(non_scrambled_conds) 


    return contrast_val

def vis_loc_to_prt(log,cond):
    # Find duration of a single image
    # Compute differences and round
    differences = np.round(np.array(log['offset'] - log['time']), 2)
    # Get the most common value
    img_dur = mode(differences, keepdims=True).mode[0]

    # Find blocks onsets and offsets 
    log['block_start'] = log['category'] != log['category'].shift(1)
    log['block_end'] = log['block_start'].shift(-1)
    log.at[log.index[-1], 'block_end'] = True
    
    #for v2 - extract only the first substring of the category string
    if any(log['category'].unique()=='adult'):
        log['category'] = log['category'].str.split('-').str[0]
        log['category'] = log['category'].str.split('_').str[0]


    onset_c_cond = log[(log['block_start']==1) & (log['category'].isin([cond]))].time.to_list()
    duration_c_cond =  list(np.array(log[(log['block_end']==1) & (log['category'].isin([cond]))].time + img_dur) - np.array(onset_c_cond))
    conditions_c_cond = [cond]*len(onset_c_cond)
    
    return onset_c_cond, duration_c_cond, conditions_c_cond
    
def generate_input_for_deisgnMatrix(sub,ses,run,conds):
    # Load log
    log_name = np.sort(glob.glob(logs_path + sub.replace('-','') + '/' + sub[-3:] + '_' + str(ses) + '_' + str(run) + '*.csv'))[0]
    print(log_name)
    log = pd.read_csv(log_name)
    conditions,durations,onsets = [],[],[]
    
    for icond, cond in enumerate(conds):
        if cond=='client_active':
            n_events = len(log[(log.eventType =='client onset') & (log.is_passive == 0)].time)
            conditions.append([cond]*n_events)
            onsets.append(list(np.array(log[(log.eventType =='client onset') & (log.is_passive == 0)].time)))
            durations.append([1]*n_events)#check!
            
        elif cond=='client_passive':
            n_events = len(log[(log.eventType =='client onset') & (log.is_passive == 1)].time)
            conditions.append([cond]*n_events)
            onsets.append(list(np.array(log[(log.eventType =='client onset') & (log.is_passive == 1)].time)))
            durations.append([1]*n_events)#check!
            
        elif cond=='house_active':
            n_events = len(log[log.eventType =='house onset'].time)
            conditions.append([cond]*n_events)
            onsets.append(list(np.array(log[log.eventType =='house onset'].time)))
            durations.append([2.5]*n_events)#check!
            
        elif cond=='house_passive':
            n_events = len(log[log.eventType =='single house onset'].time)
            conditions.append([cond]*n_events)
            onsets.append(list(np.array(log[log.eventType =='single house onset'].time)))
            durations.append([1]*n_events) #check!
            
        elif cond=='reward':
            n_events = len(log[log.eventType =='reward'].time)
            conditions.append([cond]*n_events)
            onsets.append(list(np.array(log[log.eventType =='reward'].time)))
            durations.append([1]*n_events) #check! - as of a certain partcipacnt i rdieced to 750 ms
            
        elif cond=='response active':
            n_events = len(log[(log.eventType =='response') & (log.is_passive == 0) & (log['time'].notna())].time)
            conditions.append([cond]*n_events)
            onsets.append(list(np.array(log[(log.eventType =='response') & (log.is_passive == 0) & (log['time'].notna())].time)))
            durations.append([0.1]*n_events) 
            #To Do - check how this is possible - and model these events with a outliar regressor (sub 509, ses1 run2)

        elif cond=='response passive': #design was different for 501-?
            n_events = len(log[(log.eventType =='response') & (log.is_passive == 1) & (log['time'].notna())].time)
            conditions.append([cond]*n_events)
            onsets.append(list(np.array(log[(log.eventType =='response') & (log.is_passive == 1)  & (log['time'].notna())].time)))
            durations.append([0.1]*n_events) 
        
        elif cond=='passive_probe':
            n_events = len(log[log.eventType =='passive probe'].time)
            conditions.append([cond]*n_events)
            onsets.append(list(np.array(log[log.eventType =='passive probe'].time)))
            durations.append([2]*n_events) 
            
        elif cond in ['object', 'scrambled_object', 'face', 'face_profile', 'face_glasses', 'body', 'house', 'scene', 'car', 'scrambled', 'adult', 'instrument', 'corridor', 'pattern']:
            onset_c_cond, duration_c_cond, conditions_c_cond =  vis_loc_to_prt(log,cond)
            conditions.append(conditions_c_cond)
            onsets.append(onset_c_cond)
            durations.append(list(duration_c_cond)) 
            
    # Flatten list of lists into one long list
    conditions, durations, onsets  = flatten(conditions), flatten(durations), flatten(onsets),
    
    return  conditions, durations, onsets


def lss_transformer(events_df, row_number):
    """Label one trial for one LSS model.

    Parameters
    ----------
    df : pandas.DataFrame
        BIDS-compliant events file information.
    row_number : int
        Row number in the DataFrame.
        This indexes the trial that will be isolated.

    Returns
    -------
    df : pandas.DataFrame
        Update events information, with the select trial's trial type isolated.
    trial_name : str
        Name of the isolated trial's trial type.
    """
    events_df = events_df.copy()

    # Determine which number trial it is *within the condition*
    trial_condition = events_df.loc[row_number, "trial_type"]
    trial_type_series = events_df["trial_type"]
    trial_type_series = trial_type_series.loc[
        trial_type_series == trial_condition
    ]
    trial_type_list = trial_type_series.index.tolist()
    trial_number = trial_type_list.index(row_number)

    # We use a unique delimiter here (``__``) that shouldn't be in the
    # original condition names.
    # Technically, all you need is for the requested trial to have a unique
    # 'trial_type' *within* the dataframe, rather than across models.
    # However, we may want to have meaningful 'trial_type's (e.g., 'Left_001')
    # across models, so that you could track individual trials across models.
    trial_name = f"{trial_condition}__{trial_number:03d}"
    events_df.loc[row_number, "trial_type"] = trial_name
    return events_df, trial_name

#%% Loop over subjetcs to build a first level GLM (and then, if GLM_model == stadndart and run types are not 'vis_loc' - where there is only one run, run FE across runs):
    
if not isVolume: #surface
    if not isNative: #common
        # Load the fsaverage template
        fsaverage = load_fsaverage(mesh='fsaverage')
        fsaverage_data = load_fsaverage_data(mesh='fsaverage', mesh_type='pial', data_type="sulcal")


for isub, sub in enumerate(subjects): 
    
    # Get filenames of all func files from all relevant sessions
    func_files_per_ses=[];
    csub_block_types = []
    
    if sub=='sub-534' and 3 in sessions: #add the 302 session folder 
        sessions.append(302)
    for ses in sessions:
        # Define the path and the file extension pattern
        sub_dir = derviatives_path + sub + '/ses-' + str(ses) + '/func/'
        if isVolume:
            if not isNative: # Volume, MNI
                func_pattern = sub + '_ses-' + str(ses) + '_task-sf_run-*_space-MNI152NLin6Asym_desc-preproc_bold.nii.gz'
            else: # Volumne, Native
                func_pattern = sub + '_ses-' + str(ses) + '_task-sf_run-*_space-T1w_desc-preproc_bold.nii.gz'
        else: 
            if not isNative: # Surface, FSAverage
                func_pattern = sub + '_ses-' + str(ses) + '_task-sf_run-*_hemi-L_space-fsnative_bold.func.gii'
            else: # Surface, Native
                func_pattern = sub + '_ses-' + str(ses) + '_task-sf_run-*_hemi-L_space-fsaverage_bold.func.gii'
 
        func_files_per_ses.append(np.sort(glob.glob(sub_dir + func_pattern)))
 
    func_files = flatten(func_files_per_ses)
    # Remove runs that were stopped spontanously and the second run will be taken instead:
    if sub=='sub-507':
        substrings_to_remove = ['sub-507_ses-1_task-sf_run-03_space']
    elif sub=='sub-539':
        substrings_to_remove = ['sub-539_ses-2_task-sf_run-02_space']
    elif sub=='sub-542' :
        substrings_to_remove = ['sub-542_ses-2_task-sf_run-02_space']
    elif sub=='sub-544' :
        substrings_to_remove = ['sub-544_ses-1_task-sf_run-03_space']
    elif sub=='sub-553' :
        substrings_to_remove = ['sub-553_ses-1_task-sf_run-05_space','sub-553_ses-3_task-sf_run-05_space' ]
    elif sub=='sub-554' :
        substrings_to_remove = ['sub-554_ses-1_task-sf_run-07_space']
    else:
        substrings_to_remove = []

    func_files = [path for path in func_files if not any(sub in path for sub in substrings_to_remove)]
    print('found ' + str(len(func_files)) + ' func files for ' + sub + ', loading them now...')

    # Brain mask is derived per-run from the functional file (same space/resolution, no resampling needed).
    # Restricts AR1 estimation to in-brain voxels, preventing NaN rho from near-zero residuals
    # in outside-brain voxels which causes SVD failure in AR whitening.

    # Initialize a 2d list (run, contrast) to store single run GLM results
    n_contrasts, n_runs = 20, len(func_files) # overshooting the num of contratsts, some of which will be left empty depending on the run type 
    summary_statistics = [[None for _ in range(n_contrasts)] for _ in range(n_runs)]

    for irun in range(len(func_files)):
        # Generate design matrix per run
        run = func_files[irun][func_files[irun].find('run')+5]
        ses = func_files[irun][func_files[irun].find('ses-')+4]
        if sub=='sub-534' and 'ses-302' in func_files[irun]:
            block_type = sesrun2runType(sub,'302',run)
            if block_type == 'visloc_categories': run='8'
        elif sub=='sub-553' and 'run-0802' in func_files[irun]:
            # run-0802 is the restarted full active16 scan (run-05 was stopped mid-scan)
            block_type = 'active16'
            run = '5'  # log is named 553_3_5_active16_log.csv
        else:
            block_type = sesrun2runType(sub,ses,run)
        
        # Check if this run is included in the alaysis (e.g., is we a running LSS, no need to include visloc)
        if any(run_type in block_type for run_type in run_types_2_include):
            csub_block_types.append(block_type)
        else:
            continue
        
        # Load func file
        if isVolume:
            func_img = image.load_img(func_files[irun])
            # Per-run brain mask: same space/resolution as func, no resampling needed
            brain_mask_path = func_files[irun].replace('_desc-preproc_bold.nii.gz', '_desc-brain_mask.nii.gz')
            brain_mask = brain_mask_path if os.path.exists(brain_mask_path) else None
            if brain_mask is None:
                print(f'WARNING: no per-run brain mask found for {func_files[irun]}, AR1 may be unstable')
            # Create mean image for plotting purposes (to do - plot on MNI T1/Native T1)
            mean_img_ = mean_img(func_img)
        else:
            func_img_l_hemi = load_surf_data(func_files[irun]) 
            func_file_rh = func_files[irun].replace('hemi-L', 'hemi-R') 
            func_img_r_hemi = load_surf_data(func_file_rh) 
            # Combine the two hemispheres into one dataset (time points X vertices)
            func_img = SurfaceImage(mesh=fsaverage["pial"], data = PolyData(left=func_img_l_hemi, right=func_img_r_hemi)) 
            del func_img_l_hemi, func_img_r_hemi  # Free memory
        
        # Define conditions and contrast depending in the block_type: 
        if 'passive' in block_type:
            conds=conds_passive_blocks.copy()
            contrasts=contrasts_passive_blocks.copy()
        elif sub=='sub-534' and 'ses-302' in func_files[irun] and 'active' in block_type:
            conds=conds_active_blocks_day2.copy()
            contrasts=contrasts_active_blocks_day2.copy()
        elif 'active' in  block_types[int(ses)-1][int(run)-1]:
            if ses==3:
                conds=conds_active_blocks_day2.copy()
                contrasts=contrasts_active_blocks_day2.copy()
            else:
                conds=conds_active_blocks.copy()
                contrasts=contrasts_active_blocks.copy()
        elif 'visloc_categories' in block_type:
            if int(sub[-3:])<514: # until 513 including, v1
                conds=conds_vis_loc1_v1.copy()
                contrasts=contrasts_vis_loc1_v1.copy()
            else: # 514 onwards, v2
                conds=conds_vis_loc1_v2.copy()
                contrasts=contrasts_vis_loc1_v2.copy()
        else:
            raise ValueError('Conditions and contrast are not specified for block type ' + block_types[ses-1][run] + '!')
            
        print(sub + ': first level GLM run ' + run + ' (' + block_type  + ')')
         
        conditions,durations,onsets = generate_input_for_deisgnMatrix(sub,ses,run,conds)
        events = pd.DataFrame({"trial_type": conditions, "onset": onsets, "duration": durations})
        n_scans = func_img.shape[-1] if isVolume else np.min(func_img.shape)
        frame_times = (np.arange(n_scans) * 1.75) 
        
        # To do: add motion and noise regressors from fmriprep
        
        # Load fmriprep confounds
        if add_compcor:
            confounds = load_confounds(func_files[irun], strategy=('motion', 'high_pass', 'compcor'),
                                       motion='full',
                                       compcor=compcor_strategy,
                                       n_compcor=5,
                                       demean=True)
        else:
            confounds = load_confounds(func_files[irun], strategy=('motion', 'high_pass', 'wm_csf'),
                                       motion='full',
                                       wm_csf='basic',
                                       demean=True)

        confounds=confounds[0].copy()


        if add_physio_regressors:
            if sub=='sub-549' and ses=='2' and run=='2':
                print('Skipping physio for sub-549 ses2 run2 since physio has an extra TR')
            else:
                # Load processed PhysIO regressors
                pattern = f"SUB{sub[-3:]}/SUB{sub[-3:]}_0{ses}/RUN-0{run}_*_multiple_regressors.txt"
                physio_files = glob.glob(os.path.join(physio_path, pattern))
                if physio_files:
                    physio_file=physio_files[0]
                    physio_df = pd.read_csv(physio_file, sep="\t", header=None)
                    physio_df = physio_df.dropna(axis=1, how='all') # remove columns that are all nans
                    
                    # Make sure length is the same as fmriprep confounds:
                    if confounds.shape[0] != physio_df.shape[0]:
                        print(f"WARNING: physio length mismatch for {sub} ses{ses} run{run} "
                              f"({physio_df.shape[0]} rows vs {confounds.shape[0]} expected) - skipping physio for this run")
                        physio_df = None

                    # concat with fmriprep confounds:
                    if physio_df is not None:
                        confounds = pd.concat([confounds.reset_index(drop=True), physio_df.reset_index(drop=True)], axis=1)
    
                else:
                    print("No matching physio file file found for " + sub + "ses" + ses + "run " + run )


        # Build design matrix 
        X = make_first_level_design_matrix(
            frame_times,
            events,
            drift_model=None,
            drift_order=0,
            add_regs=np.array(confounds),
            add_reg_names=confounds.columns.tolist(),
            hrf_model=hrf_model,)


        if GLM_model =='standard':
            # Fit first level GLM for current run 
            fmri_glm = FirstLevelModel(
                    t_r=1.75,
                    hrf_model=hrf_model,
                    smoothing_fwhm=5,
                    drift_model=None,
                    noise_model='ar1',           # <-- Prewhitening added
                    minimize_memory=False,       # <-- Required for noise_model to take effect
                    mask_img=brain_mask)
            fmri_glm.fit(func_img, design_matrices=X)
            
            # Compute run-specific contrasts to later pass to fixed effects GLM per particiapnt:
            for icontrast, contrast in enumerate(contrasts):
                contrast_val = get_contrast_val(contrast,conds,X)
                summary_statistics[irun][icontrast] = fmri_glm.compute_contrast(
                    contrast_val,
                    output_type="all")
                
                # If it is a vis_loc run, we are done here and we can save the maps    
                if 'visloc' in block_type:
                   
                    beta_map = summary_statistics[irun][icontrast]["stat"]
                    if contrast in contrasts2save:
                        save_path = output_dir_vol.replace(GLM_model + suf_str1 + suf_str2, block_type + suf_str1 + suf_str2)
                        if not os.path.exists(save_path):
                            # Create the folder
                            os.makedirs(save_path)
                            
                        print('saving visloc run contrast ' + contrast + ' in:' + save_path)
                        if isNative:
                            
                            beta_map.to_filename( save_path + sub + '_' + contrast + "_t_map.nii.gz")
                        else: 
                            beta_map.to_filename( save_path + sub + '_' + contrast + "_t_map_standard.nii.gz")
                    
        elif GLM_model =='LSS':


            # Loop through the trials of interest and transform the DataFrame for LSS
            lss_beta_maps = {cond: [] for cond in events["trial_type"].unique()}
            
            for i_trial in range(events.shape[0]):
                lss_events, trial_condition = lss_transformer(events, i_trial)

                # Compute and collect beta maps
                lss_glm = FirstLevelModel(
                        t_r=1.75,
                        hrf_model=hrf_model,
                        smoothing_fwhm=None,
                        drift_model=None,
                        noise_model='ar1',           # <-- Prewhitening added
                        minimize_memory=False,       # <-- Required for noise_model to take effect
                        mask_img=brain_mask)

                try:
                    lss_glm.fit(func_img, lss_events, confounds=confounds)
                except np.linalg.LinAlgError:
                    print(f"WARNING: AR1 SVD failed for {sub} {block_type} trial {i_trial}, falling back to OLS")
                    lss_glm = FirstLevelModel(
                            t_r=1.75,
                            hrf_model=hrf_model,
                            smoothing_fwhm=None,
                            drift_model=None,
                            noise_model='ols',
                            mask_img=brain_mask)
                    lss_glm.fit(func_img, lss_events, confounds=confounds)
            
                # We will save the design matrices across trials to show them later
            
                beta_map = lss_glm.compute_contrast(
                    trial_condition,
                    output_type="effect_size",
                )
            
                # Drop the trial number from the condition name to get the original name
                condition_name = trial_condition.split("__")[0]
                lss_beta_maps[condition_name].append(beta_map)

            # Conctanate maps into a list
            lss_beta_maps_4d = {
                    name: concat_imgs(maps) for name, maps in lss_beta_maps.items()}


            if save_resid:
                lss_beta_maps_4d_resid = {}
                lss_beta_maps_4d_data_resid = {}
            
                for condition_name, img_4d in lss_beta_maps_4d.items():
            
                    # Get residuals from the last fitted GLM
                    resid_img = lss_glm.residuals[0]   # 4D image (X,Y,Z,T)
            
                    resid_data = resid_img.get_fdata()
            
                    # Store in same structure as beta maps
                    lss_beta_maps_4d_data_resid[condition_name] = resid_data
                    lss_beta_maps_4d_resid[condition_name] = resid_img


            lss_beta_maps_4d_data = {
                    name: img.get_fdata() for name, img in lss_beta_maps_4d.items()}

            # Save the dictionary for current run using np.savez
            if isVolume:
                np.savez(output_dir_vol + sub + '_' + block_type + '_LSS.npz', **lss_beta_maps_4d_data, allow_pickle=True)
                if save_resid:
                    np.savez(output_dir_vol + sub + '_' + block_type + '_LSS_residuals.npz', **lss_beta_maps_4d_data, allow_pickle=True)
            else:
                np.savez(output_dir_surf + sub +  '_' + block_type + '_LSS.npz', **lss_beta_maps_4d_data, allow_pickle=True)  
            
            del lss_beta_maps_4d, lss_beta_maps, lss_beta_maps_4d_data

    # End of loop over runs for current subject
    
    #%% Compute FE map across runs per contrast for current subject
    non_visloc = [s for s in run_types_2_include if 'visloc' not in s]
    if GLM_model =='standard' and len(non_visloc)>0:
        
        for i_block_type, block_type in enumerate(['active', 'passive']):
            runs_idx = np.where(np.array([block_type in x for x in csub_block_types]))[0]
            
            if block_type=='passive':
                conds=conds_passive_blocks.copy()
                contrasts=contrasts_passive_blocks.copy()
            elif block_type=='active':
                conds=conds_active_blocks.copy()
                contrasts=contrasts_active_blocks.copy()
            else:
                raise ValueError('Conditions and contrast are not specified for block type ' + block_type + '!')
    
            for icontrast, contrast in enumerate(contrasts):
                contrast_imgs = [summary_statistics[irun][icontrast]["effect_size"] for irun in runs_idx]
                variance_imgs = [summary_statistics[irun][icontrast]["effect_variance"] for irun in runs_idx]
        
                fe_contrast, fe_variance, fe_stat = compute_fixed_effects(
                contrast_imgs,
                variance_imgs,)
                #data["mask"],)
                
                # Plotting 
                plt.ioff() 
                if isVolume:
                    plotting.plot_stat_map(fe_stat,
                    bg_img=mean_img_,
                    threshold=5.0,
                    display_mode="ortho",
                    title=f"{sub} - {block_type} blocks: {contrast}, fixed effects",) 
                else:
    
                    # we plot it on the surface, on the inflated fsaverage mesh,
                    # together with a suitable background to give an impression
                    # of the cortex folding.
                    
                    # Set up a Matplotlib figure for side-by-side plots
                    fig, axes = plt.subplots(3, 2, subplot_kw={"projection": "3d"}, figsize=(10, 12))
                    fig.tight_layout(pad=0, h_pad=0, w_pad=0) 
                    fig.subplots_adjust(wspace=0.01)
                    fig.suptitle(f"{sub} - {block_type} blocks: {contrast}, fixed effects", fontsize=20, weight="bold", fontname = 'Arial' )
                    axes = axes.flatten()
                    views = ["lateral","lateral","medial","medial", [-90, 90], [-90, 90]] #"ventral","ventral","anterior","anterior"]
                    hemis = ["right", "left"]*(len(views)//2)
                    brain_shape = ["inflated"]*4 + ["pial"]*2
                    
                    # set a unified colorbar for both hemis
                    vmin = np.min([np.nanmin(fe_stat.data.parts['left']),np.nanmin(fe_stat.data.parts['right'])])
                    vmax = np.max([np.nanmax(fe_stat.data.parts['left']),np.nanmax(fe_stat.data.parts['right'])])
                    
                    vmin = -1*np.max([-1*vmin,vmax])
                    vmax = np.max([-1*vmin,vmax])
                    
                    suf = "vol" if isVolume else "surf"
                    for i in range(len(views)):
                        plot_surf_stat_map(
                                surf_mesh=fsaverage[brain_shape[i]], 
                                stat_map=fe_stat,
                                hemi=hemis[i],
                                view = views[i],#“lateral”, “medial”, “dorsal”, “ventral”, “anterior”, “posterior”
                                colorbar=True if i==1 else False,
                                cmap='cold_hot',  #default:'cold_hot'
                                axes=axes[i],
                                threshold=4.0,
                                vmin=vmin, 
                                vmax=vmax,
                                bg_map=fsaverage_data)
    
                    pos = axes[4].get_position() 
                    axes[4].set_position([pos.x0 + 0.14, pos.y0, pos.width, pos.height]) #move to the right
                    pos = axes[5].get_position() 
                    axes[5].set_position([pos.x0 - 0.14, pos.y0, pos.width, pos.height]) #move to the left


                    plt.close(fig)


                # Save stat map 
                if isVolume:
                    nib.save(fe_contrast, f"{output_dir_vol}{sub}_{block_type}_blocks_{contrast}_fe_beta_map.nii.gz")
                    nib.save(fe_stat, f"{output_dir_vol}{sub}_{block_type}_blocks_{contrast}_fe_t_map.nii.gz")


                else:
                    nib.save(fe_contrast.data,f"{output_dir_surf}{sub}_{block_type}_blocks_{contrast}_fe_beta_map.gii")
                    nib.save(fe_stat.data, f"{output_dir_surf}{sub}_{block_type}_blocks_{contrast}_fe_t_map.gii")
                del contrast_imgs,variance_imgs


