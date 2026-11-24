#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Group maps (second-level GLM) of the visual localizer.
#
# For each localizer contrast ('all_cat>scrambled_objects' for VTC, 'faces>scenes' for FFA), takes
# every participant's t map in MNI space and fits a one-sample group model (nilearn SecondLevelModel,
# 5 mm smoothing). The group z map is saved, then only its positive values inside the same
# FreeSurfer anatomical labels that bound the participant-level ROIs (a_extract_ROI_signals.py),
# and this masked map is projected onto the fsaverage surface (FreeSurfer mri_vol2surf) for display.
#
# Input:  participant t maps in MNI space from runGLM_nilearn.py with GLM_model = 'standard',
#         run_types_2_include = ['visloc_categories'] and isNative = False
#         (nilearn_outputs/vol_maps/visloc_categories_physio_regressed_out_updated/<sub>_<contrast>_t_map_standard.nii.gz),
#         summary_files_behaviour/learners_dict.json (participants: full + partial learners, N = 51)
# Output: nilearn_outputs/<contrast>.nii (group z map), <contrast>_positive_only.nii and
#         lh/rh.<contrast>_positive_only.mgh (surface projection)
# Requires FreeSurfer (mri_vol2surf; version 7.4.1) for the surface projection.
#
# Author: Shany Grossman

import glob
import json
import os
import re
import subprocess
import numpy as np
import pandas as pd
import nilearn
from nilearn import image
import nibabel as nib 
from nilearn.glm import threshold_stats_img
from nilearn.glm.first_level import FirstLevelModel
from nilearn.plotting import plot_design_matrix
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.glm.contrasts import compute_fixed_effects
from nilearn import plotting
from nilearn.plotting import plot_stat_map, plot_surf_stat_map
from nilearn.image import mean_img   
from nilearn.glm.second_level import SecondLevelModel, non_parametric_inference
from nilearn.datasets import fetch_surf_fsaverage,load_fsaverage,load_fsaverage_data
from nilearn.surface import load_surf_data, SurfaceImage, PolyData, PolyMesh
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


#%%set basic flags:
isVolume = True
isVisloc = True

# GUI-launched interpreters (e.g. Spyder) don't inherit the login shell's PATH, so
# `mri_vol2surf` isn't found by subprocess.run() even though it works fine from a terminal.
# Build an explicit env for those calls instead of relying on inherited PATH.
FREESURFER_HOME_LOCAL = '/Applications/freesurfer/7.4.1'
freesurfer_env = os.environ.copy()
freesurfer_env.setdefault('FREESURFER_HOME', FREESURFER_HOME_LOCAL)
freesurfer_env.setdefault('SUBJECTS_DIR', FREESURFER_HOME_LOCAL + '/subjects')
freesurfer_env['PATH'] = FREESURFER_HOME_LOCAL + '/bin:' + freesurfer_env.get('PATH', '')

#%% Set paths
base_dir = '/Users/grossman/Volumes/'
if not isVolume:
    stat_maps_dir = base_dir+ 'STATE_FORMATION/MRI_DATA/MRI_REAL/nilearn_outputs/surf_maps/'
else:
    # The physio-regressed visual localizer maps, the same folder as func_roi_dir in
    # a_extract_ROI_signals.py, which defines the per-subject FFA/VTC ROIs.
    stat_maps_dir = base_dir+ 'STATE_FORMATION/MRI_DATA/MRI_REAL/nilearn_outputs/vol_maps/visloc_categories_physio_regressed_out_updated/'

output_dir = base_dir + 'STATE_FORMATION/MRI_DATA/MRI_REAL/nilearn_outputs/'

# Participants: the full + partial learners of learners_dict.json (N = 51), as in all analyses.
# Folder with the participant-level behavioural summary files, found by searching upwards
# from this script (works on the cluster, on a mounted drive and in the code repository).
def _find_summary_files_behaviour():
    d = os.path.dirname(os.path.abspath(__file__))
    while not os.path.isdir(os.path.join(d, 'summary_files_behaviour')):
        if os.path.dirname(d) == d:
            raise FileNotFoundError('summary_files_behaviour/ not found above ' + __file__)
        d = os.path.dirname(d)
    return os.path.join(d, 'summary_files_behaviour') + os.sep
SUMMARY_DIR = _find_summary_files_behaviour()
with open(SUMMARY_DIR + 'learners_dict.json') as _f:
    _learners_dict = json.load(_f)
INCLUDED_SUBJECTS = set(_learners_dict['full_and_partial'])


#%% After looping over subjects, we are ready for generating 2nd level group maps:

second_level_glm = SecondLevelModel(smoothing_fwhm=5.0, n_jobs=2) #n_jobs: number of CPUs to use to do the computation. -1 means ‘all CPUs’
if not isVisloc:
    second_level_contrasts = ['active_blocks_face_active>houses_active','active_blocks_response active',
                                'active_blocks_face_passive>house_passive', 
                                'active_blocks_client_active', 'active_blocks_client_passive',
                                'active_blocks_house_active', 'active_blocks_house_passive',
                                'active_blocks_reward', 'active_blocks_response active', 
                                'active_blocks_response passive', 'active_blocks_passive_probe',
                                'passive_blocks_client_passive','passive_blocks_house_passive', 
                                'passive_blocks_response passive', 'passive_blocks_passive_probe','passive_blocks_face_passive>house_passive']
else:
    second_level_contrasts = ['all_cat>scrambled_objects', 'faces>scenes'] #'all_visual', 'objects>scrambled_objects']
    
if not isVolume:
    # Load the fsaverage template #this cntaims path to the different gii files 
    fsaverage = load_fsaverage(mesh='fsaverage')
    fsaverage_data = load_fsaverage_data(mesh='fsaverage', mesh_type='pial', data_type="sulcal")


plt.rcParams.update({
    "figure.facecolor":  (1.0, 1.0, 1.0, 0),  # red   with alpha = 30%
    "axes.facecolor":    (1.0, 1.0, 1.0, 0),  # green with alpha = 50%
    "savefig.facecolor": (1.0, 1.0, 1.0, 0),  # blue  with alpha = 20%
})

for icontrast, contrast in enumerate(second_level_contrasts):
    # Load single subjects' stat maps 
    if not isVolume:
        stat_map_files_rh = glob.glob(stat_maps_dir + '*' +  contrast +  '*t_map_hemi-R.gii')
        stat_map_files_lh = glob.glob(stat_maps_dir + '*' +  contrast +  '*t_map_hemi-L.gii')
        assert(len(stat_map_files_rh)==len(stat_map_files_lh))
        nsubj = len(stat_map_files_rh)
       
    else: #volume
        if isVisloc:
            stat_map_files = glob.glob(f"{stat_maps_dir}/**/sub*{contrast}_t_map_standard.nii.gz", recursive=True)
            stat_map_files = [f for f in stat_map_files
                               if re.search(r'sub-\d+', f).group(0) in INCLUDED_SUBJECTS]
            nsubj = len(stat_map_files)

    print('found ' + str(nsubj) + ' subjects for contrast ' + contrast)
        
    
    # Set 2nd level design matrix
    design_matrix = pd.DataFrame([1] * nsubj, columns=["intercept"])
    
    # Load stat maps into a list of SurfaceImages
    if not isVolume:
        ss_maps = []
        for i in range(nsubj):
            lh_map = load_surf_data(stat_map_files_lh[i])
            rh_map = load_surf_data(stat_map_files_rh[i])
            ss_maps.append(SurfaceImage(mesh=fsaverage["pial"], data = PolyData(left=lh_map[:, np.newaxis], right=rh_map[:, np.newaxis]))) 
            del lh_map, rh_map
    else: #volume
        ss_maps = stat_map_files
        
        
    second_level_glm.fit(second_level_input=ss_maps, design_matrix=design_matrix)
    results = second_level_glm.compute_contrast("intercept", output_type="z_score")
    
    
    #%% Plotting
    
    if not isVolume:
        # Set up a Matplotlib figure for side-by-side plots
        plt.ioff() 
        fig, axes = plt.subplots(3, 2, subplot_kw={"projection": "3d"}, figsize=(10, 12))
        fig.tight_layout(pad=0, h_pad=0, w_pad=0) 
        fig.subplots_adjust(wspace=0.01)
        fig.suptitle(f"{str(nsubj)} subjects - {contrast}", fontsize=20, weight="bold", fontname = 'Arial' )
        axes = axes.flatten()
        views = ["lateral","lateral","medial","medial", [-90, 90], [-90, 90]] #"ventral","ventral","anterior","anterior"]
        hemis = ["right", "left"]*(len(views)//2)
        brain_shape = ["inflated"]*4 + ["inflated"]*2
        
        # set a unified colorbar for both hemis
        vmin = np.min([np.nanmin(results.data.parts['left']),np.nanmin(results.data.parts['right'])])
        vmax = np.max([np.nanmax(results.data.parts['left']),np.nanmax(results.data.parts['right'])])
        
        vmin = -1*np.max([-1*vmin,vmax])
        vmax = np.max([-1*vmin,vmax])
        
        suf = "vol" if isVolume else "surf"
        for i in range(len(views)):
            plot_surf_stat_map(
                    surf_mesh=fsaverage[brain_shape[i]], 
                    stat_map=results,
                    hemi=hemis[i],
                    view = views[i],#“lateral”, “medial”, “dorsal”, “ventral”, “anterior”, “posterior”
                    colorbar=True if i==1 else False,
                    cmap='cold_hot',  #default:'cold_hot'
                    axes=axes[i],
                    threshold=2.58, #p=0.01
                    vmin=vmin, 
                    vmax=vmax,
                    bg_map=fsaverage_data)
        hemi_space = [0.03,0.03,0.17]
        for r in range(3):
                pos = axes[r*2].get_position() 
                axes[r*2].set_position([pos.x0 + hemi_space[r], pos.y0, pos.width, pos.height]) #move to the right
                pos = axes[r*2+1].get_position() 
                axes[r*2+1].set_position([pos.x0 - hemi_space[r], pos.y0, pos.width, pos.height]) #move to the left
       
        
        output_file = f"{output_dir}GroupMap_{contrast}_{suf}.png"
        plt.savefig(output_file,format='png',dpi=300)
        plt.close(fig)
    else:

        results.to_filename(output_dir + contrast + '.nii')
        
        # save only positive values:
        # Load data array from the results image
        data = results.get_fdata()
        
        # Set negative values to 0 (or np.nan if you prefer masking)
        data[data <= 0] = 0
        
        # Create a new Nifti image with the thresholded data
        positive_only_img = nib.Nifti1Image(data, affine=results.affine, header=results.header)
        
        
        
        # now add the anatomical mask (FreeSurfer labels in MNI space; one participant's
        # aparcaseg image in MNI space is used only as the label template)
        # Load freesurfer aparcaseg atlas in native space from all func runs
        fs_aparcaseg=[]
        pathname =  base_dir + 'STATE_FORMATION/MRI_DATA/MRI_REAL/derivatives/sub-511/ses-1/func/'
        fs_aparcaseg_img = nilearn.image.load_img(pathname + 'sub-511_ses-1_task-sf_run-05_space-MNI152NLin6Asym_desc-aparcaseg_dseg.nii.gz', wildcards=True, dtype=None)
        fs_aparcaseg_data = image.get_data(fs_aparcaseg_img)
        
        # Anatomical label sets copied verbatim from a_extract_ROI_signals.py's roi_dict, keyed
        # by contrast so each group map uses exactly the same anatomical bound as the matching
        # per-subject functional ROI (FFA/PPA/LO/VISUALFUNC_HIGHORDER all share one set there;
        # VISUALFUNC/all_visual uses a wider one).
        contrast_to_roi_labels = {
            'faces>scenes':               [1009,2009,1007,2007,1011,2011,1016,2016,1013,2013],  # FFA/PPA
            'objects>scrambled_objects':  [1009,2009,1007,2007,1011,2011,1016,2016,1013,2013],  # LO
            'all_cat>scrambled_objects':  [1009,2009,1007,2007,1011,2011,1016,2016,1013,2013],  # VISUALFUNC_HIGHORDER (VTC)
            'all_visual':                 [1005,2005,1007,2007,1009,2009,1011,2011,1013,2013,1021,2021,1029,2029,1016,2016],  # VISUALFUNC
        }
        ROIs_idx = contrast_to_roi_labels[contrast]

        # Define ROI masks based on the union of brain_mask and relevant freesurfer labels across all runs
        roi_mask = np.isin(fs_aparcaseg_data, ROIs_idx)
            
        data = positive_only_img.get_fdata()
        masked_data = np.zeros_like(data)
        masked_data[roi_mask] = data[roi_mask]
            
        masked_img = nib.Nifti1Image(masked_data, affine=positive_only_img.affine, header=positive_only_img.header)
        masked_img.to_filename(output_dir + contrast + '_positive_only.nii')

        # Also project the masked group map onto fsaverage pial, for the FreeView composite
        # figure (OFC/HPC/FFA/VTC together). --mni152reg is FreeSurfer's built-in MNI152->
        # fsaverage mapping, so this needs no per-subject registration: it goes straight from
        # the MNI-standard-space volume saved above to lh/rh.<contrast>_positive_only.mgh.
        if isVisloc:
            vol_file = output_dir + contrast + '_positive_only.nii'
            for hemi in ['lh', 'rh']:
                surf_file = output_dir + f'{hemi}.{contrast}_positive_only.mgh'
                subprocess.run([
                    'mri_vol2surf',
                    '--mov', vol_file,
                    '--mni152reg',
                    '--hemi', hemi,
                    '--projfrac-avg', '0', '1', '0.1',
                    '--o', surf_file,
                ], check=True, env=freesurfer_env)
