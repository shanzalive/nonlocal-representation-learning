#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Visual check of the OFC masks: plots every participant's functional OFC mask
# (after the dropout threshold) on their T1 image, as a gallery.
#
# Input:  ROI masks from a_extract_ROI_signals.py, fMRIPrep T1 images
# Output: OFC_ROIs_gallery_batch<N>.png
#
# Author: Shany Grossman

import os
import glob
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import plotting, image
from matplotlib.colors import LinearSegmentedColormap
from nilearn.image import new_img_like
from nilearn.plotting import find_xyz_cut_coords
from nilearn.image import math_img
import numpy as np
# === Set your paths ===
mask_folder = "/home/mpib/grossman/STATE_FORMATION/MRI_DATA/MRI_REAL/ROI_timeseries_LSS_5mm_smoothing_dropout_30_physio_reg_out_updated/"  # Update this
parent_anat_folder = "/home/mpib/grossman/STATE_FORMATION/MRI_DATA/MRI_REAL/derivatives/"  # Assuming you have native T1s named like sub-XXX_T1w.nii.gz


# === Load all OFC functional mask files ===
mask_paths = sorted(glob.glob(os.path.join(mask_folder, "sub-*OFC_func_mask.nii.gz")))


# === Setup figure ===
n_subs = 6 #len(mask_paths)
n_cols = 2
n_rows = (n_subs + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
axes = axes.flatten()
for ifig in range(int(np.ceil(len(mask_paths)//6))):
    cbatch_mask_paths = mask_paths[ifig*6:(ifig+1)*6]
    for i, mask_path in enumerate(cbatch_mask_paths):
        subj_id = os.path.basename(mask_path).split('_')[0]
        print(subj_id)
    
        # === Load images ===
      
        anat_path = os.path.join(parent_anat_folder, subj_id, 'anat', subj_id + '_desc-preproc_T1w.nii.gz')
        anat_img = nib.load(anat_path)
    
        func_mask_img = nib.load(mask_path)
        
        # Load anatomical OFC mask for this subject
        anat_mask_path = os.path.join(mask_folder, f"{subj_id}_OFC_anat_mask.nii.gz") 
        if not os.path.exists(anat_mask_path):
            print(f"Missing anatomical mask for {subj_id}")
            continue
        
        anat_mask_img = nib.load(anat_mask_path)
    
        # === Make sure the mask is float and has values > 0 ===
        func_mask_float = image.new_img_like(func_mask_img, (func_mask_img.get_fdata() > 0).astype(float))
        anat_mask_float = image.new_img_like(anat_mask_img, (anat_mask_img.get_fdata() > 0).astype(float))
    
        # Difference: func-only voxels (not in anat)
        diff_img = math_img("img1 - img2", img1=func_mask_float, img2=anat_mask_float)
        
        # === Find center of mass of the functional mask ===
        cut_coords = find_xyz_cut_coords(diff_img)
        
        # Plot
        display = plotting.plot_anat(anat_img, title=subj_id, display_mode='ortho',
                                     cut_coords=cut_coords, dim=0.6, black_bg=True,  draw_cross=False)
        
        display.add_overlay(anat_mask_float, cmap='Reds_r', alpha=1)
        display.add_overlay(func_mask_float, cmap='brown_cyan_r', alpha = 0.5)


        # Save to temporary image
        tmpfile = os.path.join(mask_folder, f"tmp_{subj_id}.png")
        display.savefig(tmpfile)
        display.close()
    
        # Show in subplot
        img = plt.imread(tmpfile)
        axes[i].imshow(img)
        axes[i].axis('off')
    
    # Hide any unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.savefig(mask_folder + 'OFC_ROIs_gallery_batch' + str(ifig) + '.png')

# Optional: remove temporary PNGs
for mask_path in mask_paths:
    subj_id = os.path.basename(mask_path).split('_')[0]
    tmpfile = os.path.join(mask_folder, f"tmp_{subj_id}.png")
    if os.path.exists(tmpfile):
        os.remove(tmpfile)

