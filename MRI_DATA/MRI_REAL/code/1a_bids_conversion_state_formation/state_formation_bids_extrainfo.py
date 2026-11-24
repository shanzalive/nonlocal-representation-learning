#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Adds extra information to the BIDS dataset after conversion.
#
# Updates the dataset description and the json sidecar files of the field maps and
# functional runs (needed by fMRIPrep / MRIQC).
# Run with state_formation_bids_extrainfo.sh.
#
# Author: Shany Grossman

# ======================================================================
# SCRIPT INFORMATION:
# ======================================================================
# SCRIPT: UPDATE OF BIDS DIRECTORY
# PROJECT: STATEFORMATION
# ======================================================================
# IMPORT RELEVANT PACKAGES
# chmod u+rwx -R .\
# RUN all script on cluster for the right file seperator    
# ======================================================================
import os.path as op
import glob
import json
import numpy as np
import os


sublist=["561", "562", "563", "564"]

for sub in sublist:
        print(sub)
        path_bids="/Users/grossman/Volumes/Tardis/STATE_FORMATION/MRI_DATA/MRI_REAL/BIDs/"
        path_fmap = op.join(path_bids,'*'+sub,'*','fmap','*.json')
        path_func = op.join(path_bids,'*' + sub,'*','func','*.nii.gz')
        path_desc = op.join(path_bids,'dataset_description.json')
        # ======================================================================
        # UPDATE DATA-SET DESCRIPTION FILE
        # ======================================================================
        # open the dataset_description.json file:
        with open(path_desc) as json_file:
            json_desc = json.load(json_file)
        # update fields of the json file:
        json_desc["Name"] = "STATE FORMATION EXPERIMENT"
        json_desc["Authors"] = ["Shany Grossman", "Nicolas W. Schuck"]
        
        # save updated data-set_description.json file:
        with open(path_desc, 'w') as outfile:
            json.dump(json_desc, outfile, indent=4)
        # ======================================================================
        # UPDATE FIELDMAP JSON FILES
        # ======================================================================
        # get all fieldmap files in the data-set:
        files_fmap = glob.glob(path_fmap)
        # loop over all field-map files:
        
        for file_path in files_fmap:
            with open(file_path,'r') as in_file:
                json_info = json.load(in_file)
            in_file.close()
            file_base = op.dirname(op.dirname(file_path))
            files_func = np.sort(glob.glob(op.join(file_base,'func','*nii.gz')))
            grandparent = op.basename(op.abspath(op.join(file_path,'../..')))
            parent = 'func'
            up_dirs = op.join(grandparent, parent)
            if sub=='525':
                if 'ses-1' in file_path:
                    if '2' in file_path:
                       files_func=files_func[-3:].copy()
                    else:
                        files_func=files_func[:-3].copy() 
                    
                elif 'ses-3' in file_path:
                    if '2' in file_path:
                       files_func=files_func[-5:] 
                    else:
                        files_func=files_func[:-5] 
                    
            elif (sub=='526') and ('ses-3' in file_path):
                    if '2' in file_path:
                       files_func=files_func[-4:] 
                    else:
                        files_func=files_func[:-4] 

            intended_for = [op.join(up_dirs, op.basename(file)) for file in files_func]
            json_info["IntendedFor"] = intended_for
            
            # add wiritn gpermission to fmap json file
            
            # save updated fieldmap json-file:
            os.chmod(file_path, 0o777)
            with open(file_path, 'w') as out_file:
                json.dump(json_info, out_file, indent=2)
                        
