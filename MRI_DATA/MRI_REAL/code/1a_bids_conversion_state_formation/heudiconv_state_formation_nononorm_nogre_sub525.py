#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# heudiconv heuristic for sub-525 only: DICOM -> BIDS conversion.
#
# Same as heudiconv_state_formation_nononorm_nogre.py, adapted to this participant's
# scanner exceptions (e.g. a run that was stopped and repeated, or a second set of
# field maps after the participant left and re-entered the scanner).
#
# Author: Shany Grossman

import os

def create_key(template, outtype=('nii.gz',), annotation_classes=None):
	if template is None or not template:
		raise ValueError('Template must be a valid format string')
	return template, outtype, annotation_classes

def infotodict(seqinfo):
    anat = create_key('sub-{subject}/{session}/anat/sub-{subject}_{session}_T1w')
    task = create_key('sub-{subject}/{session}/func/sub-{subject}_{session}_task-sf_run-{item:02d}_bold')
    fmap_topup = create_key('sub-{subject}/{session}/fmap/sub-{subject}_{session}_dir-{dir}_run-1_epi')    
    fmap_topup_2nd = create_key('sub-{subject}/{session}/fmap/sub-{subject}_{session}_dir-{dir}_run-2_epi') #for cases of going out of the mri and coming back e.g. sub-525 ses1 and ses3 
    info = {anat: [], task: [], fmap_topup: [], fmap_topup_2nd: []}
    last_run = len(seqinfo)
    for s in seqinfo:
        print('whatsup ' + s.series_id)
        if 'T1w' in s.series_id: 
            info[anat].append({'item': s.series_id})
        if 'NORM' in s.image_type: 
            if ('dir-AP' in s.series_id): 
                if ('25-fmap_ses-01' in s.series_id) or ('29-fmap_ses-03' in s.series_id):
                    info[fmap_topup_2nd].append({'item': s.series_id, 'dir': 'AP'})
                else: 
                    info[fmap_topup].append({'item': s.series_id, 'dir': 'AP'})
                       
                
            if ('dir-PA' in s.series_id): 
                if ('23-fmap_ses-01' in s.series_id) or ('27-fmap_ses-03' in s.series_id):
                    info[fmap_topup_2nd].append({'item': s.series_id, 'dir': 'PA'})	
                else:
                    info[fmap_topup].append({'item': s.series_id, 'dir': 'PA'})	
            if ('run' in s.series_id): info[task].append({'item': s.series_id})
    return info



