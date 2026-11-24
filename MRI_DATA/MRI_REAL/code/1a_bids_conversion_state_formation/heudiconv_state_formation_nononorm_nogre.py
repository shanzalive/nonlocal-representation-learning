#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# heudiconv heuristic: DICOM -> BIDS conversion.
#
# Tells heudiconv which scanner series is the T1w image, which are the task runs
# and which are the field maps, and how to name them in BIDS.
# Used by heudiconv_state_formation_tardis.sh for all participants, except those
# with their own heuristic (heudiconv_state_formation_nononorm_nogre_subXXX.py).
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
    task_2nd = create_key('sub-{subject}/{session}/func/sub-{subject}_{session}_task-sf_run-{item:02d}_2nd_bold') #for case where MRI stopped at the middle of the block and we reran it: e.g. 542 ses2run2 
    fmap_topup = create_key('sub-{subject}/{session}/fmap/sub-{subject}_{session}_dir-{dir}_epi')    
    info = {anat: [], task: [], fmap_topup: [], task_2nd: []}
    last_run = len(seqinfo)
    for s in seqinfo:
        if 'T1w' in s.series_id: 
            info[anat].append({'item': s.series_id})
        if 'NORM' in s.image_type: 
            if ('dir-AP' in s.series_id): info[fmap_topup].append({'item': s.series_id, 'dir': 'AP'})
            if ('dir-PA' in s.series_id): info[fmap_topup].append({'item': s.series_id, 'dir': 'PA'})	
            if ('run' in s.series_id): info[task].append({'item': s.series_id})
    return info

