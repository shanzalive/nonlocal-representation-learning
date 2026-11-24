#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# heudiconv heuristic for sub-507 only: DICOM -> BIDS conversion.
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
    task_2nd = create_key('sub-{subject}/{session}/func/sub-{subject}_{session}_task-sf_run-0302_bold') #for case where MRI stopped at the middle of the block and we reran it: e.g. 542 ses2run2 
    fmap_topup = create_key('sub-{subject}/{session}/fmap/sub-{subject}_{session}_dir-{dir}_epi')    
    info = {anat: [], task: [], fmap_topup: [], task_2nd: []}
    last_run = len(seqinfo)
    for s in seqinfo:
        print(s.series_id)
        if 'T1w' in s.series_id: 
            info[anat].append({'item': s.series_id})
        if 'NORM' in s.image_type: 
            if ('dir-AP' in s.series_id): info[fmap_topup].append({'item': s.series_id, 'dir': 'AP'})
            if ('dir-PA' in s.series_id): info[fmap_topup].append({'item': s.series_id, 'dir': 'PA'})	  
            if ('run' in s.series_id):
                if ('9-func_ses-01_task-pm_acq-MB4_run-03' in s.series_id): 
                    info[task_2nd].append({'item': s.series_id})
                    print(s.series_id)
                else:
                    info[task].append({'item': s.series_id})
    return info

