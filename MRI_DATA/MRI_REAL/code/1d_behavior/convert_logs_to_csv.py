#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Take mat log files of the State Formation experiment and convert them into a single csv that can be used for analyses  
#
# Input:  task logs of each participant (MRI_REAL/LOGs/)
# Output: summary_files_behaviour/State_Formation_behaviour.csv
#         (one row per trial: input features, state, response, correct response,
#         values, reaction time, ...). All later analyses read this file.
#
# Notes on participants with stopped / repeated runs are listed below.
#
# Author: Shany Grossman

## Subject with double/missing logs for stopped/skipped runs - in the formaer case consider simply concatnate their logs (but consider this is noisy... some repeated trials):
# 507_1_3 (?? TRs) --> "1_3_0_..stopped_run_log"
# 535_3_4 (41 Trs) --> "3_4_0_..stopped_run_log", didn't learn
# 539_2_2 (64 Trs) --> "2_2_0_..stopped_run_log", didn't learn
# 542_2_2 (150 Trs) --> "2_2_0_..stopped_run_log"
# 544_1_3 (41 Trs) --> "1_3_0_..stopped_run_log"
# 553_1_5 (50 Trs) --> "1_5_0_..stopped_run_log"
# 553_3_5 (?? Trs) --> "3_5_0_..stopped_run_log"
# 554_1_7 (79 Trs) --> "1_7_0_..stopped_run_log"
# 520 did not do active block 6 (skipped from 5 to 7 due to lack of time)   

import os
import numpy as np
import glob
import pandas as pd
from scipy.io import loadmat

# Path ot where logs are (after being converted into CSVs!)
log_dir = '../../LOGs/'

def _find_summary_files_behaviour():
    d = os.path.dirname(os.path.abspath(__file__))
    while not os.path.isdir(os.path.join(d, 'summary_files_behaviour')):
        if os.path.dirname(d) == d:
            raise FileNotFoundError('summary_files_behaviour/ not found above ' + __file__)
        d = os.path.dirname(d)
    return os.path.join(d, 'summary_files_behaviour') + os.sep
SUMMARY_DIR = _find_summary_files_behaviour()

save_dir = SUMMARY_DIR

# Get all subject names
subjx = np.sort(os.listdir(log_dir))
subj = [subjx[x] for x in range(len(subjx)) if 'sub' in subjx[x]]
subj = [subjx[x] for x in range(len(subjx)) if 'sub' in subjx[x] and not subjx[x].startswith('.')]
print(str(len(subj)) + ' particpants'' log folders found')

# Init df to collect single trial data from all participants 
df_logs = pd.DataFrame(columns=['sub', 'ses', 'block_type', 'block_type_num', 'block_num', 'is_passive', 'input',
                                'state', 'response', 'correct_response', 'value_left', 'value_right',
                                'value_single_house','single_house_prob_value'])
df_logs['input'] = df_logs['input'].astype(int)

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

#%% Loop over participants 
for isub, sub in enumerate(subj):
    
    # Get names of all log
    log_files = np.sort(glob.glob(log_dir + sub + '/' + sub[3:] + '*_log.csv'))
    print('found ' + str(len(log_files)) + ' logs for ' + sub +'\n' )
    
    for ilog, log in enumerate(log_files):
        if 'active' in log:
            c_log =  pd.read_csv(log)
            c_ses = int(log.split('/')[-1][4])
            c_block_type = 'active'
            
            for j in range(c_log.shape[0]):
                if c_log.eventType[j]=='client onset':
                
                    c_block = int(log.split('/')[-1].split('_')[2])
                    c_block_type_num = int(log.split('/')[-1].split('active')[1].split('_')[0])
                    # c_input # the key is - as in the simulations - 16 binary inputs: clientf1(x2) clientf2(x2) lhousef1(X2) lhousef2(X2) lhousef3(X2) rhousef1(X2) rhousef2(X2) rhousef3(X2)
                    # from one hot to sign: 1 0 --> 1; 0 1 --> -1
                    c_input = np.zeros(16, 'int')
                    if c_log.client_img[j].split('_')[1]=='1': c_input[0]=1 
                    else: c_input[1]=1
                    
                    if c_log.client_img[j].split('_')[2][0]=='1': c_input[2]=1 
                    else: c_input[3]=1
                    
                    if c_log.is_passive[j]: #passive trial --> fill in single house params on the right 
                    
                        client_img, phouse_img = c_log.client_img[j], c_log.passive_house_img[j+1]
                        state = convert_input2single_house_state(client_img, phouse_img)
                        RT = np.nan
                    
                        if c_log.passive_house_f1[j+1]==1: c_input[4]=1 
                        else: c_input[5]=1
                        
                        if c_log.passive_house_f2[j+1]==1: c_input[6]=1 
                        else: c_input[7]=1
                        
                        if c_log.passive_house_f3[j+1]==1: c_input[8]=1 
                        else: c_input[9]=1
                        
                        #log passive house value 
                        c_value_passive = c_log.passive_house_reward[j+1]
                        
                        # nan in active house values:
                        c_value_left = np.nan
                        c_value_right = np.nan
                        
                        # init nans in resp related field and update later in case it was a probe trial
                        c_resp = np.nan
                        c_cor_resp = np.nan
                        c_prob_th = np.nan
                        
                        if sub=='sub501' or sub=='sub502':
                            if len(c_log)>j+2:
                                if c_log.eventType[j+2]=='response': #passive trial with prompt - log th_value and response
                                    c_resp = c_log.response[j+2]
                                    c_cor_resp = c_log.correctResp[j+1]
                                    c_prob_th = 50
                                    RT = c_log.RT[j+2]

                        else:
                            if len(c_log)>j+3:
                                if c_log.eventType[j+3]=='response':  #passive trial with prompt - log th_value and response
                                    c_resp = c_log.response[j+3]
                                    c_cor_resp = c_log.correctResp[j+1]
                                    c_prob_th = c_log.probe_th[j+2]
                                    RT = c_log.RT[j+3]


                    else: #active trial--> fill in both 
                    
                        client_img,  lhouse_img, rhouse_img = c_log.client_img[j], c_log.lhouse_img[j+1], c_log.rhouse_img[j+1]
                        state = convert_input2two_house_state(client_img,  lhouse_img, rhouse_img)
                        RT = c_log.RT[j+2]
                    
                        if c_log.lhouse_f1[j+1]==1: c_input[4]=1 
                        else: c_input[5]=1
                        
                        if c_log.lhouse_f2[j+1]==1: c_input[6]=1 
                        else: c_input[7]=1
                        
                        if c_log.lhouse_f3[j+1]==1: c_input[8]=1 
                        else: c_input[9]=1
                        
                        if c_log.rhouse_f1[j+1]==1: c_input[10]=1 
                        else: c_input[11]=1
                        
                        if c_log.rhouse_f2[j+1]==1: c_input[12]=1 
                        else: c_input[13]=1
                        
                        if c_log.rhouse_f3[j+1]==1: c_input[14]=1 
                        else: c_input[15]=1
                        
                        #log active houses value 
                        c_value_left = c_log.lhouse_reward[j+1]
                        c_value_right = c_log.rhouse_reward[j+1]
                        c_value_passive = np.nan
                        
                        #log response and correct response
                        c_resp = c_log.response[j+2]
                        c_cor_resp = c_log.correctResp[j+1]
                        
                        # insert nan for passive related fields
                        c_prob_th = np.nan


                    df_logs = pd.concat([df_logs, pd.DataFrame({'sub':sub, 'ses':c_ses, 'block_type':c_block, 'block_type_num': c_block_type_num, 'block_num':c_block,
                                              'is_passive':c_log.is_passive[j],
                                              'input':[c_input], 
                                              'state': state,  
                                              'is_correct': 1 if c_resp==c_cor_resp else 0,
                                              'rt': RT, 
                                              'response':c_resp, 
                                              'correct_response':c_cor_resp, 
                                              'value_left':c_value_left, 'value_right':c_value_right, 'value_single_house': c_value_passive,
                                              'single_house_prob_value':c_prob_th,
                                              'face': client_img[-5]})],
                                              ignore_index = True)


# save df as a csv on tardis
df_logs.to_csv(save_dir + 'State_Formation_behaviour.csv', index=False)


