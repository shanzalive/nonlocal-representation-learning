#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Collects the co-update results of all participants (second level).
#
# Reads the output of c_coupdate_analysis.py for every participant, labels each state pair
# (same / different context, value difference, feedback, ...) and writes the group tables.
# Input:  <ROI folder>/coupdate_LSS/<sub>_coupdate_LSS_zscored.csv
# Output: coupdate_LSS_zscored_miniblock_level_coupdate_values.csv (one row per pair and mini-block;
#         used by the figure scripts for Figs. 4-6, S2-S4) and a subject-level summary csv.
# doMultiverse = True does the same for the robustness variants of Fig. S3.
#
# Author: Shany Grossman

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import stats
# Set higher DPI for better resolution
plt.rcParams['figure.dpi'] = 300  


analysis = 'coupdate_LSS' #'_procrustes' #'feature_selection' #'LSS' #'coupdate_LSS' # coupdate_feature_selection, coupdate
zscore_per_run = True  # if True: load z-scored per-run output files from c_coupdate_analysis

doMultiverse = False  # False = main pipeline (main text). True = robustness variants (Fig. S3): loop over the folders below
# Robustness variants of Fig. S3; each folder is an output of a_extract_ROI_signals.py with runMultiverse = True.
MULTIVERSE_UNIVERSES = [
    ('ROI_timeseries_LSS_5mm_smoothing_dropout_30_compcorr', 'coupdate_LSS'),                  # aCompCor instead of physio regressors
    ('ROI_timeseries_LSS_5mm_smoothing_dropout_50_physio_reg_out_updated', 'coupdate_LSS'),   # dropout threshold 50%
    ('ROI_timeseries_LSS_5mm_smoothing_dropout_80_physio_reg_out_updated', 'coupdate_LSS'),   # dropout threshold 80%
    ('ROI_timeseries_LSS_5mm_smoothing_physio_reg_out_updated', 'coupdate_LSS'),              # no dropout threshold
    ('ROI_timeseries_LSS_3mm_smoothing_dropout_30_physio_reg_out_updated', 'coupdate_LSS'),   # 3 mm smoothing
]

# Paths
base_path = '/Users/grossman/Volumes/STATE_FORMATION/'

if doMultiverse:
    base_dirs = [(base_path + 'MRI_DATA/MRI_REAL/' + d + '/' + a + '/', a) for d, a in MULTIVERSE_UNIVERSES]
elif 'LSS' in analysis:
    base_dirs = [(base_path + 'MRI_DATA/MRI_REAL/ROI_timeseries_LSS_5mm_smoothing_dropout_30_physio_reg_out_updated/' + analysis + '/', analysis)]
else:
    base_dirs = [(base_path + 'MRI_DATA/MRI_REAL/ROI_timeseries_zscore_5mm_smoothing_dropout_30/' + analysis + '/', analysis)]


# Fisher Z-transformation
def fisher_z_transform(r):
    return np.arctanh(r)  # Equivalent to 0.5 * np.log((1 + r) / (1 - r))

def inverse_fisher_z(z):
    return np.tanh(z)

# Updated aggregation functions with Fisher transformation
def compute_fisher_mean(x, condition_mask):
    valid_corrs = x[condition_mask & x.notnull()]  # Ensure valid correlations
    if len(valid_corrs) == 0:
        return np.nan
    fisher_transformed = fisher_z_transform(valid_corrs)
    return inverse_fisher_z(np.mean(fisher_transformed))

# A function that regress out mean temp dist between pre and post pairs and add a new corr column to the df:
def regress_out_temp_dist(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Drop rows with NaN in 'corr' or 'corr_ordinal' or 'pre_post_pair_temporal_dist'
    before = len(df)
    df = df.dropna(subset=['corr', 'corr_ordinal'])
    after = len(df)
    if before-after>0:
        print(f"{df.subject.values[0]} : Dropped {before - after} rows due to NaNs in 'corr' or 'corr_ordinal'.")
     
    #For pearson corr
    residuals_list = []
    
    for roi, group in df.groupby('ROI'):
        # Design matrix with intercept
        X = np.vstack([group['pre_post_pair_temporal_dist'].values, np.ones(len(group))]).T
        y = group['corr'].values
    
        # Least squares regression
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_pred = X @ beta
        residuals = y - y_pred
    
        # Append residuals back with the original index
        residuals_list.append(pd.Series(residuals, index=group.index))

    # Concatenate and assign to new column
    df['corr_temp_dist_regressed_out'] = pd.concat(residuals_list).sort_index()
    
    #For spearm corr
    residuals_list_rho, residuals_list_r = [],[]
    
    for roi, group in df.groupby('ROI'):
        # Design matrix with intercept
        X = np.vstack([group['pre_post_pair_temporal_dist'].values, np.ones(len(group))]).T
        y = group['corr_ordinal'].values
    
        # Least squares regression
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_pred = X @ beta
        residuals = y - y_pred
    
        # Append residuals back with the original index
        residuals_list_rho.append(pd.Series(residuals, index=group.index))
        
        # Repeat for pearson corr:
        y = group['corr'].values
    
        # Least squares regression
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_pred = X @ beta
        residuals = y - y_pred
    
        # Append residuals back with the original index
        residuals_list_r.append(pd.Series(residuals, index=group.index))


    # Concatenate and assign to new column
    df['corr_ordinal_temp_dist_regressed_out'] = pd.concat(residuals_list_rho).sort_index()
    df['corr_temp_dist_regressed_out'] = pd.concat(residuals_list_r).sort_index()
    
    return df

# A function that rgresses out the temporal distances betwen pre/post pairs on which we compute simple correlations and then subtract to get a context speraiton measure per pair
def regress_out_temp_dist_conttext_separation(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Drop rows with NaN in 'corr' or 'corr_ordinal' or 'pre_post_pair_temporal_dist'
    before = len(df)
    df = df.dropna(subset=['corr', 'corr_ordinal'])
    after = len(df)
    if before-after>0:
        print(f"{df.subject.values[0]} : Dropped {before - after} rows due to NaNs in 'corr' or 'corr_ordinal'.")

    # Set empty lists
    residuals_list_r = []
    
    for roi, group in df.groupby('ROI'):
        # Design matrix with intercept
        X = np.vstack([group['temporal_dist_post'].values, np.ones(len(group))]).T
        y = group['post_corr'].values
    
        # Least squares regression
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_pred = X @ beta
        residuals_post = y - y_pred
        
        # Design matrix with intercept
        X = np.vstack([group['temporal_dist_pre'].values, np.ones(len(group))]).T
        y = group['pre_corr'].values
    
        # Least squares regression
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_pred = X @ beta
        residuals_pre = y - y_pred

        # Append residuals back with the original index
        residuals_list_r.append(pd.Series(residuals_post-residuals_pre, index=group.index))


    # Concatenate and assign to new column
    df['context_seperation_temp_dist_regressed_out'] = pd.concat(residuals_list_r).sort_index()
    
    return df


# for regressing out both pre and post:
def regress_out_pre_corr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Regress out both pre_corr and post_corr from 'corr' per subject and ROI.
    Adds a new column 'corr_pre_post_sim_corrected' containing residuals.
    """
    df = df.copy()
    
    # Drop rows with NaN in relevant columns
    before = len(df)
    df = df.dropna(subset=['corr', 'pre_corr', 'post_corr'])
    after = len(df)
    if before - after > 0:
        print(f"{df.subject.values[0]} : Dropped {before - after} rows due to NaNs in 'corr', 'pre_corr', or 'post_corr'.")
    
    residuals_list = []
    
    # Group by subject and ROI
    for (subject, roi), group in df.groupby(['subject', 'ROI']):
        if len(group) < 2 or (np.std(group['pre_corr']) < 1e-8 and np.std(group['post_corr']) < 1e-8):
            # fallback: subtract mean if too few points or no variance
            residuals_list.append(pd.Series(group['corr'] - np.mean(group['corr']), index=group.index))
            continue
        
        # Design matrix: [pre_corr, post_corr, intercept]
        X = np.vstack([group['pre_corr'].values, group['post_corr'].values, np.ones(len(group))]).T
        y = group['corr'].values
        
        try:
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            y_pred = X @ beta
            residuals = y - y_pred
            residuals_list.append(pd.Series(residuals, index=group.index))
        except np.linalg.LinAlgError:
            # fallback: subtract mean if regression fails
            residuals_list.append(pd.Series(y - np.mean(y), index=group.index))
    
    df['corr_pre_sim_corrected'] = pd.concat(residuals_list).sort_index()
    
    return df
#%%

for base_dir, analysis in base_dirs:
    print(f'--- universe: {base_dir} (analysis={analysis}) ---')
    if 'coupdate_trials_df' in dir():
        del coupdate_trials_df

    # Get all subjects for whom the df with coupdates was computed:
    _zscore_suffix = '_zscored' if zscore_per_run else ''
    files = np.sort(glob.glob("%s/sub-5[0-9][0-9]_%s%s.csv"%(base_dir, analysis, _zscore_suffix)))
    subjects = [s.split('/')[-1][:7] for s in files]
    print(f"Found {len(subjects)} subjects from CSVs: {subjects}")

    # Initialize an empty dataframe with specified columns to store mean effects across all learning period
    columns = ['subject', 'ROI', 'corr_var_name',
               'corr_diff_context',
               'corr_same_context',
               'corr_diff_context_valdiff_100',
               'corr_diff_context_valdiff_75',
               'corr_diff_context_valdiff_25',
               'corr_diff_context_valdiff_50',
               'corr_diff_context_valdiff_0',
               'corr_diff_context_valdiff_12',
               'corr_diff_context_valdiff_62',
               'corr_diff_context_valdiff_37',
               'corr_same_context_valdiff_100',
               'corr_same_context_valdiff_75',
               'corr_same_context_valdiff_25',
               'corr_same_context_valdiff_50',
               'corr_same_context_valdiff_0',
               'corr_same_context_valdiff_12',
               'corr_same_context_valdiff_62',
               'corr_same_context_valdiff_37',
               'pre_post_pair_temporal_dist_diff_context',
               'pre_post_pair_temporal_dist_same_context']  #12==13, 38==37

    # Define the dtypes for the columns
    dtypes = {
        'subject': 'str',
        'ROI': 'str',
        'corr_var_name': 'str',
        'corr_diff_context': 'float64',
        'corr_same_context': 'float64',
        'corr_diff_context_valdiff_100': 'float64',
        'corr_diff_context_valdiff_75': 'float64',
        'corr_diff_context_valdiff_25': 'float64',
        'corr_diff_context_valdiff_50': 'float64',
        'corr_diff_context_valdiff_0': 'float64',
        'corr_diff_context_valdiff_12': 'float64',
        'corr_diff_context_valdiff_62': 'float64',
        'corr_diff_context_valdiff_37': 'float64',
        'corr_same_context_valdiff_100': 'float64',
        'corr_same_context_valdiff_75': 'float64',
        'corr_same_context_valdiff_25': 'float64',
        'corr_same_context_valdiff_50': 'float64',
        'corr_same_context_valdiff_0': 'float64',
        'corr_same_context_valdiff_12': 'float64',
        'corr_same_context_valdiff_62': 'float64',
        'corr_same_context_valdiff_37': 'float64',
        'pre_post_pair_temporal_dist_diff_context': 'float64',
        'pre_post_pair_temporal_dist_same_context': 'float64'}

    group_mean_df = pd.DataFrame(columns=columns).astype(dtypes) 


    #%% Load all single subjects' CSVs and conctanate the means of the conditions 
    sub_counter = 0
    for sub, csv_path in zip(subjects, files):
        if True:
            df = pd.read_csv(csv_path) 
            df = df[df['runType'].str.extract(r'(\d+)', expand=False).astype(int)>0] #for filtering specific periods (i.e. blocks) of learning 

            # --- Check for presence of all active blocks ---
            expected_active_blocks = [f'active{i}' for i in range(1, 12)]
            present_blocks = df['runType'].unique()
            missing_blocks = [block for block in expected_active_blocks if block not in present_blocks]

            if missing_blocks:
                print(f"{sub} is missing blocks: {missing_blocks}")

            df['same_value'] = (df['value_state_from'] == df['value_state_to']).astype(int)
            df['same_house_type'] = (df['state_from'].str[1] == df['state_to'].str[1]).astype(int) 
            df['value_diff'] = (df['value_state_from'] - df['value_state_to']).abs()
            df['value_diff'] = df['value_diff'].fillna(np.nan).astype('Int64')  # pandas nullable int type
            df['block'] = df['runType'].str.extract(r'(\d+)', expand=False).astype(int)
            df = df[df['runType'].str.extract(r'(\d+)', expand=False).astype(int)<200]#for filtering specific periods (i.e. blocks) of learning 
            df.loc[df['value_diff'] == 13, 'value_diff'] = 12
            df.loc[df['value_diff'] == 38, 'value_diff'] = 37
            df.loc[df['value_diff'] == 63, 'value_diff'] = 62

            df['pre_post_pair_temporal_dist'] = df[['temporal_dist_pre', 'temporal_dist_post']].mean(axis=1)
            df = df[df['isMean']==0].copy()

            #regress out pre_post mean temporal dist
            df = regress_out_temp_dist(df)

            #regress out pre_corr and get a corr_pre_sim_corrected column
            df = regress_out_pre_corr(df)

            # Regress out remporal dist for context sepration measure 
            df = regress_out_temp_dist_conttext_separation(df)

            filtered_df = df[(df['is_feedbacked_state_to'] != 1) & (df['is_feedbacked_state_from'] != 1)]
            filtered_df = filtered_df[filtered_df['isMean'] == 0]
            filtered_df["context_separation"] = filtered_df["post_corr"] - filtered_df["pre_corr"]
            filtered_df = filtered_df.reset_index(drop=True)

            corr_var_names = ['corr','corr_temp_dist_regressed_out','corr_pre_sim_corrected']

            # Compute context-separation at the trial level
            filtered_df["context_separation_value"] = filtered_df["post_corr"] - filtered_df["pre_corr"]

            #Add a colu for abs. value difference bettwen the correlatied states 
            filtered_df["val_diff"] = (filtered_df["value_state_from"] - filtered_df["value_state_to"]).abs()

            # Identify "diff" vs "same" context in a clean way
            filtered_df["diff_context"] = (filtered_df["same_context"] == 0).astype(int)
            filtered_df["same_context"] = (filtered_df["same_context"] == 1).astype(int)
            filtered_df["imini_block"] = filtered_df["imini_block"].astype(int)


            # Avergae corr  variablesfor same and different context per roi and miniblock:     
            filtered_df_mean_per_miniblock = (
                filtered_df
                .groupby(["subject", "ROI", "runType", "imini_block", "same_context"])[corr_var_names]
                .mean()
                .reset_index())

            # Append to a NEW master dataframe
            try:
                coupdate_trials_df = pd.concat([coupdate_trials_df, filtered_df], ignore_index=True)
            except NameError:
                coupdate_trials_df = filtered_df.copy()


            for corr_var_name in corr_var_names:

                for ROI in filtered_df.ROI.unique():

                    corr_diff_context  = filtered_df_mean_per_miniblock[
                        (filtered_df_mean_per_miniblock['ROI']==ROI) &
                        (filtered_df_mean_per_miniblock['same_context']==0)
                    ][corr_var_name].mean()

                    corr_same_context = filtered_df_mean_per_miniblock[
                        (filtered_df_mean_per_miniblock['ROI']==ROI) &
                        (filtered_df_mean_per_miniblock['same_context']==1)
                    ][corr_var_name].mean()

                    pre_post_pair_temporal_dist_diff_context = filtered_df[
                        (filtered_df['ROI']==ROI) &
                        (filtered_df['same_context']==0)
                    ]['pre_post_pair_temporal_dist'].mean()

                    pre_post_pair_temporal_dist_same_context = filtered_df[
                        (filtered_df['ROI']==ROI) &
                        (filtered_df['same_context']==1)
                    ]['pre_post_pair_temporal_dist'].mean()

                    corr_diff_context_valdiff_100 = filtered_df[
                        (filtered_df['ROI']==ROI) &
                        (filtered_df['same_context']==0) &
                        (filtered_df['value_diff']==100)
                    ][corr_var_name].mean()

                    corr_same_context_valdiff_100 = filtered_df[
                        (filtered_df['ROI']==ROI) &
                        (filtered_df['same_context']==1) &
                        (filtered_df['value_diff']==100)
                    ][corr_var_name].mean()

                    corr_diff_context_valdiff_75 = filtered_df[
                        (filtered_df['ROI']==ROI) &
                        (filtered_df['same_context']==0) &
                        (filtered_df['value_diff']==75)
                    ][corr_var_name].mean()

                    corr_same_context_valdiff_75 = filtered_df[
                        (filtered_df['ROI']==ROI) &
                        (filtered_df['same_context']==1) &
                        (filtered_df['value_diff']==75)
                    ][corr_var_name].mean()

                    corr_diff_context_valdiff_50 = filtered_df[
                        (filtered_df['ROI']==ROI) &
                        (filtered_df['same_context']==0) &
                        (filtered_df['value_diff']==50)
                    ][corr_var_name].mean()

                    corr_same_context_valdiff_50 = filtered_df[
                        (filtered_df['ROI']==ROI) &
                        (filtered_df['same_context']==1) &
                        (filtered_df['value_diff']==50)
                    ][corr_var_name].mean()

                    corr_diff_context_valdiff_25 = filtered_df[
                        (filtered_df['ROI']==ROI) &
                        (filtered_df['same_context']==0) &
                        (filtered_df['value_diff']==25)
                    ][corr_var_name].mean()

                    corr_same_context_valdiff_25 = filtered_df[
                        (filtered_df['ROI']==ROI) &
                        (filtered_df['same_context']==1) &
                        (filtered_df['value_diff']==25)
                    ][corr_var_name].mean()

                    corr_diff_context_valdiff_0 = filtered_df[
                        (filtered_df['ROI']==ROI) &
                        (filtered_df['same_context']==0) &
                        (filtered_df['value_diff']==0)
                    ][corr_var_name].mean()

                    corr_same_context_valdiff_0 = filtered_df[
                        (filtered_df['ROI']==ROI) &
                        (filtered_df['same_context']==1) &
                        (filtered_df['value_diff']==0)
                    ][corr_var_name].mean()

                    corr_diff_context_valdiff_12 = filtered_df[
                        (filtered_df['ROI']==ROI) &
                        (filtered_df['same_context']==0) &
                        (filtered_df['value_diff']==12)
                    ][corr_var_name].mean()

                    corr_same_context_valdiff_12 = filtered_df[
                        (filtered_df['ROI']==ROI) &
                        (filtered_df['same_context']==1) &
                        (filtered_df['value_diff']==12)
                    ][corr_var_name].mean()

                    corr_diff_context_valdiff_62 = filtered_df[
                        (filtered_df['ROI']==ROI) &
                        (filtered_df['same_context']==0) &
                        (filtered_df['value_diff']==62)
                    ][corr_var_name].mean()

                    corr_same_context_valdiff_62 = filtered_df[
                        (filtered_df['ROI']==ROI) &
                        (filtered_df['same_context']==1) &
                        (filtered_df['value_diff']==62)
                    ][corr_var_name].mean()

                    corr_diff_context_valdiff_37 = filtered_df[
                        (filtered_df['ROI']==ROI) &
                        (filtered_df['same_context']==0) &
                        (filtered_df['value_diff']==37)
                    ][corr_var_name].mean()

                    corr_same_context_valdiff_37 = filtered_df[
                        (filtered_df['ROI']==ROI) &
                        (filtered_df['same_context']==1) &
                        (filtered_df['value_diff']==37)
                    ][corr_var_name].mean()

                    new_row = pd.DataFrame([[sub, ROI, corr_var_name,
                                             corr_diff_context,
                                             corr_same_context,
                                             corr_diff_context_valdiff_100,
                                             corr_diff_context_valdiff_75,
                                             corr_diff_context_valdiff_25,
                                             corr_diff_context_valdiff_50,
                                             corr_diff_context_valdiff_0,
                                             corr_diff_context_valdiff_12,
                                             corr_diff_context_valdiff_62,
                                             corr_diff_context_valdiff_37,
                                             corr_same_context_valdiff_100,
                                             corr_same_context_valdiff_75,
                                             corr_same_context_valdiff_25,
                                             corr_same_context_valdiff_50,
                                             corr_same_context_valdiff_0,
                                             corr_same_context_valdiff_12,
                                             corr_same_context_valdiff_62,
                                             corr_same_context_valdiff_37,
                                             pre_post_pair_temporal_dist_diff_context,
                                             pre_post_pair_temporal_dist_same_context]],
                                           columns=group_mean_df.columns)

                    new_row[group_mean_df.select_dtypes(include=['float']).columns] = (new_row[group_mean_df.select_dtypes(include=['float']).columns].astype(float))

                    group_mean_df = pd.concat([group_mean_df, new_row], ignore_index=True)

    # Select numeric columns only and ensure they stay numeric
    id_cols = ["subject", "ROI", "corr_var_name"]   # columns that should stay non-numeric
    num_cols = group_mean_df.columns.difference(id_cols)
    group_mean_df[num_cols] = group_mean_df[num_cols].apply(pd.to_numeric, errors='coerce')

    # Save final CSV with summary data (value per subjects per ROI)
    columns_to_save = ['subject', 'ROI', 'corr_var_name', 'corr_diff_context', 'corr_same_context']
    group_mean_df[columns_to_save].to_csv(base_dir + analysis + _zscore_suffix + '.csv', index=False, header=True, float_format='%.10f')

    # Save per miniblock CSV
    trial_output_path = base_dir + analysis + _zscore_suffix + "_miniblock_level_coupdate_values.csv"
    coupdate_trials_df.to_csv(trial_output_path, index=False)


