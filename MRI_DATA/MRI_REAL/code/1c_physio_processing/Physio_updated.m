%% Compute physiological noise regressors (RETROICOR) for all participants and sessions
%  Reads the cardiac and respiratory recordings (PHYSIOs/) and the scan timing from the
%  DICOMs, and creates the physiological noise regressors that are added to the GLMs
%  (5_GLM/runGLM_nilearn.py). Run on the cluster with PhysioTardis.sh (MATLAB R2021b).
%
%  REQUIRED TOOLBOXES (not included in this repository) - download both and add them
%  to the MATLAB path (here done with addpath below; change these paths to your own):
%   - TAPAS PhysIO toolbox: TAPAS release 6.0.1 (PhysIO R2022a, v8.1.0)
%     https://github.com/translationalneuromodeling/tapas
%   - SPM12 (version 7771): https://www.fil.ion.ucl.ac.uk/spm/software/spm12/
%  tapas_physio_new is a function of the PhysIO toolbox; it creates the default
%  settings that are filled in below (for documentation of the parameters: edit tapas_physio_new).
%
%  Author: Shany Grossman
%
%  CHANGES vs Physio.m:
%  1. DICOM deduplication fix: later sessions stored raw+NORM reconstructions
%     in the same series directory (2x DICOMs per volume). The old code used
%     assert(all(dcm_suff(1)<dcm_suff(2:end))) which crashed on duplicate suffixes.
%     New code deduplicates before asserting.
%  2. Session exclusion list updated: sessions previously excluded only because
%     of the DICOM duplication issue are now re-enabled.
%  3. Subjects still excluded (non-DICOM issues, investigate separately):
%     - SUB502 ses2 (signal problem), SUB520 ses2 (empty processed folder),
%       SUB541 ses1+2 (telnet/signal), SUB545/546/550 all ses (TAPAS internal
%       assertion - likely signal quality), SUB534 ses3 (glob ambiguity with
%       ses302 folder - handled via ses302 exclusion logic below).

function Physio_updated()

tardis = true; % if false - work locally on server data
if tardis
    base_dir = '/home/mpib/grossman/STATE_FORMATION/MRI_DATA/MRI_REAL';
    addpath(genpath('/home/mpib/grossman/matlab_toolboxes/tapas-master'));
    addpath(genpath('/home/mpib/grossman/matlab_toolboxes/spm12'));
else
    base_dir  = '/Volumes/MPRG-Neurocode/Data/immobilienagentur_2022_shany/MRI_DATA/MRI_REAL';
end

dcm_dir = fullfile(base_dir,'DICOMs');
phys_dir = fullfile(base_dir,'PHYSIOs');

save_dir = fullfile(base_dir,'PHYSIO_processed');
if ~exist(save_dir, 'dir')
    mkdir(save_dir)
end

% Only the subjects/sessions newly enabled by the DICOM deduplication fix.
% Previously-working sessions are NOT re-run here.
%todo = { ...
%    '506', {'3'};        % ses3 was skipped; ses1+2 already done
%    '507', {'3'};        % ses3 was skipped; ses1+2 already done
%    '518', {'1'};        % ses1 was skipped; ses2+3 already done
%    '526', {'3'};        % ses3 was skipped; ses1+2 already done
%    '530', {'1'};        % ses1 was skipped; ses2+3 already done
%    '534', {'302'};      % ses302 was never run; ses1+2 already done
%    '535', {'1'};        % ses1 was skipped; ses2+3 already done
%    '542', {'3'};        % ses3 was skipped; ses1+2 already done
%    '543', {'1','2','3'};% all sessions were skipped
%    '552', {'1','2','3'};% all sessions were skipped
%    '554', {'3'};        % ses3 was skipped; ses1+2 already done
%    '559', {'1'};        % ses1 was skipped; ses2+3 already done
%    '560', {'2'};        % ses2 was skipped; ses1+3 already done
%   '561', {'2','3'};    % ses2+3 were skipped; ses1 already done
%    '563', {'3'};        % ses3 was skipped; ses1+2 already done
%};

% 2 additional participants left out form previous round: 
todo = { ...
    '502', {'1', '2'};        % ses3 was cancelled 
    '504', {'1', '2'};        % ses3 was cancelled

};

for isub = 1:size(todo, 1)
    sub      = todo{isub, 1};
    sessions = todo{isub, 2};


    for ises=1:numel(sessions)
        ses = sessions{ises};
        %check if there are physio files for this session
        if exist(fullfile(phys_dir,['SUB' sub],['StateFormation_' sub '_0' ses(1) '.puls']), 'file')
            disp(['sub-' sub ' ses' ses])
        else
            disp(['sub-' sub ' ses' ses ' no physio files'])
            continue
        end

        %get paths to all functional runs in the current session
        % For SUB534 ses302: use exact folder suffix '0302' to avoid matching
        % the plain ses3 folder (STATEFORMATION_534_03_534 also matches '*_03*').
        if strcmpi(sub,'534') && strcmpi(ses,'302')
            ses_glob_pattern = [sub '_0302'];
        else
            ses_glob_pattern = [sub '_0' ses(1)];
        end
        func_dirs = {dir([dcm_dir '/SUB' sub '/*' ses_glob_pattern '*/*/*RUN*']).name};
        a = unique({dir([dcm_dir '/SUB' sub '/*' ses_glob_pattern '*/*/*RUN*']).folder});
        if isempty(a)
            disp(['  No DICOM run dirs found for SUB' sub ' ses' ses ' - skipping'])
            continue
        end
        ses_dir = a{1};

        for irun=1:numel(func_dirs)
            % Filter for NORM directories only (the second copy, which ironically
            % has 'NORM' in image type).
            dcm_list = dir([ses_dir, '/', func_dirs{irun},'/*.IMA']);
            if isempty(dcm_list)
                disp(['  No IMA files in ' func_dirs{irun} ' - skipping'])
                continue
            end

            % --- DICOM deduplication fix ---
            % Some sessions store both raw and NORM reconstructions in the same
            % series directory, resulting in 2 files per volume with identical
            % SOP-UID suffixes but different timestamps. Lexicographic sort puts
            % the scan-time file (earlier timestamp) before the recon-time copy,
            % so unique(...,'stable') retains the right one.
            dcm_names_all = sort({dcm_list.name});
            dcm_suff_cell = cellfun(@(x) str2num(x(end-11:end-4)), dcm_names_all, 'UniformOutput', false);
            valid_mask = ~cellfun(@isempty, dcm_suff_cell);
            dcm_names_all = dcm_names_all(valid_mask);
            dcm_suff_all  = [dcm_suff_cell{valid_mask}];
            [~, unique_idx] = unique(dcm_suff_all, 'stable');
            dcm_names = dcm_names_all(unique_idx);
            dcm_suff  = dcm_suff_all(unique_idx); %#ok<NASGU>
            % Sort by numeric suffix for consistent ordering
            [~, sort_idx] = sort(dcm_suff_all(unique_idx));
            dcm_names = dcm_names(sort_idx);
            % --- end deduplication fix ---

            info = dicominfo(fullfile(dcm_list(1).folder, dcm_names{1}));
            if  contains(info.ImageType , 'NORM')

                %% Create default parameter structure with all fields
                physio = tapas_physio_new();

                %% Individual Parameter settings. Modify to your need and remove default settings
                physio.save_dir = {fullfile(save_dir, ['SUB' sub], ['SUB' sub '_0' ses])};
                physio.log_files.vendor = 'Siemens';
                physio.log_files.cardiac = {fullfile(phys_dir,['SUB' sub],['StateFormation_' sub '_0' ses(1) '.puls'])};
                disp(physio.log_files.cardiac)
                physio.log_files.respiration = {fullfile(phys_dir,['SUB' sub],['StateFormation_' sub '_0' ses(1) '.resp'])};
                physio.log_files.scan_timing = {fullfile(dcm_list(1).folder, dcm_names{1})}; % first DICOM of session (scan-time copy after dedup)
                % The time stamp in the DICOM header is on the same time
                % axis as the time stamp in the physiological log file
                physio.log_files.sampling_interval = 0.02;           % 50Hz for both puls and resp (400 Hz for ecg, 200Hz ext)
                physio.log_files.align_scan = 'first';               % start of logfile will be aligned to last scan volume
                physio.scan_timing.sqpar.Nslices = 80/4;               % number of slices per volume in fMRI scan
                physio.scan_timing.sqpar.TR = 1.75;
                physio.scan_timing.sqpar.Ndummies = 0;               % number of dummy volumes - CHECK!!!
                physio.scan_timing.sqpar.Nscans = length(dcm_names); % number of full volumes saved (correct after dedup)
                physio.scan_timing.sqpar.time_slice_to_slice = 1.75/80/4; % CHECK IF THIS IS CORRECT GIVEN MULTIBAND;
                physio.scan_timing.sqpar.onset_slice = 10; %the slice for hich the physio signal will be sample
                physio.scan_timing.sync.method = 'scan_timing_log';  % uses individual scan timing logfile with time stamps
                %                                                                  specified in log_files.scan_timing
                physio.preproc.cardiac.modality = 'PPU';             % 'PPU' (for pulse oximetry)
                physio.preproc.cardiac.filter.include = false;
                physio.preproc.cardiac.filter.type = 'butter';
                physio.preproc.cardiac.filter.passband = [0.3 9];
                physio.preproc.cardiac.initial_cpulse_select.method = 'auto_matched';
                physio.preproc.cardiac.initial_cpulse_select.max_heart_rate_bpm = 120;
                physio.preproc.cardiac.initial_cpulse_select.file = 'initial_cpulse_kRpeakfile.mat';
                physio.preproc.cardiac.initial_cpulse_select.min = 0.4;
                physio.preproc.cardiac.posthoc_cpulse_select.method = 'off';
                physio.preproc.cardiac.posthoc_cpulse_select.percentile = 80;
                physio.preproc.cardiac.posthoc_cpulse_select.upper_thresh = 60;
                physio.preproc.cardiac.posthoc_cpulse_select.lower_thresh = 60;
                physio.preproc.respiratory.filter.passband = [0.01 2];
                physio.preproc.respiratory.despike = false;
                physio.model.orthogonalise = 'none';
                physio.model.censor_unreliable_recording_intervals = false;
                physio.model.output_multiple_regressors = [func_dirs{irun}(end-10:end) '_multiple_regressors.txt'];
                physio.model.output_physio = [func_dirs{irun}(end-10:end) '_physio.mat'];
                physio.model.retroicor.include = true;
                physio.model.retroicor.order.c = 3;
                physio.model.retroicor.order.r = 4;
                physio.model.retroicor.order.cr = 1;
                physio.model.rvt.include = false;
                physio.model.rvt.method = 'hilbert';
                physio.model.rvt.delays = 0;
                physio.model.hrv.include = false;
                physio.model.hrv.delays = 0;
                physio.model.noise_rois.include = false;
                physio.model.noise_rois.thresholds = 0.9;
                physio.model.noise_rois.n_voxel_crop = 0;
                physio.model.noise_rois.n_components = 1;
                physio.model.noise_rois.force_coregister = 1;
                physio.model.movement.include = false;
                physio.model.movement.order = 6;
                physio.model.movement.censoring_threshold = 0.5;
                physio.model.movement.censoring_method = 'FD';
                physio.model.other.include = false;
                physio.verbose.level = 0; %no graphic output
                physio.verbose.process_log = cell(0, 1);
                physio.verbose.fig_handles = zeros(1, 0);
                physio.verbose.use_tabs = false;
                physio.verbose.show_figs = false;
                physio.verbose.save_figs = false;
                physio.verbose.close_figs = false;
                physio.ons_secs.c_scaling = 1;
                physio.ons_secs.r_scaling = 1;
                physio.version = 'R2022a-v8.1.0';

                %% Run physiological recording preprocessing and noise modeling
                try
                    physio = tapas_physio_main_create_regressors(physio);
                catch ME
                    warning('SUB%s ses%s run %s FAILED: %s', sub, ses, func_dirs{irun}(end-10:end), ME.message);
                end
            end
        end
    end
end
end
