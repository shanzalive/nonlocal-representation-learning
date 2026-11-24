#!/bin/bash

# ==============================================================================
# SCRIPT INFORMATION:
# ==============================================================================
# SCRIPT: RUN c_coupdate_analysis.py FOR EVERY PARTICIPANT ON THE MPIB CLUSTER (TARDIS)
# Participants: the full + partial learners of summary_files_behaviour/learners_dict.json (N = 51).
# Runs the main pipeline, or the robustness variants of Fig. S3 if doMultiverse = True in
# c_coupdate_analysis.py.
# ==============================================================================
# DEFINE ALL PATHS:
# ==============================================================================
PATH_LOG="/home/mpib/grossman/STATE_FORMATION/MRI_DATA/MRI_REAL/code/logs/6_multivariate_analyses/"$(date '+%Y%m%d_%H%M%S')
PATH_SCRIPT="/home/mpib/grossman/STATE_FORMATION/MRI_DATA/MRI_REAL/code/6_multivariate_analyses/"
mkdir -p ${PATH_LOG}
# ==============================================================================
# DEFINE PARAMETERS:
# ==============================================================================
# maximum number of cpus per process:
N_CPUS=1
# maximum number of threads per process:
N_THREADS=8
# memory demand in *GB*
MEM_GB=20
# memory demand in *MB*
MEM_MB="$((${MEM_GB} * 1000))"
# ==============================================================================
# RUN PYTHON FITTING SCRIPT:
# ==============================================================================
BASE="/home/mpib/grossman/STATE_FORMATION/"
LEARNERS_JSON="${BASE}summary_files_behaviour/learners_dict.json"
mapfile -t participants < <(python3 -c "
import json
with open('${LEARNERS_JSON}') as f:
    d = json.load(f)
for s in sorted(d['full_and_partial']):
    print(s.replace('-', ''))
")

# Output folders that the next run writes to (must match c_coupdate_analysis.py):
# the main pipeline if doMultiverse = False, the 5 robustness variants of Fig. S3 if True.
if grep -q "^doMultiverse = True" ${PATH_SCRIPT}c_coupdate_analysis.py; then
  OUTPUT_DIRS=(
    "ROI_timeseries_LSS_5mm_smoothing_dropout_30_compcorr"
    "ROI_timeseries_LSS_5mm_smoothing_dropout_50_physio_reg_out_updated"
    "ROI_timeseries_LSS_5mm_smoothing_dropout_80_physio_reg_out_updated"
    "ROI_timeseries_LSS_5mm_smoothing_physio_reg_out_updated"
    "ROI_timeseries_LSS_3mm_smoothing_dropout_30_physio_reg_out_updated"
  )
else
  OUTPUT_DIRS=("ROI_timeseries_LSS_5mm_smoothing_dropout_30_physio_reg_out_updated")
fi

# Delete the participant csvs of a previous run in these folders only, so that no
# participant from an older run is left over
for UNIVERSE in "${OUTPUT_DIRS[@]}"; do
  COUPDATE_DIR="${BASE}MRI_DATA/MRI_REAL/${UNIVERSE}/coupdate_LSS/"
  echo "Cleaning up previous per-subject CSVs in ${COUPDATE_DIR}..."
  rm -f ${COUPDATE_DIR}sub*_coupdate_LSS*.csv
  echo "Deleted, $(ls ${COUPDATE_DIR}sub*_coupdate_LSS*.csv 2>/dev/null | wc -l) files remaining (should be 0)."
done

# loop over all fsfs:
  cd ${PATH_SCRIPT}
  #for fsf in *visloc_v2.fsf
  for subj in ${participants[*]};
  do
  	# create a new job file:
	echo "#!/bin/bash" > job
	# name of the job
	echo "#SBATCH --job-name coupdate_${subj}" >> job
	# add partition to job
	echo "#SBATCH --partition short" >> job #long
	# set the expected maximum running time for the job:
	echo "#SBATCH --time 4:00:00" >> job
	# determine how much RAM your operation needs:
	echo "#SBATCH --mem ${MEM_GB}GB" >> job
	# email notification on abort/end, use 'n' for no notification:
	echo "#SBATCH --mail-type NONE" >> job
	# write log to log folder
	echo "#SBATCH --output ${PATH_LOG}/%j.out" >> job
	# request multiple cpus
	echo "#SBATCH --cpus-per-task ${N_CPUS}" >> job
   	# submit the feat command with the current design file
	echo "python3 c_coupdate_analysis.py ${subj}"  >> job
	# submit job to cluster queue and remove it to avoid confusion:
	sbatch job
	rm -f job
 done

