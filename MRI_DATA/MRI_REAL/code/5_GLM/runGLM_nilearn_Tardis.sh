#!/bin/bash

# ==============================================================================
# SCRIPT INFORMATION:
# ==============================================================================
# SCRIPT: RUN NLOPT OPTIMIZER THE FIT RNN TO PPT RESPONSES ON THE MPIB CLUSTER (TARDIS)
# ==============================================================================
# DEFINE ALL PATHS:
# ==============================================================================
PATH_LOG="/home/mpib/grossman/STATE_FORMATION/MRI_DATA/MRI_REAL/code/logs/5_GLM/"$(date '+%Y%m%d_%H%M%S')
PATH_SCRIPT="/home/mpib/grossman/STATE_FORMATION/MRI_DATA/MRI_REAL/code/5_GLM/"
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
participants=("sub501" "sub503" "sub505" "sub506" "sub507" "sub508" 
  "sub509" "sub510" "sub511" "sub512" "sub513" "sub514" "sub515" "sub516" 
  "sub517" "sub518" "sub519" "sub520" "sub521" "sub522" "sub523" "sub524" 
  "sub525" "sub526" "sub527" "sub528" "sub529" "sub530" "sub531" "sub532" 
  "sub533" "sub534" "sub535" "sub536" "sub537" "sub538" "sub539" "sub540" 
  "sub541" "sub542" "sub543" "sub544" "sub545" "sub546" "sub547" "sub548" 
  "sub549" "sub550" "sub551" "sub552" "sub553" "sub554" "sub555" "sub556" 
  "sub557" "sub558" "sub559" "sub560" "sub561" "sub562" "sub563" "sub564")


# loop over all fsfs:
  cd ${PATH_SCRIPT}
  #for fsf in *visloc_v2.fsf
  for subj in ${participants[*]};
  do
  	# create a new job file:
	echo "#!/bin/bash" > job
	# name of the job
	echo "#SBATCH --job-name GLM_${subj}" >> job
	# add partition to job
	echo "#SBATCH --partition long" >> job #long
	# set the expected maximum running time for the job:
	echo "#SBATCH --time 24:00:00" >> job
	# determine how much RAM your operation needs:
	echo "#SBATCH --mem ${MEM_GB}GB" >> job
	# email notification on abort/end, use 'n' for no notification:
	echo "#SBATCH --mail-type NONE" >> job
	# write log to log folder
	echo "#SBATCH --output ${PATH_LOG}/%j.out" >> job
	# request multiple cpus
	echo "#SBATCH --cpus-per-task ${N_CPUS}" >> job
   	# submit the feat command with the current design file
	echo "python3 runGLM_nilearn.py ${subj}"  >> job
	# submit job to cluster queue and remove it to avoid confusion:
	sbatch job
	rm -f job
 done
