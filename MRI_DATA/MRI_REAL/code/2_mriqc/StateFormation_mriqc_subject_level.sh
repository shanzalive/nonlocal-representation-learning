#!/bin/bash
# ==============================================================================
# SCRIPT INFORMATION:
# ==============================================================================
# SCRIPT: CREATE PARTICIPANT-LEVEL MRIQC REPORTS FOR A BIDS-STRUCTURED DATASET
# PROJECT: STATE_FORMATION
# WRITTEN BY LENNART WITTKUHN, ADAPTED BY SAM HALL-MCMASTER 2023, ADAPTED BY GROSSMAN 2023
# CONTACT: GROSSMAN AT MPIB HYPHEN BERLIN DOT MPG DOT DE
# MAX PLANCK RESEARCH GROUP NEUROCODE
# MAX PLANCK INSTITUTE FOR HUMAN DEVELOPMENT (MPIB)
# MAX PLANCK UCL CENTRE FOR COMPUTATIONAL PSYCHIATRY AND AGEING RESEARCH
# LENTZEALLEE 94, 14195 BERLIN, GERMANY
# ACKNOWLEDGEMENT: THANKS TO ALEXANDER SKOWRON AND NIR MONETA @ MPIB FOR HELP
# ==============================================================================
# DEFINE ALL PATHS:
# ==============================================================================
# path to the base directory:
PATH_HOME="${HOME}"
# define the name of the current task:
TASK_NAME="mriqc"
# define the path to the project folder:
PATH_PROJECT="${PATH_HOME}/STATE_FORMATION/MRI_DATA/MRI_REAL"
# define the path to the script main directory:
PATH_CODE="${PATH_PROJECT}/code/${TASK_NAME}"
# define the path to the singularity container:
PATH_CONTAINER="/data/container/mriqc/mriqc-23.0.1.sif" #mriqc-0.16.1.sif"
# define the path for the templateflow cache
PATH_TEMPLATEFLOW="${PATH_HOME}/.cache/templateflow"
# path to the data directory (in bids format):
PATH_INPUT="${PATH_HOME}/STATE_FORMATION/MRI_DATA/MRI_REAL/BIDs"
# update the input directory:
#datalad update  -d ${PATH_INPUT} --merge
#datalad get ${PATH_INPUT}
# path to the output directory:
PATH_OUTPUT="${PATH_PROJECT}/mriqc"
# path to the working directory:
PATH_WORK="${PATH_PROJECT}/mriqc/work"
# path to the log directory:
PATH_LOG="$HOME/STATE_FORMATION/logs/mriqc/"	 
# ==============================================================================
# CREATE RELEVANT DIRECTORIES:
# ==============================================================================
# create output directory:
if [ ! -d ${PATH_OUTPUT} ]
then
	mkdir -p ${PATH_OUTPUT}
fi
# create working directory:
if [ ! -d ${PATH_WORK} ]
then
	mkdir -p ${PATH_WORK}
fi
# create directory for log files:
if [ ! -d ${PATH_LOG} ]
then
	mkdir -p ${PATH_LOG}
fi
# create template flow directory:
#if [ ! -d ${PATH_TEMPLATEFLOW} ]
#then
#	mkdir -p ${PATH_TEMPLATEFLOW}
#fi

# ==============================================================================
# DEFINE PARAMETERS:
# ==============================================================================
# maximum number of cpus per process:
N_CPUS=8
# memory demand in *GB*
MEM_GB=40
# maximum number of threads per process:
N_THREADS=8
# declare an array with sessions you want to run:
declare -a SESSIONS=("1" "2" "3") #("02") #("01" "02")

# ==============================================================================
# RUN MRIQC:
# ==============================================================================
# loop across all subjects
SUB_LIST=("501" "503" "553") 
#SUB_LIST=("505" "506" "507" "508" "509" "510" "511" "512" "513" "514" "515" 
#       "516" "517" "518" "519" "520" "521" "522" "523" "524" "525" "526" "527" "528" "529" "530"
#       "531" "532" "533" "534" "535" "536" "537" "538" "539" "540" "541" "542" "543" "544" "545" "546" "547" "548" "549"
#        "550" "551" "552" "553" "554" "555" "556" "557" "558" "559" "560")
for SUB in ${SUB_LIST[*]}; do
	# create the subject number with zero-padding:
	SUB_PAD=$(printf "%02d\n" ${SUB})
	# loop over all sessions:
	for SES in ${SESSIONS[@]}; do
		# check the existence of the input files and continue if data is missing:
		if [ ! -d ${PATH_INPUT}/sub-${SUB_PAD}/ses-${SES} ]; then
			echo "No data input available for sub-${SUB_PAD} ses-${SES}!"
			continue
		fi
		
                # define a subject-specific work path, to allow parallel running
		PATH_WORK_SUB="${PATH_WORK}/sub-${SUB_PAD}/ses-${SES}"
		# create a subject-specific work path
		if [ ! -d ${PATH_WORK_SUB} ]
		then
			mkdir -p ${PATH_WORK_SUB}
		fi

		# create a new job file:
		echo "#!/bin/bash" > job
		# name of the job
		echo "#SBATCH --job-name STATE_FORMATION-mriqc_sub-${SUB_PAD}_ses-${SES}" >> job
		# add partition to job
		echo "#SBATCH --partition long" >> job
		# set the expected maximum running time for the job:
		echo "#SBATCH --time 40:00:00" >> job
		# determine how much RAM your operation needs:
		echo "#SBATCH --mem ${MEM_GB}GB" >> job
		# email notification on abort/end, use 'n' for no notification:
		echo "#SBATCH --mail-type NONE" >> job
		# write log to log folder:
		echo "#SBATCH --output \
		${PATH_LOG}/slurm_STATE_FORMATION-mriqc_sub-${SUB_PAD}_ses-${SES}%j.out" >> job
		# request multiple cpus:
		echo "#SBATCH --cpus-per-task ${N_CPUS}" >> job
		# export template flow environment variable:
		#echo "export SINGULARITYENV_TEMPLATEFLOW_HOME=/templateflow" >> job
		echo "export APPTAINERENV_TEMPLATEFLOW_HOME=/templateflow" >> job
		# define the main command:
		echo "singularity run --home $HOME --cleanenv --contain \
		-B ${PATH_INPUT}:/input:ro \
		-B ${PATH_OUTPUT}:/output:rw \
		-B ${PATH_WORK_SUB}:/work:rw \
		-B ${PATH_TEMPLATEFLOW}:/templateflow:rw \
		${PATH_CONTAINER} /input/ /output/ participant \
		--participant-label ${SUB_PAD} \
		--session-id ${SES_PAD} -w /work/ --verbose-reports --write-graph \
		--n_cpus ${N_CPUS} --mem_gb ${MEM_GB} --no-sub" >> job
		# submit job to cluster queue and remove it to avoid confusion:
		sbatch job
		rm -f job
	done
done