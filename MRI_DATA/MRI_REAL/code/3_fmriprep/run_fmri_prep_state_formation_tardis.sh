#!/bin/bash
# ==============================================================================
# SCRIPT INFORMATION:
# ==============================================================================
# SCRIPT: RUN FMRIPREP ON THE MPIB CLUSTER (TARDIS)
# PROJECT: STATE FORMATION
# ==============================================================================
# DEFINE ALL PATHS:
# ==============================================================================
# define the path to the project folder:
PATH_PROJECT="$HOME/STATE_FORMATION"
# define the name of the current task:
TASK_NAME="fmriprep"
# path to the fmriprep singularity image:
PATH_CONTAINER="/mnt/beegfs/container/fmriprep/fmriprep-23.1.4.sif" #23.0.0
# path to the freesurfer license file on tardis:
PATH_FS_LICENSE="$HOME/.license"
# path to the data directory (in bids format):
PATH_BIDS="${PATH_PROJECT}/MRI_DATA/MRI_REAL/BIDs"
# path to the output directory:
PATH_OUT="${PATH_PROJECT}/MRI_DATA/MRI_REAL/derivatives"
# path to the freesurfer output directory:
PATH_FREESURFER="${PATH_OUT}/freesurfer"
# path to the working directory:
PATH_WORK="${PATH_PROJECT}/work"
# path to the log directory:
PATH_LOG="${PATH_PROJECT}/logs/fmriprep/"
# define the path for the templateflow cache
PATH_TEMPLATEFLOW="$HOME/.templateflow"
# define the path for the fmriprep cache:
PATH_CACHE_FMRIPREP="$HOME/.cache"
# ==============================================================================
# CREATE RELEVANT DIRECTORIES:
# ==============================================================================
mkdir -p ${PATH_OUT}
mkdir -p ${PATH_FREESURFER}
mkdir -p ${PATH_WORK}
mkdir -p ${PATH_LOG}
mkdir -p ${PATH_TEMPLATEFLOW}
mkdir -p ${PATH_CACHE_FMRIPREP}
# ==============================================================================
# DEFINE PARAMETERS:
# ==============================================================================
# maximum number of cpus per process:
N_CPUS=8
# maximum number of threads per process:
N_THREADS=8
# memory demand in *GB*
MEM_GB=40
# memory demand in *MB*
MEM_MB="$((${MEM_GB} * 1000))"
       
#participants=("501" "503" "505" "506" "507" "508" "509" "510" "511" "512" "513" "514" "515" 
#       "516" "517" "518" "519" "520" "521" "522" "523" "524" "525" "526" "527" "528" "529" "530"
#       "531" "532" "533" "534" "535" "536" "537" "538" "539" "540" "541" "542" "543" "544" "545" "546" "547" "548" "549"
#        "550" "551" "552" "553" "554" "555" "556" "557" "558" "559" "560")
        
participants=("561" "562" "563" "564")  

# ==============================================================================
# RUN FMRIPREP:
# ==============================================================================
# loop over all subjects:
for SUB in ${participants[*]}; do
	# get the subject number with zero padding:
	#SUB_PAD=$(printf "%02d\n" ${SUB})
	# skip fmriprep for input data that does not exist (yet):
	if [ ! -d "${PATH_BIDS}/sub-${SUB}" ]; then
		echo "BIDS input data for sub-${SUB} does not exist! Skipping ..."
		continue
	fi
	# create participant-specific working directory:
	PATH_WORK_SUB="${PATH_WORK}/sub-${SUB}"
	mkdir -p ${PATH_WORK_SUB}
	# create a new job file:
	echo "#!/bin/bash" > job
	# name of the job
	echo "#SBATCH --job-name fmriprep_fs_sub-${SUB}" >> job
	# add partition to job
	echo "#SBATCH --partition long" >> job
	# set the expected maximum running time for the job:
	echo "#SBATCH --time 40:00:00" >> job
	# determine how much RAM your operation needs:
	echo "#SBATCH --mem ${MEM_GB}GB" >> job
	# write log to log folder
	echo "#SBATCH --output ${PATH_LOG}/slurm_fmriprep_sub-${SUB}_%j.out" >> job
	# request multiple cpus
	echo "#SBATCH --cpus-per-task ${N_CPUS}" >> job
	# export template flow environment variable:
	echo "export APPTAINERENV_TEMPLATEFLOW_HOME=/templateflow" >> job
	# define the fmriprep command:
	echo "singularity run --cleanenv --contain \
	-B ${PATH_FS_LICENSE}:/.license:ro \
	-B ${PATH_BIDS}:/input:ro \
	-B ${PATH_OUT}:/output:rw \
	-B ${PATH_FREESURFER}:/output/freesurfer:rw \
	-B ${PATH_WORK_SUB}:/work:rw \
	-B ${PATH_TEMPLATEFLOW}:/templateflow:rw \
	${PATH_CONTAINER} \
	--fs-license-file /.license --notrack \
	/input/ /output/ participant --participant_label ${SUB} -w /work/ \
	--mem_mb ${MEM_MB} --nthreads ${N_CPUS} --omp-nthreads ${N_THREADS} \
	--write-graph --stop-on-first-crash \
	--output-spaces T1w MNI152NLin6Asym fsnative fsaverage\
	--notrack --verbose " >> job
	# submit job to cluster queue and remove it to avoid confusion:
	sbatch job
	rm -f job
done
