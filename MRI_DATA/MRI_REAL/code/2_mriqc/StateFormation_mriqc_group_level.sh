#!/bin/bash
# ==============================================================================
# SCRIPT INFORMATION:
# ==============================================================================
# SCRIPT: CREATE GROUP-LEVEL MRIQC REPORTS FOR A BIDS-STRUCTURED DATASET
# ONLY RUN AFTER ALL PARTICIPANT-LEVEL REPORTS ARE FINISHED!
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
PATH_CONTAINER="/data/container/mriqc/mriqc-23.0.1.sif"
# define the path for the templateflow cache
PATH_TEMPLATEFLOW="${PATH_BASE}/.cache/templateflow"
# path to the data directory (in bids format):
PATH_INPUT="${PATH_HOME}/STATE_FORMATION/MRI_DATA/MRI_REAL/BIDs"
# path to the output directory:
PATH_OUTPUT="${PATH_PROJECT}/${TASK_NAME}"
#datalad unlock ${PATH_OUTPUT}/sub-*/ses-*/*/*.json
#datalad unlock ${PATH_OUTPUT}/group*.tsv
# path to the working directory:
PATH_WORK=${PATH_PROJECT}/work
# path to the log directory:
PATH_LOG=${PATH_PROJECT}/logs
# ==============================================================================
# CREATE RELEVANT DIRECTORIES:
# ==============================================================================
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
# ==============================================================================
# RUN MRIQC TO CREATE THE GROUP REPORTS:
# ==============================================================================
#-B ${PATH_TEMPLATEFLOW}:/templateflow:rw \
# create group reports for the functional data:
singularity run --home $HOME --cleanenv --contain -B ${PATH_INPUT}:/input:ro \
-B ${PATH_OUTPUT}:/output:rw -B ${PATH_WORK}:/work:rw \
${PATH_CONTAINER} /input/ /output/ group --no-sub