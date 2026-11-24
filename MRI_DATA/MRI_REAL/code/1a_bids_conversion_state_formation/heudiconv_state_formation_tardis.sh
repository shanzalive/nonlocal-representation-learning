#!/bin/bash
HEURISTIC_PATH="$HOME/STATE_FORMATION/MRI_DATA/MRI_REAL/code/1_bids_conversion_state_formation/heudiconv_state_formation_nononorm_nogre.py" # path to your heuristic 
LOGS_PATH="$HOME/STATE_FORMATION/logs/heudiconv/"	# Path where error and out logs of cluster jobs are saved


#sublist=("501" "503")
#sublist=("501" "503" "505" "506" "507" "508" "509" "510" "511" "512" "513" "514" "515" 
#       "516" "517" "518" "519" "520" "521" "522" "523" "524" "525" "526" "527" "528" "529" "530"
#       "531" "532" "533" "534" "535" "536" "537" "538" "539" "540" "541" "542" "543" "544" "545" "546" "547" "548" "549"
#        "550" "551" "552" "553" "554" "555" "556" "557" "558" "559" "560")
        
sublist=("561" "562" "563" "564")


for sub in ${sublist[@]}; do # loop over all studies
    if [ "$sub" == "507" ]; then
        HEURISTIC_PATH="$HOME/STATE_FORMATION/MRI_DATA/MRI_REAL/code/1a_bids_conversion_state_formation/heudiconv_state_formation_nononorm_nogre_sub507.py" # 
    #elif [ "$sub" == "525" ]; then
    #    HEURISTIC_PATH="$HOME/STATE_FORMATION/MRI_DATA/MRI_REAL/code/1_bids_conversion_state_formation/heudiconv_state_formation_nononorm_nogre_sub525.py" # 
    #elif [ "$sub" == "526" ]; then
    #    HEURISTIC_PATH="$HOME/STATE_FORMATION/MRI_DATA/MRI_REAL/code/1_bids_conversion_state_formation/heudiconv_state_formation_nononorm_nogre_sub526.py" # 
    elif [ "$sub" == "535" ]; then
        HEURISTIC_PATH="$HOME/STATE_FORMATION/MRI_DATA/MRI_REAL/code/1a_bids_conversion_state_formation/heudiconv_state_formation_nononorm_nogre_sub535.py" # 
    elif [ "$sub" == "539" ]; then
        HEURISTIC_PATH="$HOME/STATE_FORMATION/MRI_DATA/MRI_REAL/code/1a_bids_conversion_state_formation/heudiconv_state_formation_nononorm_nogre_sub539.py" # 
    elif [ "$sub" == "542" ]; then
        HEURISTIC_PATH="$HOME/STATE_FORMATION/MRI_DATA/MRI_REAL/code/1a_bids_conversion_state_formation/heudiconv_state_formation_nononorm_nogre_sub542.py" # 
    elif [ "$sub" == "544" ]; then
        HEURISTIC_PATH="$HOME/STATE_FORMATION/MRI_DATA/MRI_REAL/code/1a_bids_conversion_state_formation/heudiconv_state_formation_nononorm_nogre_sub544.py" # 
    elif [ "$sub" == "553" ]; then
        HEURISTIC_PATH="$HOME/STATE_FORMATION/MRI_DATA/MRI_REAL/code/1a_bids_conversion_state_formation/heudiconv_state_formation_nononorm_nogre_sub553.py" # 
    elif [ "$sub" == "554" ]; then
        HEURISTIC_PATH="$HOME/STATE_FORMATION/MRI_DATA/MRI_REAL/code/1a_bids_conversion_state_formation/heudiconv_state_formation_nononorm_nogre_sub554.py" # 
    else
        HEURISTIC_PATH="$HOME/STATE_FORMATION/MRI_DATA/MRI_REAL/code/1a_bids_conversion_state_formation/heudiconv_state_formation_nononorm_nogre.py"
    fi
    
    if [ "$sub" == "534" ]; then
        sessions=("1" "2" "3" "302")
    else
        sessions=("1" "2" "3")
    fi
        
	for ses in ${sessions[@]}; do
		INPUT_PATH=$HOME/STATE_FORMATION/MRI_DATA/MRI_REAL/DICOMs/SUB${sub}/STATEFORMATION_${sub}_0${ses}_${sub}/*/*/*IMA 
		OUTPUT_PATH=$HOME/STATE_FORMATION/MRI_DATA/MRI_REAL/BIDs/
		echo $sub
		echo $ses
		echo $HEURISTIC_PATH
		echo '#!/bin/bash'                             > job.slurm
		echo "#SBATCH --job-name bidsconve-$sub$ses"  >> job.slurm
		echo "#SBATCH --partition quick"               >>   job.slurm
		echo "#SBATCH --time 2:0:0"                   >> job.slurm
		echo "#SBATCH --mem 16GB"                      >> job.slurm
		echo "#SBATCH --cpus-per-task 1 "              >> job.slurm
		echo "#SBATCH --output ${LOGS_PATH}\logbids_${sub}ses${ses}.out"   >> job.slurm
		echo "#SBATCH --mail-type NONE"                >> job.slurm
		echo "heudiconv --files ${INPUT_PATH} -s ${sub} -ss ${ses} -f ${HEURISTIC_PATH} -c dcm2niix -o ${OUTPUT_PATH} -b --overwrite"  >> job.slurm # run the job
		sbatch job.slurm
		rm -f job.slurm
	done

done


