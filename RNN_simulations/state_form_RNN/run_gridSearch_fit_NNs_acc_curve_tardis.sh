#!/bin/bash

# ==============================================================================
# Fit RNN to participants using accuracy-curve MSE objective
# Calls fit_RNN_to_participants.py for each subject.
# ==============================================================================
PATH_LOG="/home/mpib/grossman/STATE_FORMATION/RNN_simulations/state_form_RNN_outputs/NN_activations/gradientDescentPerfromanceMatched/grid_search_fits/logs/"$(date '+%Y%m%d_%H%M%S')
PATH_SCRIPT="/home/mpib/grossman/STATE_FORMATION/RNN_simulations/state_form_RNN/"
mkdir -p ${PATH_LOG}

N_CPUS=1
N_THREADS=8
MEM_GB=20
MEM_MB="$((${MEM_GB} * 1000))"


participants=("sub503" "sub505" "sub506" "sub507" "sub508" "sub509" "sub510" "sub512"
  "sub513" "sub514" "sub516" "sub517" "sub518" "sub519" "sub520" "sub521" "sub522"
  "sub523" "sub524" "sub526" "sub527" "sub528" "sub529" "sub530" "sub531" "sub532"
  "sub533" "sub534" "sub536" "sub537" "sub538" "sub540" "sub541" "sub542" "sub544"
  "sub545" "sub546" "sub547" "sub548" "sub549" "sub552" "sub553" "sub554" "sub555"
  "sub557" "sub558" "sub559" "sub560" "sub561" "sub563" "sub564")


  cd ${PATH_SCRIPT}
  for subj in ${participants[*]};
  do
	echo "#!/bin/bash" > job
	echo "#SBATCH --job-name fit_acc_${subj}" >> job
	echo "#SBATCH --partition long" >> job
	echo "#SBATCH --time 24:00:00" >> job
	echo "#SBATCH --mem ${MEM_GB}GB" >> job
	echo "#SBATCH --mail-type NONE" >> job
	echo "#SBATCH --output ${PATH_LOG}/%j.out" >> job
	echo "#SBATCH --cpus-per-task ${N_CPUS}" >> job
	echo "python3 fit_RNN_to_participants.py ${subj}" >> job
	sbatch job
	rm -f job
 done
