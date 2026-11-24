#!/bin/bash

echo '#!/bin/bash'                    > job.slurm
echo "python /home/mpib/grossman/STATE_FORMATION/MRI_DATA/PILOT1/code/1_bids_conversion_state_formation/state_formation_bids_extrainfo.py"           >> job.slurm
sbatch job.slurm
rm -f job.slurm
