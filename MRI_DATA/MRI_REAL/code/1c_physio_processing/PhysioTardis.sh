#!/bin/bash
sbatch -c 2 --mem 2gb --wrap '. /etc/profile ; module load matlab/R2021b ; matlab -r "Physio_updated() ; exit"'
