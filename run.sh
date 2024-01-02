#!/bin/bash
#SBATCH --job-name=EAGLE_offset         # Job name
#SBATCH --mail-type=END,FAIL            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=alexgarcia@ufl.edu  # Where to send mail	
#SBATCH --ntasks=1                      # Run on a single CPU
#SBATCH --mem=32gb                      # Job memory request
#SBATCH --time=10:00:00                 # Time limit hrs:min:sec
#SBATCH --output=EAGLE_%j.log           # Standard output and error log
pwd; hostname; date

module purge
module load conda
conda activate myenv

python read_eagle.py

date
