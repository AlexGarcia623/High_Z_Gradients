#!/bin/bash
#SBATCH --job-name=EAGLE_nsf            # Job name
#SBATCH --mail-type=END,FAIL            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=alexgarcia@ufl.edu  # Where to send mail	
#SBATCH --ntasks=1                      # Run on a single CPU
#SBATCH --mem=32gb                      # Job memory request
#SBATCH --time=36:00:00                 # Time limit hrs:min:sec
#SBATCH --output=sSFMS_nsf_%j.log       # Standard output and error log
pwd; hostname; date

module purge
module load conda

conda activate py3

python sSFMS_test.py
# python read_eagle.py
# python read_illustris.py

# conda activate py3 
# python read_fire.py # May want to consider requesting 64GB when doing `core` sample

date
