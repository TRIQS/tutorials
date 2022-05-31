#!/bin/bash
#SBATCH --time=01:00:00    
#SBATCH --ntasks=8   
#SBATCH --mem-per-cpu=1024M
#SBATCH --output=lco.dmft.out   
#SBATCH --error=lco.dmft.err

module load StdEnv/2020 gcc/10.3.0 openmpi/4.1.1 triqs

mpirun solid_dmft
