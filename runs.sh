#!/bin/bash

#SBATCH --ntasks 100
#SBATCH --output logfile.out
#SBATCH --qos normal
#SBATCH --time=02:30:00

module load intel/17.0.0
module load python/3.5.1
module load impi/2017.0.098


echo Beginning Job: 
mpiexec -n 100 python main.py 0


