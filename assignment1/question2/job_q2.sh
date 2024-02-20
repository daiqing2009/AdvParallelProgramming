#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH --account=mcs
mpirun -np 32 q2.o

