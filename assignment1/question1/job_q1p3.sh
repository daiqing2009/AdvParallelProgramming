#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH --account=mcs
time mpirun -np 40 q1p3.o -s 5 -d 0.1
