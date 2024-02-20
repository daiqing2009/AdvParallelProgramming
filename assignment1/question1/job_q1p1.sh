#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH --account=mcs
time mpirun -np 40 q1p1.o -n 205000000
