#!/bin/csh
#PBS -e err.check.$PBS_JOBID
#PBS -A kyang-group
#PBS -M Jianli@ucsd.edu
#PBS -o out.check.$PBS_JOBID
#PBS -N check_Fe45
#PBS -q condo
#PBS -m a
#PBS -l nodes=1:ppn=24:haswell
#PBS -l walltime=0:2:0
cd $PBS_O_WORKDIR
aimsflow vasp -cj s
