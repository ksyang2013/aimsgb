#!/bin/csh
#PBS -e err.$PBS_JOBID
#PBS -A kyang-group
#PBS -M Jianli@ucsd.edu
#PBS -o out.$PBS_JOBID
#PBS -N r_Fe
#PBS -q condo
#PBS -m a
#PBS -l nodes=1:ppn=24:haswell
#PBS -l walltime=08:00:00
cd $PBS_O_WORKDIR

qsub -W depend=afterany:$PBS_JOBID checkscript.sh
mpirun -v -machinefile $PBS_NODEFILE -np $PBS_NP  mpivasp54s > vasp.out.$PBS_JOBID
