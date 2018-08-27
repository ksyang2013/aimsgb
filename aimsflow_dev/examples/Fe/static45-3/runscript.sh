#!/bin/csh
#PBS -e err.$PBS_JOBID
#PBS -A kyang-group
#PBS -M Jianli@ucsd.edu
#PBS -o out.$PBS_JOBID
#PBS -N s_Fe45-3
#PBS -q condo
#PBS -m a
#PBS -l nodes=1:ppn=24:haswell
#PBS -l walltime=4:0:0
cd $PBS_O_WORKDIR
cp ../relax/CONTCAR POSCAR
qsub -W depend=afterany:$PBS_JOBID checkscript.sh
mpirun -v -machinefile $PBS_NODEFILE -np $PBS_NP  mpivasp54sLS > vasp.out.$PBS_JOBID
