#!/bin/bash

# NOTE: this script was run from an interactive node

EXE=/eagle/projects/datascience/balin/Polaris/CFDML/data_driven_SGS_modeling/train/src/train_driver.py

# MPI and OpenMP settings
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=1
let NRANKS=${NNODES}*${NRANKS_PER_NODE}
echo 
echo Number of nodes: $NNODES
echo Number of ranks: $NRANKS
echo Number of ranks per node: $NRANKS_PER_NODE
echo

# Set up software deps and env:
module load conda/2022-09-08
conda activate base

# Set up additional env settings
#export NVIDIA_TF32_OVERRIDE=0

# Loop over precisions and batch sizes
for PRECISION in "fp64" "tf32"
#for PRECISION in "fp32"
do
   for BATCH in 512 2048 8192 32768
   do
      echo
      echo
      ARGS="data_path=synthetic logging=no \
         mini_batch=${BATCH} epochs=200 precision=${PRECISION} \
         model=sgs sgs.neurons=80 sgs.layers=5 \
         device=cuda distributed=ddp ppn=${NRANKS_PER_NODE}"
      echo Running with args $ARGS
      echo
      date
      mpiexec -n ${NRANKS} -ppn ${NRANKS_PER_NODE} --cpu-bind=numa \
            python $EXE $ARGS 2>&1 | tee sgs_${PRECISION}_${BATCH}.log
      date
   done
done

