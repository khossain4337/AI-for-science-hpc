#!/bin/bash

EXE=/gila/cfdml_aesp_CNDA/CFDML/data_driven_SGS_modeling/train/src/train_driver.py

# MPI and OpenMP settings
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=4
CCS=4
let NRANKS=${NNODES}*${NRANKS_PER_NODE}
echo 
echo Number of nodes: $NNODES
echo Number of ranks: $NRANKS
echo Number of ranks per node: $NRANKS_PER_NODE
echo Number of CCS per tile: $CCS
echo

# Set up software deps and env:
module load frameworks/2023.05.15.001

# Set up additional env settings
export ZE_AFFINITY_MASK=0.0
if [ $CCS -eq 1 ]
then
    export ZEX_NUMBER_OF_CCS=0:1
elif [ $CCS -eq 2 ]
then
    export ZEX_NUMBER_OF_CCS=0:2
elif [ $CCS -eq 4 ]
then
    export ZEX_NUMBER_OF_CCS=0:4
fi
#export ONEDNN_DEFAULT_FPMATH_MODE=TF32 
BIND_LIST="0-1,104-105:2-3,106-107:4-5,108-109:6-7,110-111"

# Loop over precisions and batch sizes
for PRECISION in "fp32"
do
   for BATCH in 8192
   do
      echo
      echo
      ARGS="data_path=synthetic logging=debug \
            mini_batch=${BATCH} epochs=200 precision=${PRECISION} \
            model=sgs sgs.neurons=80 sgs.layers=5 \
            device=xpu distributed=ddp ppn=${NRANKS_PER_NODE} ppd=${CCS}"
      echo Running with args $ARGS
      echo
      date
      mpiexec --pmi=pmix -n ${NRANKS} --ppn ${NRANKS_PER_NODE} --cpu-bind=verbose,list:${BIND_LIST} \
            python $EXE $ARGS 
      date
   done
done


