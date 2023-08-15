#!/bin/bash
##################################
## This is a sample job script for polaris GPU runs
# Here we are trying to loop over instances of tensorflow with 
# mixed and float32 precision + XLA.
# 1 Node in polaris
#PBS -l select=1:ngpus=4:system=polaris
#PBS -l filesystems=home
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -q debug
#PBS -A datascience
#PBS -k doe
#PBS -o /home/hossainm/CosmicTagger/test_stdout
#PBS -e /home/hossainm/CosmicTagger/test_stderr
#PBS -j oe
#PBS -N xla-full-node

#CosmicTagger working directory
WORK_DIR=/home/hossainm/CosmicTagger
cd ${WORK_DIR}

# MPI and OpenMP settings
NNODES=`wc -l < $PBS_NODEFILE` #Should be 1, detemined by the select (?)
NRANKS_PER_NODE=$(nvidia-smi -L | wc -l) #should be 4

let NRANKS=${NNODES}*${NRANKS_PER_NODE}

LOCAL_BATCH_SIZE=2 #minibatch_size (?)
let GLOBAL_BATCH_SIZE=${LOCAL_BATCH_SIZE}*${NRANKS}

echo $GLOBAL_BATCH_SIZE

echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NRANKS}" 
echo "RANKS_PER_NODE= ${NRANKS_PER_NODE}"

# Set up computing environment:
module load conda/2022-09-08
conda activate
module load cray-hdf5/1.12.1.3

datatype=( "float32" "mixed" )
frameworks=("tensorflow")
batchsizes=("1" "2" "4" "8")

echo ${frameworks[1]}

lendatatype=${#datatype[@]}
echo "len-data-type = ${lendatatype}"

echo "${datatype[$1]}"

lenframeworks=${#frameworks[@]}
echo "len-frameworks = ${lenframeworks}"

#for ((i=1; i<=${lendatatype}; i++)); do
for i in ${!datatype[@]}; do
    for j in ${!batchsizes[@]}; do
        echo "i = ${i}"
        echo "datatype = ${datatype[$i]}"
        echo "batchsize = ${batchsizes[$j]}"
        TF_XLA_FLAGS=--tf_xla_auto_jit=2 mpiexec -n 4 -ppn 4 --cpu-bind=numa \
	    python bin/exec.py --config-name a21 \
	    run.id=parsl_${datatype[$i]}_${batchsizes[$j]}_${frameworks[$1]}_i500-XLA-fullnode-Rep4 \
	    run.compute_mode=GPU \
	    run.distributed=True \
	    framework=${frameworks[$1]} \
	    run.minibatch_size=${batchsizes[$j]} \
	    run.precision=${datatype[$i]} \
	    mode.optimizer.loss_balance_scheme=light \
	    run.iterations=500
    done    
done

