#!/bin/bash/ -l
## This is to test a code for Demo at SC23
WORKDIR="/home/hossainm/CosmicTagger"
#cd ${WORKDIR}

#### For PyTorch runs ########
# module load conda/2023-10-04
###############################

Trial=2
#datatype=("float32" "mixed")
datatype=("float32")
frameworks=("torch")
#frameworks=("tensorflow")
#batchsizes=("1" "2" "4" "8")
batchsizes=("8")

echo ${frameworks[$1]}

lendatatype=${#datatype[@]}
echo "len-data-type = ${lendatatype}"

echo "${datatype[$1]}"

lenframeworks=${#frameworks[@]}
echo "len-frameworks = ${lenframeworks}"

iterations=100
rank=4

# MPI and OpenMP settings
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=${rank}

let NRANKS=${NNODES}*${NRANKS_PER_NODE}

export NVIDIA_TF32_OVERRIDE=0


#for ((i=1; i<=${lendatatype}; i++)); do
for i in ${!datatype[@]}; do
    for j in ${!batchsizes[@]}; do
        echo "i = ${i}"
        echo "datatype = ${datatype[$i]}"
        echo "batchsize = ${batchsizes[$j]}"
        LOCAL_BATCH_SIZE=${batchsizes[$j]}
        let GLOBAL_BATCH_SIZE=${LOCAL_BATCH_SIZE}*${NRANKS}
        echo "Global batch size: ${GLOBAL_BATCH_SIZE}"
        #TF_XLA_FLAGS=--tf_xla_auto_jit=2 
        mpiexec -n ${NRANKS} -ppn ${NRANKS_PER_NODE} --cpu-bind numa \
            python ${WORKDIR}/bin/exec.py --config-name a21 \
            run.id=run_plrs_SC23Demo_g${NRANKS}_t${datatype[$i]}_b${batchsizes[$j]}_${frameworks[$1]}_i${iterations}_T${Trial} \
            run.compute_mode=GPU \
            run.distributed=True \
            framework=${frameworks[$1]} \
            run.minibatch_size=${batchsizes[$j]} \
            run.precision=${datatype[$i]} \
            mode.optimizer.loss_balance_scheme=light \
            run.iterations=${iterations}
    done    
done
