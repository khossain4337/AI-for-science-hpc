#!/bin/bash
SCRIPTDIR="/home/hossainm/pointnet-atlas/atlas-pointnet/run_scripts"
WORKDIR="/home/hossainm/pointnet-atlas/atlas-pointnet"
#source set_affinity_gpu_polaris.sh
BATCHSIZE=1
TYPEDIR="float32_runs"
RESULTDIR="test_${TYPEDIR}_b${BATCHSIZE}_e4_tfile10k_N1_G4-no-aux"
TRIAL=1
LOGFILE="LOG_${RESULTDIR}_trial_${TRIAL}.txt"
OUTFILE="OUT_${RESULTDIR}_trial_${TRIAL}.txt"

NDEPTH=8

mpiexec -n 8 -npernode 8  --bind-to numa \
python3 ${WORKDIR}/run_pointnet.py -c ${WORKDIR}/configs/atlas-jlse.json \
    --device "cuda" --logdir ${SCRIPTDIR}/logdir/$(date "+%Y-%m-%d")/${RESULTDIR}/TRIAL_${TRIAL} \
    --logfile ${SCRIPTDIR}/logfiles/${TYPEDIR}/${LOGFILE} --batch ${BATCHSIZE} \
    --horovod \
    2>&1 | tee ${SCRIPTDIR}/logfiles/${TYPEDIR}/${OUTFILE}
