#!/bin/bash -l
## This script runs FFN batch job with max_steps=5,000
## fp32
## Running with max_steps=6000 for Intel compatiable parameters, num_inter=1
#PBS -l select=1:ngpus=1:system=polaris
#PBS -l filesystems=home:grand
#PBS -l place=scatter
#PBS -l walltime=00:30:00
#PBS -q debug
#PBS -A datascience
#PBS -k doe
#PBS -o /home/hossainm/flood-filling-networks/ffn_stdout
#PBS -e /home/hossainm/flood-filling-networks/ffn_stderr
#PBS -j oe
#PBS -N FFN-FP32

BATCH_SIZE=16
MAX_STEPS=200
PRECISION="fp32"
BIND="numa"

TRIAL=30

module use /soft/modulefiles
module load conda/2022-09-08
conda activate 
source /home/hossainm/envs/ffn_tf_213_3/bin/activate
module load cmake
module load cudatoolkit-standalone/11.8.0

export CRAY_ACCEL_TARGET=nvidia80
export MPICH_GPU_SUPPORT_ENABLED=1
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:/home/hossainm/envs/ffn_tf_213_3/lib/python3.8/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

export NNODES=1
export NUM_RANKS_PER_NODE=1
export NUM_CORES_PER_RANK=1
export NUM_RANKS=$(( NNODES * NUM_RANKS_PER_NODE ))
export NUM_INTER=$NUM_CORES_PER_RANK 
export NUM_INTRA=1

WORKDIR="/home/hossainm/flood-filling-networks/ffn"
SCRIPTDIR="/home/hossainm/flood-filling-networks/ffn/train_scripts"
OUTFILEDIR="/home/hossainm/flood-filling-networks/ffn/train_scripts/logfiles/intel/tensorflow-213"

DIR_TAG="test_${PRECISION}_b${BATCH_SIZE}_mst${MAX_STEPS}_N${NNODES}_R${NUM_RANKS_PER_NODE}_${BIND}bind_cores_per_rank${NUM_CORES_PER_RANK}_T${TRIAL}_intel-2022-09-08-TF213"


DATA_DIR="/home/hossainm/flood-filling-networks/ffn/FFN_DATA/validation_sample"
TRAIN_DIR="/home/hossainm/flood-filling-networks/ffn/FFN_DATA/training/ffn_unified/intel/${DIR_TAG}"

echo ${TRAIN_DIR}

export NVIDIA_TF32_OVERRIDE=0
#export BIND_TO="verbose,list:0,1,2,3,4,5,6,7:8,9,10,11,12,13,14,15:16,17,18,19,20,21,22,23:24,25,26,27,28,29,30,31"
export BIND_TO="numa"

mpiexec -n ${NUM_RANKS} -ppn ${NUM_RANKS_PER_NODE} --envall --cpu-bind ${BIND_TO} \
./set_affinity_gpu_polaris.sh \
python /home/hossainm/flood-filling-networks/ffn/horovod_train2.py \
    --train_coords ${DATA_DIR}/tf_record_file \
    --data_volumes valdation1:${DATA_DIR}/grayscale_maps.h5:raw \
    --label_volumes valdation1:${DATA_DIR}/groundtruth.h5:stack \
    --model_name convstack_3d.ConvStack3DFFNModel \
    --model_args_file ${SCRIPTDIR}/params \
    --image_mean 128 --image_stddev 33 \
    --learning_rate 1.6e-6 \
    --num_intra_threads=${NUM_INTRA} \
    --num_inter_threads=${NUM_INTER} \
    --optimizer adam \
    --max_steps ${MAX_STEPS} \
    --batch_size ${BATCH_SIZE} \
    --summary_rate_secs 1020 \
    --train_dir ${TRAIN_DIR} \
    --horovod \
    2>&1 | tee ${OUTFILEDIR}/${DIR_TAG}.txt

# /home/hossainm/flood-filling-networks/ffn/train_scripts
