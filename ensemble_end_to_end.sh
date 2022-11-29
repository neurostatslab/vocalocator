#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --constraint=v100
#SBATCH -c 1
#SBATCH --gpus=1
#SBATCH --mem=64GB
#SBATCH --time=2:00:00
#SBATCH -o slurm_logs/end_to_end_%j.log
pwd; hostname; date;

PRETRAIN_DIR=$1
FINETUNE_DIR=$2
OUTPUT_BASE_DIR=$3
CONFIG=$4


if [ -z $PRETRAIN_DIR ]; then
    echo "Path to directory containing train/val/test datasets should be provided as the first positional argument"
    exit 1
fi

if [ -z $FINETUNE_DIR ]; then
    echo "Path to config JSON should be provided as second positional argument"
    exit 1
fi

if [ -z $OUTPUT_BASE_DIR ]; then
    echo "No output directory provided. Defaulting to /mnt/ceph/users/${USER}/end_to_end."
    OUTPUT_BASE_DIR=/mnt/ceph/users/${USER}/end_to_end
fi

# if the output base directory doesn't exist, create it
if [ ! -d $OUTPUT_BASE_DIR ]; then
    mkdir $OUTPUT_BASE_DIR
fi

# create a subdirectory for the current job id
OUTPUT_DIR=$OUTPUT_BASE_DIR/$SLURM_JOB_ID

if [ ! -d $OUTPUT_DIR ]; then
    mkdir $OUTPUT_DIR
fi

if [ -z $CONFIG ]; then
    echo "No configuration file provided. Currently no default available. Please provide one!"
    exit 1
fi

# 1. run end_to_end for each model
# 2. run buiold
