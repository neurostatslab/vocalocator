#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH -c 1
#SBATCH --gpus=1
#SBATCH --mem=32768mb
#SBATCH --time=1:00:00
#SBATCH -o slurm_logs/train_model_%j.log
pwd; hostname; date;

##################################################
# Usage:
# sbatch run_gpu.sh ~/ceph/path_to_data ~/path/to/config.json
# sbatch run_gpu.sh ~/ceph/path_to_data default
# path_to_data/ should contain the files train_set.h5, val_set.h5, and test_set.h5
# config.json should contain the CONFIG_NAME key
# Alternatively, a string containing one of the configs listed in configs.py may be provided
##################################################

# Expects the data dir as first positional argument
# config path/name as second argument
DATA_DIR=$1
CONFIG=$2

if [ -z $DATA_DIR ]; then
    echo "Path to train/val/test datasets should be provided as the first positional argument"
    exit 1
fi

if [ -z $CONFIG ]; then
    echo "Config name or path to config JSON should be provided as second positional argument"
    exit 1
fi


if [ -f $CONFIG ]; then
    pipenv run python training/train.py \
        --config_file $CONFIG \
        --datafile $DATA_DIR \
        --save_path /mnt/ceph/users/${USER}/gerbilizer
else
    pipenv run python training/train.py \
        --config $CONFIG \
        --datafile $DATA_DIR \
        --save_path /mnt/ceph/users/${USER}/gerbilizer
fi

date;