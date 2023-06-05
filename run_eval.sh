#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH -c 2
#SBATCH --gpus=1
#SBATCH --mem=64GB
#SBATCH --time=0:15:00
#SBATCH -o slurm_logs/eval_model_%j.log
pwd; hostname; date;

##################################################
# Usage:
# sbatch run_gpu.sh ~/ceph/path_to_data ~/path/to/config.json [~/path/to/outdir]
# path_to_data/ should be a directory containing a file named 'test_set.h5'
# config.json should contain the CONFIG_NAME key
# ~/path/to/outdir is an optional path to a directory where the outputs should be stored.
##################################################

DATA_FILE=$1
CONFIG=$2
OUT_PATH=$3

if [ -z $DATA_FILE ]; then
    echo "Path to dataset should be provided as the first positional argument"
    exit 1
fi

if [ -z $CONFIG ]; then
    echo "Config name or path to config JSON should be provided as second positional argument"
    exit 1
fi

if [ -z $OUT_PATH ]; then
    pipenv run python -u -m gerbilizer.main \
        --eval \
        --data $DATA_FILE \
        --config $CONFIG
else
    pipenv run python -u -m gerbilizer.main \
        --eval \
        --data $DATA_FILE \
        --config $CONFIG \
        --output_path $OUT_PATH
fi

date;
