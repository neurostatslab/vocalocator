#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --constraint=v100
#SBATCH -c 1
#SBATCH --gpus=1
#SBATCH --mem=32768mb
#SBATCH --time=0:15:00
#SBATCH -o slurm_logs/eval_model_%j.log
pwd; hostname; date;

##################################################
# Usage:
# sbatch run_gpu.sh ~/ceph/path_to_data ~/path/to/config.json
# sbatch run_gpu.sh ~/ceph/path_to_data default 4
# path_to_data/ should be an hdf5 file containing the dataset "vocalizations"
# config.json should contain the CONFIG_NAME key
# Alternatively, a string containing one of the configs listed in configs.py may be provided
# If a config name is provided in place of a file, a job ID must also be provided
##################################################

# Expects the data dir as first positional argument
# config path/name as second argument
DATA_FILE=$1
CONFIG=$2
JOB_ID=$3

if [ -z $DATA_FILE ]; then
    echo "Path to dataset should be provided as the first positional argument"
    exit 1
fi

if [ -z $CONFIG ]; then
    echo "Config name or path to config JSON should be provided as second positional argument"
    exit 1
fi

if [ -z $JOB_ID ]; then
    pipenv run python training/eval.py \
        $DATA_FILE \
        --config $CONFIG
else
    pipenv run python training/eval.py \
        $DATA_FILE \
        --config $CONFIG \
        --job_id $JOB_ID
fi


date;