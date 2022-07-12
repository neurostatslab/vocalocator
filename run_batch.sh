#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --constraint=v100
#SBATCH -c 1
#SBATCH --gpus=1
#SBATCH --mem=32768mb
#SBATCH --time=1:30:00
#SBATCH --array=1-500%4
#SBATCH -o slurm_logs/train_model_%a.log
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
# DATA_DIR=$1
BATCH_DIR=$1

# if [ -z $DATA_DIR ]; then
#     echo "Path to train/val/test datasets should be provided as the first positional argument"
#     exit 1
# fi

if [ -z $BATCH_DIR ]; then
    echo "Path to directory containing config files should be provided as the second positional argument"
    exit 1
fi

#if [ -z $JOB_ID ]; then
#    echo "Job id should be provided as second positional argument"
#    exit 1
#fi

FMT_BATCH_IDX=$(python3 /mnt/home/atanelus/scripts/pad_integer.py 4 ${SLURM_ARRAY_TASK_ID})

# Note, config file should include path to data file under DATAFILE_PATH key
# Else this will crash
pipenv run python training/train.py \
    --config_file $BATCH_DIR/batch_config_${FMT_BATCH_IDX}.json

date;