#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH -c 2
#SBATCH --gpus=1
#SBATCH --mem=64GB
#SBATCH --mem=32768mb
#SBATCH --time=2:00:00
#SBATCH --array=1-20%5
#SBATCH -o slurm_logs/train_model_%a.log
pwd; hostname; date;

##################################################
# Usage:
# sbatch run_batch.sh ~/path/to/batch/configs ~/optional/path/to/output/directory
##################################################

# Expects the batch dir as first positional argument
BATCH_DIR=$1
# And the output dir as the second argument
OUTPUT_DIR=$2

if [ -z $BATCH_DIR ]; then
    echo "Path to directory containing config files should be provided as the first argument"
    exit 1
fi

if [-z $OUTPUT_DIR ]; then
    echo "No output directory provided. Defaulting to /mnt/ceph/users/${USER}/gerbilizer."
    OUTPUT_DIR=/mnt/ceph/users/${USER}/gerbilizer
fi

FMT_BATCH_IDX=$(python3 /mnt/home/atanelus/scripts/pad_integer.py 4 ${SLURM_ARRAY_TASK_ID})

# Note, config file should include path to data file under DATAFILE_PATH key
# Else this will crash
pipenv run python -u -m gerbilizer.main \
    --config_file $BATCH_DIR/batch_config_${FMT_BATCH_IDX}.json \
    --save_path $OUTPUT_DIR

date;
