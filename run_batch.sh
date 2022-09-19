#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH -c 2
#SBATCH --gpus=1
#SBATCH --mem=64GB
#SBATCH --time=1:30:00
#SBATCH --array=1-500%4
#SBATCH -o slurm_logs/train_model_%a.log
pwd; hostname; date;

BATCH_DIR=$1

if [ -z $BATCH_DIR ]; then
    echo "Path to directory containing config files should be provided as the second positional argument"
    exit 1
fi

FMT_BATCH_IDX=$(python3 /mnt/home/atanelus/scripts/pad_integer.py 4 ${SLURM_ARRAY_TASK_ID})

# Note, config file should include path to data file under DATAFILE_PATH key
# Else this will crash
pipenv run python -u -m gerbilizer.main \
    --config $BATCH_DIR/batch_config_${FMT_BATCH_IDX}.json \
    --save_path /mnt/ceph/users/${USER}/gerbilizer

date;