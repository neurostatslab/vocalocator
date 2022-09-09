#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --constraint=v100
#SBATCH -c 1
#SBATCH --gpus=1
#SBATCH --mem=32768mb
#SBATCH --time=2:00:00
#SBATCH --array=1-20%5
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

# Expects the batch dir as first positional argument
BATCH_DIR=$1
# And the output dir as the second argument
OUTPUT_DIR=$2

if [ -z $BATCH_DIR ]; then
    echo "Path to directory containing config files should be provided as the second positional argument"
    exit 1
fi

FMT_BATCH_IDX=$(python3 /mnt/home/atanelus/scripts/pad_integer.py 4 ${SLURM_ARRAY_TASK_ID})

# Note, config file should include path to data file under DATAFILE_PATH key
# Else this will crash
python training/train.py \
    --config_file $BATCH_DIR/batch_config_${FMT_BATCH_IDX}.json \
    --save_path $OUTPUT_DIR

date;
