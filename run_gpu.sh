#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH -c 6
#SBATCH --gpus=1
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH -o slurm_logs/train_model_%j.log
pwd; hostname; date;

##################################################
# Usage:
# sbatch run_gpu.sh ~/ceph/path_to_data ~/path/to/config.json
# path_to_data/ should contain the files train_set.h5, val_set.h5, and test_set.h5
# config.json should contain the CONFIG_NAME key
##################################################

DATA_DIR=$1
CONFIG=$2
OUTPUT_DIR=$3

if [ -z $DATA_DIR ]; then
    echo "Path to directory containing train/val/test datasets should be provided as the first positional argument"
    exit 1
fi

if [ -z $CONFIG ]; then
    echo "Path to config JSON should be provided as second positional argument"
    exit 1
fi

if [ -z $OUTPUT_DIR ]; then
    echo "No output directory provided. Defaulting to /mnt/ceph/users/${USER}/gerbilizer."
    OUTPUT_DIR=/mnt/home/${USER}/ceph/gerbilizer
fi

echo "Copying dataset to local machine"
date;
cp -r $DATA_DIR /tmp/dataset
echo "Done copying dataset"
date;

source /mnt/home/${USER}/.bashrc
source /mnt/home/atanelus/venvs/general/bin/activate

# Working directory is expected to be the root of the gerbilizer repo
python -u -m gerbilizer.main \
    --config $CONFIG \
    --data /tmp/dataset \
    --save_path $OUTPUT_DIR

date;
