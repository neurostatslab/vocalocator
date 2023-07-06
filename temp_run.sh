#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH -c 6
#SBATCH --gpus=1
#SBATCH --mem=64GB
#SBATCH --time=4:00:00
#SBATCH -o slurm_logs/train_model_%j.log
pwd; hostname; date;

source ~/modules_h100.sh

DATA_DIR="/mnt/home/atanelus/ceph/merge_datasets/gpup/c3_pretrain_split/"
if [ ! -d /tmp/dataset ]; then
    cp -r $DATA_DIR /tmp/dataset
fi
CONFIG="/mnt/home/atanelus/transformer_config.json"

python -u -m gerbilizer.main --config $CONFIG \
    --data /tmp/dataset \
    --save_path ~/ceph/gerbilizer