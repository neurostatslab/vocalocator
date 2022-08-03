#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --constraint=v100
#SBATCH -c 1
#SBATCH --gpus=1
#SBATCH --mem=32768mb
#SBATCH --time=2:00:00
#SBATCH -o slurm_logs/tune_model_%j.log
pwd; hostname; date;

##################################################
# Usage:
# sbatch run_gpu.sh ~/ceph/path_to_data ~/path/to/config.json [~/path/to/outdir]
# sbatch run_gpu.sh ~/ceph/path_to_data default 4 [~/path/to/outdir]
# path_to_data/ should be an hdf5 file containing the dataset "vocalizations"
# config.json should contain the CONFIG_NAME key
# ~/path/to/outdir is an optional path to a directory where the outputs should be stored.
# Alternatively, a string containing one of the configs listed in configs.py may be provided
# If a config name is provided in place of a file, a job ID must also be provided
##################################################

# Expects the config path/name as second argument
CONFIG=$1

pipenv run python training/best_std_val.py $CONFIG

echo "To see logs, run the following command:"
echo "less +F slurm_logs/tune_model_"

date;
