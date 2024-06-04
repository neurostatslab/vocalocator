#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --constraint=v100
#SBATCH -c 1
#SBATCH --gpus=1
#SBATCH --mem=64GB
#SBATCH --time=4:00:00
#SBATCH -o slurm_logs/end_to_end_%j.log
pwd; hostname; date;

##################################################
# Usage:
# sbatch end_to_end.sh pretrain_data_dir finetune_data_dir [output_dir] [model_config] [finetune_changes]
# Example:
# sbatch end_to_end.sh \
#     /mnt/ceph/users/achoudhri/simulated_vox/small_room_correct_dims \
#     /mnt/ceph/users/achoudhri/simulated_vox/pup1_simulated \
#     /mnt/home/achoudhri/jsons/model_base_configs/best_performing_cov_mle.json
# data directories should contain the files train_set.h5, val_set.h5, and test_set.h5
# config.json should contain the CONFIG_NAME key
##################################################

# Arg 1: pretrain data dir
# Arg 2: finetune data dir
# Arg 3 (optional): output base dir (model output will be stored under a job_id subdirectory)
# Arg 4 (optional): path to config for model being trained
# Arg 5 (optional): path to json file storing desired changes to config for finetuning

PRETRAIN_DIR=$1
FINETUNE_DIR=$2
OUTPUT_BASE_DIR=$3
CONFIG=$4
FINETUNE_CHANGES=$5

if [ -z $PRETRAIN_DIR ]; then
    echo "Path to directory containing train/val/test datasets should be provided as the first positional argument"
    exit 1
fi

if [ -z $FINETUNE_DIR ]; then
    echo "Path to config JSON should be provided as second positional argument"
    exit 1
fi

if [ -z $OUTPUT_BASE_DIR ]; then
    echo "No output directory provided. Defaulting to /mnt/home/${USER}/ceph/end_to_end."
    OUTPUT_BASE_DIR=/mnt/home/${USER}/ceph/end_to_end
fi

# if the output base directory doesn't exist, create it
if [ ! -d $OUTPUT_BASE_DIR ]; then
    mkdir $OUTPUT_BASE_DIR
fi

# create a subdirectory for the current job id
OUTPUT_DIR=$OUTPUT_BASE_DIR/$SLURM_JOB_ID

if [ ! -d $OUTPUT_DIR ]; then
    mkdir $OUTPUT_DIR
fi

if [ -z $CONFIG ]; then
    echo "No config file provided. Defaulting to /mnt/home/achoudhri/jsons/best_performing_cov_mle.json"
    CONFIG=/mnt/home/achoudhri/jsons/best_performing_cov_mle.json
fi


echo "Pretraining model with config at path ${CONFIG} on simulated data in directory ${PRETRAIN_DIR}."

PRETRAIN_RESULT_DIR=$OUTPUT_DIR/pretrain
# pretrain the model
python -u -m vocalocator \
    --config $CONFIG \
    --data $PRETRAIN_DIR \
    --save-path $PRETRAIN_RESULT_DIR

# get the new config file with updated WEIGHTS_PATH
PRETRAINED_CONFIG=$PRETRAIN_RESULT_DIR/config.json

if [ ! -e $PRETRAINED_CONFIG ]; then
    echo "Model pretraining failed! Check logs at end_to_end_${SLURM_JOB_ID}.log"
    exit 1
fi

ASSESSMENT_DIR=$OUTPUT_DIR/assessment
PRETRAIN_ASSESS_FILE=$ASSESSMENT_DIR/pretrain.h5

echo "Assessing model performance on pretraining validation set, storing output to ${PRETRAIN_ASSESS_FILE}."
python -m vocalocator.assess \
    --config $PRETRAINED_CONFIG \
    --data $PRETRAIN_DIR/val_set.h5 \
    -o $PRETRAIN_ASSESS_FILE \
    --visualize

ZERO_SHOT_ASSESS_FILE=$ASSESSMENT_DIR/zero_shot.h5

echo "Assessing zero-shot model performance on finetuning validation set, storing output to ${ZERO_SHOT_ASSESS_FILE}."
python -m vocalocator.assess \
    --config $PRETRAINED_CONFIG \
    --data $FINETUNE_DIR/val_set.h5 \
    -o $ZERO_SHOT_ASSESS_FILE \
    --visualize

FINETUNE_RESULT_DIR=$OUTPUT_DIR/finetune

echo "Finetuning model on data in directory ${FINETUNE_DIR}."

# update config for finetuning if desired
CONFIG_FOR_FINETUNING=$PRETRAINED_CONFIG

if [ ! -z $FINETUNE_CHANGES ]; then
    CONFIG_FOR_FINETUNING=$PRETRAIN_RESULT_DIR/config_for_finetuning.json
    python -m vocalocator.update_json $PRETRAINED_CONFIG $FINETUNE_CHANGES $CONFIG_FOR_FINETUNING
fi

# finetune the model
python -u -m vocalocator \
    --config $CONFIG_FOR_FINETUNING \
    --data $FINETUNE_DIR \
    --save-path $FINETUNE_RESULT_DIR

# get the new config file with updated WEIGHTS_PATH
FINETUNED_CONFIG=$FINETUNE_RESULT_DIR/config.json

if [ ! -e $FINETUNED_CONFIG ]; then
    echo "Model finetuning failed! Check logs at end_to_end_${SLURM_JOB_ID}.log"
    exit 1
fi

FINAL_ASSESS_FILE=$ASSESSMENT_DIR/final.h5

echo "Assessing finetuned model performance on ground truth validation set, storing output to ${FINAL_ASSESS_FILE}."

python -m vocalocator.assess \
    --config $FINETUNED_CONFIG \
    --data $FINETUNE_DIR/val_set.h5 \
    -o $FINAL_ASSESS_FILE \
    --visualize
date;
