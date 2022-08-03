#!/bin/bash

JOB_ID=$(
    sbatch --parsable run_eval.sh /mnt/ceph/users/atanelus/iteration/small_room_4/test_set.h5 \
    /mnt/home/achoudhri/gerbilizer/trained_models/alex/00206/config.json \
    ~/ceph/calibration/alex_00206
    )

logfile="~/gerbilizer/slurm_logs/eval_model_${JOB_ID}.log"
time_waited=0
while [[ ! -e $logfile ]]
do
sleep 5
time_waited=$(($time_waited + 5))
if [[ $time_waited > 1000 ]]; then
    echo "To see output:"
    echo "less +F ${logfile}"
    exit
fi
done
tail -F $logfile

