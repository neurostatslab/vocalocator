CNN_CONFIG=/mnt/home/achoudhri/ceph/models/00266/config.json
TRANSFORMER_CONFIG=/mnt/home/achoudhri/ceph/models/00365/config.json

CNN_JOB=$(sbatch --parsable tune_std_single.sh $CNN_CONFIG)
TRANSFORMER_JOB=$(sbatch --parsable tune_std_single.sh $TRANSFORMER_CONFIG)

echo "CNN JOB:"
echo "less +F slurm_logs/tune_model_${CNN_JOB}.log"
echo "TRANSFORMER JOB:"
echo "less +F slurm_logs/tune_model_${TRANSFORMER_JOB}.log"
