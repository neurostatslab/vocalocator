CNN_CONFIG=/mnt/home/achoudhri/ceph/models/00266/config.json
TRANSFORMER_CONFIG=/mnt/home/achoudhri/ceph/models/00365/config.json

bash run_calib_single.sh $CNN_CONFIG
bash run_calib_single.sh $TRANSFORMER_CONFIG

echo "CNN JOB:"
echo "less +F ~/logs/convnet/eval_calib.log"
echo "TRANSFORMER JOB:"
echo "less +F ~/logs/transformer/eval_calib.log"
