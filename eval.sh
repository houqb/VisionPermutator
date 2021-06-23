if [ ! $1 ];
then
    DATA_DIR=/path/to/imagenet/val
else
    DATA_DIR="$1"
fi
if [ ! $2 ];
then
    MODEL_DIR=/path/to/checkpoint
else
    MODEL_DIR="$2"
fi
python3 validate.py $DATA_DIR  --model vip_s7 --checkpoint $MODEL_DIR/vip_s7.pth --no-test-pool --amp --img-size 224 -b 64

