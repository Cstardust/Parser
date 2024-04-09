set -ex

export PYTHONPATH=$PYTHONPATH:$(pwd)
python train.py \
    --plm ./ckpt/chinese-electra-180g-base-discriminator \
    --train_file ./data/train.conll \
    --batch_size 32 \
    --plm_lr 2e-5 \
    --head_lr 1e-4 \
    --scheduler linear \
    --num_epochs 15 \
    --random_seed 123