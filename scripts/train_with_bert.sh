set -ex

export PYTHONPATH=$PYTHONPATH:$(pwd)
python train.py \
    --plm ./plm/bert-base-chinese \
    --train_file ./data/train.conll \
    --batch_size 16 \
    --plm_lr 2e-5 \
    --head_lr 1e-4 \
    --scheduler linear \
    --num_epochs 15 \
    --random_seed 123 \
    --res_dir results_v2