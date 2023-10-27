#!/bin/bash

if [ "$1" == "same" ]; then
    gpu=$2
    echo "export CUDA_VISIBLE_DEVICES=${gpu}"
    export CUDA_VISIBLE_DEVICES=${gpu}
    # VIGOR same
    python -u myevaluate.py --iters_lev0 6 --CNN16  --flow --batch_size 32 --p_siamese --ori_noise 45 --restore_ckpt checkpoints/VIGOR/best_checkpoint_same.pth
elif [ "$1" == "cross" ]; then
    gpu=$2
    echo "export CUDA_VISIBLE_DEVICES=${gpu}"
    export CUDA_VISIBLE_DEVICES=${gpu}
    # VIGOR cross
    python -u myevaluate.py --iters_lev0 6 --CNN16 --flow --batch_size 32 --p_siamese --cross --ori_noise 45 --restore_ckpt checkpoints/VIGOR/best_checkpoint_cross.pth
else
    echo "Usage: $0 [same|cross] <GPU>"
    exit 1
fi
