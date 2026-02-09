#!/bin/bash
#
# Baked training launcher for GeneT5 fine-tuning
#
# Usage:
#   Single node:  bin/train.sh
#   Master:       bin/train.sh --nnodes 2 --node-rank 0 --master 192.168.100.10
#   Worker:       bin/train.sh --nnodes 2 --node-rank 1 --master 192.168.100.10

set -e

TRAIN_DATA="../baked/w5k_c4.5k/training.packed"
VAL_DATA="../baked/w5k_c4.5k/validation.packed"
OUTPUT_DIR="../model/exp_$(date +%Y%m%d)_train"
MODEL_PATH="../model/base"

bin/distributed.sh \
    "$TRAIN_DATA" "$VAL_DATA" "$OUTPUT_DIR" "$MODEL_PATH" \
    --epochs 4 \
    --lr 1e-4 \
    --token_budget 36400 \
    --max_batch_size 8 \
    --grad_accum 64 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1.0 \
    --label_smoothing 0.1 \
    --early_stopping 2 \
    --save_steps 500 \
    --empty_cache_steps 100 \
    --memwatch \
    "$@"
