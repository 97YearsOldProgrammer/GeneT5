#!/bin/bash
#
# Launch distributed fine-tuning across two DGX Spark systems
# Uses PyTorch distributed with NCCL backend over ConnectX-7 200Gb/s network
#
# Usage:
#   Single node:  ./launch_distributed.sh --single
#   Multi-node:   ./launch_distributed.sh --master <master_ip>
#

set -e

# Default configuration
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

# Training defaults
TRAIN_DATA="${TRAIN_DATA:-data/gene_prediction.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/genet5_finetuned}"
MODEL_PATH="${MODEL_PATH:-checkpoints/genet5_init}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-1e-4}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --single)
            NNODES=1
            NODE_RANK=0
            shift
            ;;
        --master)
            MASTER_ADDR="$2"
            shift 2
            ;;
        --port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --node-rank)
            NODE_RANK="$2"
            shift 2
            ;;
        --nnodes)
            NNODES="$2"
            shift 2
            ;;
        --nproc)
            NPROC_PER_NODE="$2"
            shift 2
            ;;
        --train-data)
            TRAIN_DATA="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Print configuration
echo "========================================"
echo "DGX Spark Distributed Training Launch"
echo "========================================"
echo "Master:          $MASTER_ADDR:$MASTER_PORT"
echo "Nodes:           $NNODES (this is node $NODE_RANK)"
echo "Procs per node:  $NPROC_PER_NODE"
echo "Train data:      $TRAIN_DATA"
echo "Output dir:      $OUTPUT_DIR"
echo "Model path:      $MODEL_PATH"
echo "Batch size:      $BATCH_SIZE"
echo "Grad accum:      $GRAD_ACCUM"
echo "Epochs:          $EPOCHS"
echo "Learning rate:   $LR"
echo "========================================"

# Set NCCL environment variables for DGX Spark ConnectX-7 network
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_SOCKET_IFNAME=eth0

# Enable TCP for initial rendezvous
export GLOO_SOCKET_IFNAME=eth0

# Launch with torchrun
torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    bin/finet.py \
    "$TRAIN_DATA" \
    "$OUTPUT_DIR" \
    "$MODEL_PATH" \
    --distributed \
    --backend nccl \
    --batch_size $BATCH_SIZE \
    --grad_accum $GRAD_ACCUM \
    --epochs $EPOCHS \
    --lr $LR \
    --find_unused_params \
    "$@"