#!/bin/bash
#
# RunPod H100 training launcher
#
# Usage:
#   bash train/run_h100.sh <data_dir> <output_dir> <model_path> [flags...]
#
# Examples:
#   bash train/run_h100.sh /workspace/data /workspace/model/mar13_50k_t1 /workspace/model/init_dense_24L
#   bash train/run_h100.sh /workspace/data /workspace/model/mar13_50k_t1 /workspace/model/init_dense_24L --epochs 2 --lr 0.03

set -e

if [ $# -lt 3 ]; then
    echo "Usage: $0 <data_dir> <output_dir> <model_path> [flags...]"
    echo ""
    echo "  data_dir    RAM dataset (tokens.bin, offsets.npy, etc.)"
    echo "  output_dir  Experiment output (checkpoints, logs)"
    echo "  model_path  Init model checkpoint"
    echo ""
    echo "All extra flags passed to train/diffusion_finet"
    exit 1
fi

DATA_DIR="$1"
OUTPUT_DIR="$2"
MODEL_PATH="$3"
shift 3

NGPU=$(nvidia-smi -L | wc -l)
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "============================================================"
echo "  GeneT5 H100 Training"
echo "============================================================"
echo "  GPUs:       ${NGPU}x H100 SXM"
echo "  Code:       ${CODE_DIR}"
echo "  Data:       ${DATA_DIR}"
echo "  Output:     ${OUTPUT_DIR}"
echo "  Model:      ${MODEL_PATH}"
echo "  Extra args: $*"
echo "============================================================"

# FA4 on Hopper
export GENET5_ATTN_BACKEND=fa4

# NCCL — NVLink handles everything, minimal config
export NCCL_DEBUG=WARN
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=1

# PyTorch memory
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8"
export CUDA_MODULE_LOADING=LAZY

# Training env
export PYTHONUNBUFFERED=1
export PYTHONPATH="${CODE_DIR}"

mkdir -p "${OUTPUT_DIR}"

torchrun \
    --standalone \
    --nproc_per_node="${NGPU}" \
    "${CODE_DIR}/train/diffusion_finet" \
    "${DATA_DIR}" \
    "${OUTPUT_DIR}" \
    "${MODEL_PATH}" \
    --memwatch \
    "$@"
