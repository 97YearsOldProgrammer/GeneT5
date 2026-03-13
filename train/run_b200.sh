#!/bin/bash
# B200 single-node training launcher with tmux persistence
#
# Wraps torchrun in a tmux session so training survives terminal disconnect
#
# Usage:
#   bash train/run_b200.sh <data_dir> <output_dir> <model_path> [flags]
#
# Examples:
#   bash train/run_b200.sh /workspace/baked/GeneT5/mar06_s51_w20k_p18k \
#       /workspace/model/GeneT5/mar10_b200_t1 \
#       /workspace/model/GeneT5/init/init_dense_24L \
#       --epochs 3 --batch_size 8 --grad_accum 32 --lr 1e-4 --compile --fused_ce
#
#   tmux attach -t sft         # reattach to running session
#   tmux kill-session -t sft   # stop training

set -euo pipefail

SESSION="sft"

if [ $# -lt 3 ]; then
    echo "Usage: $0 <data_dir> <output_dir> <model_path> [training flags]"
    echo ""
    echo "Launches training in tmux session '${SESSION}'"
    echo "  tmux attach -t ${SESSION}    # reattach"
    echo "  tmux kill-session -t ${SESSION}  # stop"
    exit 1
fi

DATA_DIR="$1"
OUTPUT_DIR="$2"
MODEL_PATH="$3"
shift 3
EXTRA_ARGS="$*"

# Auto-detect code directory
CODE_DIR=""
for d in /workspace/code /workspace/GeneT5 /workspace/Code/GeneT5; do
    [ -f "$d/train/diffusion_finet" ] && CODE_DIR="$d" && break
done
if [ -z "$CODE_DIR" ]; then
    CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

# Validate paths
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found: $DATA_DIR"
    exit 1
fi
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model path not found: $MODEL_PATH"
    exit 1
fi

# Kill existing session if present
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "WARNING: tmux session '${SESSION}' already exists"
    read -p "Kill existing session and start new? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        tmux kill-session -t "$SESSION"
    else
        echo "Attach with: tmux attach -t ${SESSION}"
        exit 0
    fi
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Log file
LOG_DIR="/workspace/data/logs/sft"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/$(date +%Y%m%d_%H%M%S)_b200.log"

# Build environment block
ENV_BLOCK="
export PYTHONPATH=${CODE_DIR}
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.9,max_split_size_mb:512
export CUDA_MODULE_LOADING=LAZY
export GENET5_ATTN_BACKEND=auto
export TRITON_CACHE_DIR=/workspace/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=/workspace/.cache/torchinductor
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_COMPILE_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_P2P_LEVEL=NVL
export NCCL_NET_GDR_LEVEL=5
export NCCL_NET_GDR_READ=1
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
"

# Auto-detect GPU count
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
NUM_GPUS=${NUM_GPUS:-1}

# Build training command
TRAIN_CMD="torchrun \
    --nnodes=1 \
    --nproc_per_node=${NUM_GPUS} \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    train/diffusion_finet \
    ${DATA_DIR} \
    ${OUTPUT_DIR} \
    ${MODEL_PATH} \
    --memwatch \
    ${EXTRA_ARGS}"

# Pre-flight
echo "=== B200 Training Launch ==="
echo "GPUs:     ${NUM_GPUS}"
echo "Code:     ${CODE_DIR}"
echo "Data:     ${DATA_DIR}"
echo "Output:   ${OUTPUT_DIR}"
echo "Model:    ${MODEL_PATH}"
echo "Extra:    ${EXTRA_ARGS}"
echo "Log:      ${LOG_FILE}"
echo "Session:  tmux attach -t ${SESSION}"
echo "==========================="

# Clear pycache
find "$CODE_DIR" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

# Launch in tmux with full env + tee to log
tmux new-session -d -s "$SESSION" "
${ENV_BLOCK}
cd ${CODE_DIR}
echo '=== Training started: $(date) ===' | tee ${LOG_FILE}
echo 'GPU: '
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a ${LOG_FILE}
echo '' | tee -a ${LOG_FILE}
${TRAIN_CMD} 2>&1 | tee -a ${LOG_FILE}
EXIT_CODE=\${PIPESTATUS[0]}
echo '' | tee -a ${LOG_FILE}
echo \"=== Training finished: \$(date), exit code: \${EXIT_CODE} ===\" | tee -a ${LOG_FILE}
echo 'Press any key to close this session...'
read -n 1
"

echo ""
echo "Training launched in tmux session '${SESSION}'"
echo ""
echo "Commands:"
echo "  tmux attach -t ${SESSION}        # watch live output"
echo "  tmux kill-session -t ${SESSION}   # stop training"
echo "  tail -f ${LOG_FILE}              # follow log without attaching"
