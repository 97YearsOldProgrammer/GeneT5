#!/bin/bash
#
# Launch distributed fine-tuning across two DGX Spark systems
# Uses PyTorch distributed with NCCL backend over ConnectX-7 200Gb/s network
#
# Usage:
#   Single node:  bin/distributed.sh <train.bin> <val.bin> <output> <model>
#   Two Sparks:   bin/distributed.sh <train.bin> <val.bin> <output> <model> --nnodes 2 --node-rank 0 --master 192.168.100.10
#   (run same command on worker with --node-rank 1)
#

set -e

# Default configuration
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

# Detect ConnectX-7 P2P interface (DGX Spark — cable is on the P2P NIC)
CX7_IF=""
for iface in enP2p1s0f1np1 enP2p1s0f0np0 enp1s0f1np1 enp1s0f0np0; do
    if [ -d "/sys/class/net/$iface" ] && [ "$(cat /sys/class/net/$iface/operstate 2>/dev/null)" = "up" ]; then
        CX7_IF="$iface"
        break
    fi
done

# Positional args
TRAIN_DATA=""
VAL_DATA=""
OUTPUT_DIR=""
MODEL_PATH=""
EXTRA_ARGS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
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
        --*)
            EXTRA_ARGS+=("$1")
            if [[ $# -gt 1 && ! "$2" =~ ^-- ]]; then
                EXTRA_ARGS+=("$2")
                shift
            fi
            shift
            ;;
        *)
            if [ -z "$TRAIN_DATA" ]; then
                TRAIN_DATA="$1"
            elif [ -z "$VAL_DATA" ]; then
                VAL_DATA="$1"
            elif [ -z "$OUTPUT_DIR" ]; then
                OUTPUT_DIR="$1"
            elif [ -z "$MODEL_PATH" ]; then
                MODEL_PATH="$1"
            fi
            shift
            ;;
    esac
done

if [ -z "$TRAIN_DATA" ] || [ -z "$VAL_DATA" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$MODEL_PATH" ]; then
    echo "Usage: $0 <train.bin> <val.bin> <output_dir> <model_path> [options]"
    echo ""
    echo "Multi-node options:"
    echo "  --nnodes N        Number of nodes (default: 1)"
    echo "  --node-rank N     This node's rank (0=master, 1=worker)"
    echo "  --master IP       Master node IP (default: localhost)"
    echo "  --port PORT       Master port (default: 29500)"
    echo ""
    echo "All other flags are passed to bin/finet (--epochs, --lr, --batch_size, etc.)"
    exit 1
fi

echo "========================================"
echo "  DGX Spark Distributed Fine-Tuning"
echo "========================================"
echo "Master:          $MASTER_ADDR:$MASTER_PORT"
echo "Nodes:           $NNODES (this is node $NODE_RANK)"
echo "Procs per node:  $NPROC_PER_NODE"
echo "Network iface:   ${CX7_IF:-auto}"
echo "Train data:      $TRAIN_DATA"
echo "Val data:        $VAL_DATA"
echo "Output dir:      $OUTPUT_DIR"
echo "Model path:      $MODEL_PATH"
echo "Extra args:      ${EXTRA_ARGS[*]}"
echo "========================================"

# NCCL environment for DGX Spark ConnectX-7 200Gb/s RoCE
export NCCL_DEBUG=INFO

# RoCE transport — pin to P2P NIC (cable is on roceP2p1s0f1, not rocep1s0f1)
export NCCL_IB_HCA="roceP2p1s0f1"
export NCCL_IB_TIMEOUT=22               # Max timeout (~17s per retry)
export NCCL_IB_RETRY_CNT=7             # Max retries
export NCCL_IB_MERGE_NICS=0            # Don't merge NICs (avoids socket type mismatch)
export NCCL_CROSS_NIC=0                 # Stay on same NIC for all connections
export NCCL_MAX_NCHANNELS=2             # Fewer channels = fewer socket connections to negotiate

# GDR disabled: nvidia-peermem broken on DGX Spark UMA (DGX OS 7.x)
export NCCL_NET_GDR_LEVEL=0
export NCCL_NET_GDR_READ=0

# Fallback: uncomment these two lines to force Socket transport (~3.4 GB/s vs ~25 GB/s)
# export NCCL_NET_PLUGIN=none
# export NCCL_IB_DISABLE=1

# Point NCCL and Gloo at the ConnectX-7 interface (not Docker bridge)
if [ -n "$CX7_IF" ]; then
    export NCCL_SOCKET_IFNAME="$CX7_IF"
    export GLOO_SOCKET_IFNAME="$CX7_IF"
    echo "Using ConnectX-7 interface: $CX7_IF"
else
    echo "WARNING: No ConnectX-7 interface found, using default"
    echo "         Make sure container runs with --network=host"
fi

# Ensure GeneT5 lib is importable
export PYTHONPATH="${PYTHONPATH:-/workspace/GeneT5}"

torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    bin/finet \
    "$TRAIN_DATA" \
    "$VAL_DATA" \
    "$OUTPUT_DIR" \
    "$MODEL_PATH" \
    --optim_8bit \
    --memwatch \
    "${EXTRA_ARGS[@]}"