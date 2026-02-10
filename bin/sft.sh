#!/bin/bash
#
# Distributed fine-tuning across two DGX Sparks
#
# One-click (from master):
#   bin/distributed.sh <train> <val> <output> <model> --worker <WORKER_IP>
#
# Manual (run on each node separately):
#   Master: bin/distributed.sh <train> <val> <output> <model> --nnodes 2 --node-rank 0 --master <MASTER_IP>
#   Worker: bin/distributed.sh <train> <val> <output> <model> --nnodes 2 --node-rank 1 --master <MASTER_IP>
#
# All extra flags (--epochs, --lr, --batch_size, etc.) are passed to bin/finet
#

set -e

MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

WORKER_IP=""
WORKER_USER="cg666"
CONTAINER="gt5"

# Detect ConnectX-7 P2P interface
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

while [[ $# -gt 0 ]]; do
    case $1 in
        --worker)
            WORKER_IP="$2"
            NNODES=2
            NODE_RANK=0
            shift 2
            ;;
        --worker-user)
            WORKER_USER="$2"
            shift 2
            ;;
        --container)
            CONTAINER="$2"
            shift 2
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
    echo "One-click (launches worker automatically):"
    echo "  --worker IP         Worker node IP (triggers one-click mode)"
    echo "  --worker-user USER  SSH user on worker (default: cg666)"
    echo "  --container NAME    Docker container name (default: gt5)"
    echo ""
    echo "Manual mode:"
    echo "  --nnodes N          Number of nodes (default: 1)"
    echo "  --node-rank N       This node's rank (0=master, 1=worker)"
    echo "  --master IP         Master node IP (default: localhost)"
    echo ""
    echo "Common:"
    echo "  --port PORT         Master port (default: 29500)"
    echo "  --nproc N           Processes per node (default: 1)"
    echo ""
    echo "All other flags passed to bin/finet (--epochs, --lr, --batch_size, etc.)"
    exit 1
fi

# ── Auto-detect master IP when using --worker ──
if [[ -n "$WORKER_IP" && "$MASTER_ADDR" == "localhost" ]]; then
    # Try to get our IP on the P2P interface
    for iface in enp1s0f1np1 enP2p1s0f1np1 enp1s0f0np0 enP7s7; do
        DETECTED_IP=$(ip -4 addr show "$iface" 2>/dev/null | grep -oP 'inet \K[^/]+' || true)
        if [[ -n "$DETECTED_IP" ]]; then
            MASTER_ADDR="$DETECTED_IP"
            break
        fi
    done
    # Fallback to hostname
    if [[ "$MASTER_ADDR" == "localhost" ]]; then
        MASTER_ADDR=$(hostname -I 2>/dev/null | awk '{print $1}')
    fi
fi


#############################
#####  NCCL Environment #####
#############################


NCCL_ENVS=(
    "NCCL_DEBUG=INFO"
    "NCCL_IB_HCA=roceP2p1s0f1"
    "NCCL_IB_TIMEOUT=22"
    "NCCL_IB_RETRY_CNT=7"
    "NCCL_IB_MERGE_NICS=0"
    "NCCL_CROSS_NIC=0"
    "NCCL_MAX_NCHANNELS=2"
    "NCCL_NET_GDR_LEVEL=0"
    "NCCL_NET_GDR_READ=0"
)

if [ -n "$CX7_IF" ]; then
    NCCL_ENVS+=("NCCL_SOCKET_IFNAME=$CX7_IF" "GLOO_SOCKET_IFNAME=$CX7_IF")
fi

# Export locally
for env in "${NCCL_ENVS[@]}"; do
    export "$env"
done
export BNB_CUDA_VERSION=130
export PYTHONPATH="${PYTHONPATH:-/workspace/GeneT5}"


####################
#####  Header  #####
####################


echo "========================================"
echo "  DGX Spark Distributed Fine-Tuning"
echo "========================================"
echo "Master:          $MASTER_ADDR:$MASTER_PORT"
echo "Nodes:           $NNODES (this is node $NODE_RANK)"
echo "Procs per node:  $NPROC_PER_NODE"
echo "Network iface:   ${CX7_IF:-auto}"
if [[ -n "$WORKER_IP" ]]; then
    echo "Mode:            one-click (auto-launching worker)"
    echo "Worker:          ${WORKER_USER}@${WORKER_IP} (container: ${CONTAINER})"
fi
echo "Train data:      $TRAIN_DATA"
echo "Val data:        $VAL_DATA"
echo "Output dir:      $OUTPUT_DIR"
echo "Model path:      $MODEL_PATH"
echo "Extra args:      ${EXTRA_ARGS[*]}"
echo "========================================"


#############################
#####  Torchrun Command #####
#############################


build_torchrun_cmd() {
    local rank="$1"
    echo "torchrun \
        --nnodes=$NNODES \
        --nproc_per_node=$NPROC_PER_NODE \
        --node_rank=$rank \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        bin/finet \
        $TRAIN_DATA \
        $VAL_DATA \
        $OUTPUT_DIR \
        $MODEL_PATH \
        --memwatch \
        ${EXTRA_ARGS[*]}"
}


##############################
#####  One-Click Launch  #####
##############################


if [[ -n "$WORKER_IP" ]]; then
    # Ensure worker container is running
    SSH_CMD="ssh -F /dev/null -o StrictHostKeyChecking=no -o IdentityFile=~/.ssh/id_ed25519_spark -o IdentitiesOnly=yes ${WORKER_USER}@${WORKER_IP}"
    WORKER_RUNNING=$(${SSH_CMD} "docker inspect -f '{{.State.Running}}' ${CONTAINER} 2>/dev/null" || echo "false")
    if [[ "$WORKER_RUNNING" != "true" ]]; then
        echo "[worker] Container not running, starting in daemon mode..."
        ${SSH_CMD} "cd /home/cg666/Code/GeneT5 && bash start-worker.sh --daemon"
        echo "[worker] Waiting for container setup (pip install)..."
        for i in $(seq 1 30); do
            if ${SSH_CMD} "docker exec ${CONTAINER} python -c 'import liger_kernel' 2>/dev/null"; then
                break
            fi
            sleep 2
        done
    fi

    # Build env string for docker exec
    ENV_STR=""
    for env in "${NCCL_ENVS[@]}"; do
        ENV_STR+="export $env; "
    done
    ENV_STR+="export BNB_CUDA_VERSION=130; "
    ENV_STR+="export PYTHONPATH=/workspace/GeneT5; "
    ENV_STR+="cd /workspace/GeneT5; "

    WORKER_CMD="${ENV_STR}$(build_torchrun_cmd 1)"

    # Launch worker via SSH
    echo ""
    echo "[worker] Launching on ${WORKER_IP}..."
    ssh -F /dev/null -o StrictHostKeyChecking=no -o IdentityFile=~/.ssh/id_ed25519_spark -o IdentitiesOnly=yes "${WORKER_USER}@${WORKER_IP}" \
        "docker exec ${CONTAINER} bash -c '${WORKER_CMD}'" \
        > >(while IFS= read -r line; do echo "[worker] $line"; done) \
        2>&1 &
    WORKER_PID=$!

    # Give worker a head start to open the rendezvous port listener
    sleep 3

    # Launch master locally
    echo "[master] Launching locally..."
    echo ""
    eval "$(build_torchrun_cmd 0)"
    MASTER_EXIT=$?

    # Wait for worker
    wait $WORKER_PID || true
    WORKER_EXIT=$?

    echo ""
    echo "========================================"
    echo "  Training Complete"
    echo "  Master exit: ${MASTER_EXIT}"
    echo "  Worker exit: ${WORKER_EXIT}"
    echo "========================================"

    exit $MASTER_EXIT


########################
#####  Manual Mode #####
########################


else
    if [ -n "$CX7_IF" ]; then
        echo "Using ConnectX-7 interface: $CX7_IF"
    else
        echo "WARNING: No ConnectX-7 interface found, using default"
        echo "         Make sure container runs with --network=host"
    fi

    eval "$(build_torchrun_cmd $NODE_RANK)"
fi
