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

# Detect ConnectX-7 P2P interface (must be up AND have an IP)
CX7_IF=""
CX7_HCA=""
for iface in enP2p1s0f1np1 enP2p1s0f0np0 enp1s0f1np1 enp1s0f0np0; do
    if [ -d "/sys/class/net/$iface" ] && [ "$(cat /sys/class/net/$iface/operstate 2>/dev/null)" = "up" ]; then
        HAS_IP=$(python3 -c "
import socket,struct,fcntl; s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
try: print(socket.inet_ntoa(fcntl.ioctl(s.fileno(),0x8915,struct.pack('256s',b'$iface'))[20:24]))
except: pass" 2>/dev/null)
        if [ -n "$HAS_IP" ]; then
            CX7_IF="$iface"
            # Derive HCA name: enP2p1s0f1np1 -> roceP2p1s0f1
            local_tmp="${iface#en}"
            CX7_HCA="roce${local_tmp%np*}"
            break
        fi
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
    # Find our IP on the same subnet as the worker (for rendezvous)
    WORKER_SUBNET=$(echo "$WORKER_IP" | grep -oP '^\d+\.\d+\.\d+\.')
    for ip in $(hostname -I 2>/dev/null); do
        if [[ "$ip" == ${WORKER_SUBNET}* ]]; then
            MASTER_ADDR="$ip"
            break
        fi
    done
    # Fallback to first IP
    if [[ "$MASTER_ADDR" == "localhost" ]]; then
        MASTER_ADDR=$(hostname -I 2>/dev/null | awk '{print $1}')
    fi
fi


#############################
#####  NCCL Environment #####
#############################


NCCL_ENVS=(
    "NCCL_DEBUG=INFO"
    "NCCL_IB_HCA=${CX7_HCA:-roceP2p1s0f1}"
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
    WORKER_STATE=$(${SSH_CMD} "docker inspect -f '{{.State.Status}}' ${CONTAINER} 2>/dev/null" || echo "missing")

    case "$WORKER_STATE" in
        running)
            echo "[worker] Container already running"
            ;;
        exited|created)
            echo "[worker] Container stopped, restarting (packages intact)..."
            ${SSH_CMD} "docker start ${CONTAINER}"
            ;;
        *)
            echo "[worker] No container found, creating fresh (will pip install)..."
            ${SSH_CMD} "cd /home/cg666/Code/GeneT5 && bash start-worker.sh --daemon"
            echo "[worker] Waiting for pip install..."
            for i in $(seq 1 30); do
                if ${SSH_CMD} "docker exec ${CONTAINER} python -c 'import liger_kernel' 2>/dev/null"; then
                    break
                fi
                if [[ $i -eq 30 ]]; then
                    echo "[worker] ERROR: pip install timed out after 60s"
                    exit 1
                fi
                sleep 2
            done
            ;;
    esac

    # Auto-patch NGC triton cluster_dims bug on both nodes (idempotent)
    TRITON_FILE="/usr/local/lib/python3.12/dist-packages/torch/_inductor/runtime/triton_heuristics.py"
    PATCH_MARKER='getattr(binary.metadata, "cluster_dims"'
    PATCH_CMD="grep -q '${PATCH_MARKER}' ${TRITON_FILE} 2>/dev/null || \
        sed -i 's|(binary.metadata.num_ctas, \*binary.metadata.cluster_dims)|(binary.metadata.num_ctas, *(getattr(binary.metadata, \"cluster_dims\", None) or getattr(binary.metadata, \"clusterDims\", (1, 1, 1))))|' ${TRITON_FILE}"

    # Patch worker
    if ! ${SSH_CMD} "docker exec ${CONTAINER} bash -c '${PATCH_CMD}'" 2>/dev/null; then
        echo "[worker] WARNING: triton patch failed (non-fatal)"
    fi
    # Patch master
    eval "$PATCH_CMD" 2>/dev/null || true

    # Build env string for docker exec (skip per-node vars — worker detects its own)
    ENV_STR=""
    for env in "${NCCL_ENVS[@]}"; do
        case "$env" in
            NCCL_SOCKET_IFNAME=*|GLOO_SOCKET_IFNAME=*|NCCL_IB_HCA=*) ;;
            *) ENV_STR+="export $env; " ;;
        esac
    done
    ENV_STR+="export BNB_CUDA_VERSION=130; "
    ENV_STR+="export PYTHONPATH=/workspace/GeneT5; "
    # Worker detects its own CX7 interface + HCA (may differ from master)
    ENV_STR+="for iface in enP2p1s0f1np1 enP2p1s0f0np0 enp1s0f1np1 enp1s0f0np0; do "
    ENV_STR+="  if [ -d /sys/class/net/\$iface ] && python3 -c \"import socket,struct,fcntl;s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);socket.inet_ntoa(fcntl.ioctl(s.fileno(),0x8915,struct.pack('256s',b'\$iface'))[20:24])\" 2>/dev/null; then "
    ENV_STR+="    export NCCL_SOCKET_IFNAME=\$iface; export GLOO_SOCKET_IFNAME=\$iface; "
    ENV_STR+="    tmp=\${iface#en}; export NCCL_IB_HCA=roce\${tmp%np*}; break; "
    ENV_STR+="  fi; "
    ENV_STR+="done; "
    ENV_STR+="cd /workspace/GeneT5; "

    WORKER_CMD="${ENV_STR}$(build_torchrun_cmd 1)"

    MASTER_PID=""
    WORKER_PID=""

    # Kill both sides on exit, Ctrl+C, or if either side dies
    kill_local_tree() {
        # Kill the entire process group: torchrun + all child python workers
        [ -n "$MASTER_PID" ] && kill -- -$MASTER_PID 2>/dev/null || true
        # Also pkill by pattern in case process group kill missed anything
        pkill -f 'torchrun.*finet' 2>/dev/null || true
        pkill -f 'python.*bin/finet' 2>/dev/null || true
    }

    kill_remote_tree() {
        ssh -F /dev/null -o StrictHostKeyChecking=no \
            -o IdentityFile=~/.ssh/id_ed25519_spark -o IdentitiesOnly=yes \
            "${WORKER_USER}@${WORKER_IP}" \
            "docker exec ${CONTAINER} bash -c 'pkill -f torchrun.*finet; pkill -f python.*bin/finet'" 2>/dev/null || true
    }

    force_kill() {
        # Escalate to SIGKILL if anything survived
        sleep 2
        pkill -9 -f 'torchrun.*finet' 2>/dev/null || true
        pkill -9 -f 'python.*bin/finet' 2>/dev/null || true
        ssh -F /dev/null -o StrictHostKeyChecking=no \
            -o IdentityFile=~/.ssh/id_ed25519_spark -o IdentitiesOnly=yes \
            "${WORKER_USER}@${WORKER_IP}" \
            "docker exec ${CONTAINER} bash -c 'pkill -9 -f torchrun.*finet; pkill -9 -f python.*bin/finet'" 2>/dev/null || true
    }

    cleanup() {
        trap - EXIT INT TERM
        echo ""
        echo "[cleanup] Shutting down both nodes..."
        kill_local_tree
        kill_remote_tree
        wait $MASTER_PID 2>/dev/null
        wait $WORKER_PID 2>/dev/null
        # Force kill stragglers
        if pgrep -f 'python.*bin/finet' >/dev/null 2>&1; then
            echo "[cleanup] Stragglers detected, sending SIGKILL..."
            force_kill
        fi
        echo "[cleanup] Done"
    }
    trap cleanup EXIT INT TERM

    # Launch worker via SSH (background)
    echo ""
    echo "[worker] Launching on ${WORKER_IP}..."
    ssh -F /dev/null -o StrictHostKeyChecking=no -o IdentityFile=~/.ssh/id_ed25519_spark -o IdentitiesOnly=yes "${WORKER_USER}@${WORKER_IP}" \
        "docker exec ${CONTAINER} bash -c '${WORKER_CMD}'" \
        > >(while IFS= read -r line; do echo "[worker] $line"; done) \
        2>&1 &
    WORKER_PID=$!

    # Give worker a head start to open the rendezvous port listener
    sleep 3

    # Launch master locally (background, in own process group for clean kill)
    echo "[master] Launching locally..."
    echo ""
    set -m
    eval "$(build_torchrun_cmd 0)" &
    MASTER_PID=$!
    set +m

    # Wait for first process to finish
    MASTER_EXIT=0
    WORKER_EXIT=0
    wait -n $MASTER_PID $WORKER_PID 2>/dev/null || true

    # Determine which finished and kill the other
    if ! kill -0 $MASTER_PID 2>/dev/null; then
        wait $MASTER_PID 2>/dev/null; MASTER_EXIT=$?
        echo ""
        echo "[master] Exited (code=$MASTER_EXIT), stopping worker..."
        kill_remote_tree
        wait $WORKER_PID 2>/dev/null; WORKER_EXIT=$?
    elif ! kill -0 $WORKER_PID 2>/dev/null; then
        wait $WORKER_PID 2>/dev/null; WORKER_EXIT=$?
        echo ""
        echo "[worker] Exited (code=$WORKER_EXIT), stopping master..."
        kill_local_tree
        wait $MASTER_PID 2>/dev/null; MASTER_EXIT=$?
    fi

    # Final sweep — nuke anything still alive
    if pgrep -f 'python.*bin/finet' >/dev/null 2>&1; then
        echo "[cleanup] Stragglers detected, sending SIGKILL..."
        force_kill
    fi

    # Disable trap — we already cleaned up
    trap - EXIT INT TERM

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
