#!/bin/bash
#
# Distributed fine-tuning across two DGX Sparks
#
# One-click (from master):
#   train/sft.sh <data_dir> <output> <model> --worker <WORKER_IP>
#
# Manual (run on each node separately):
#   Master: train/sft.sh <data_dir> <output> <model> --nnodes 2 --node-rank 0 --master <MASTER_IP>
#   Worker: train/sft.sh <data_dir> <output> <model> --nnodes 2 --node-rank 1 --master <MASTER_IP>
#
# All extra flags (--epochs, --lr, --batch_size, etc.) are passed to train/diffusion_finet
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
DATA_DIR=""
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
            if [ -z "$DATA_DIR" ]; then
                DATA_DIR="$1"
            elif [ -z "$OUTPUT_DIR" ]; then
                OUTPUT_DIR="$1"
            elif [ -z "$MODEL_PATH" ]; then
                MODEL_PATH="$1"
            fi
            shift
            ;;
    esac
done

if [ -z "$DATA_DIR" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$MODEL_PATH" ]; then
    echo "Usage: $0 <data_dir> <output_dir> <model_path> [options]"
    echo ""
    echo "  data_dir: bake output directory (contains training.bin, validation.bin, eval.json)"
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
    echo "All other flags passed to train/diffusion_finet (--epochs, --lr, --batch_size, etc.)"
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
    "NCCL_IB_TIMEOUT=23"
    "NCCL_IB_RETRY_CNT=13"
    "NCCL_IB_MERGE_NICS=0"
    "NCCL_CROSS_NIC=0"
    "NCCL_MAX_NCHANNELS=2"
    "NCCL_NET_GDR_LEVEL=0"
    "NCCL_NET_GDR_READ=0"
    "NCCL_TIMEOUT=1800"
    "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800"
)

if [ -n "$CX7_IF" ]; then
    NCCL_ENVS+=("NCCL_SOCKET_IFNAME=$CX7_IF" "GLOO_SOCKET_IFNAME=$CX7_IF")
fi

# Export locally
for env in "${NCCL_ENVS[@]}"; do
    export "$env"
done
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8"
# Auto-detect master code dir (same logic as worker detection)
MASTER_CODE_DIR=""
for d in /workspace/Code/GeneT5 /workspace/GeneT5; do
    [ -f "$d/train/diffusion_finet" ] && MASTER_CODE_DIR="$d" && break
done
if [[ -z "$MASTER_CODE_DIR" ]]; then
    MASTER_CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
export PYTHONPATH="$MASTER_CODE_DIR"


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
echo "Data dir:        $DATA_DIR"
echo "Output dir:      $OUTPUT_DIR"
echo "Model path:      $MODEL_PATH"
echo "Extra args:      ${EXTRA_ARGS[*]}"
echo "========================================"


##################################
#####  Pre-flight Cleanup   #####
##################################


cd "$MASTER_CODE_DIR"
echo "[preflight] Clearing stale caches..."
find "$MASTER_CODE_DIR" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
rm -rf /tmp/torchinductor_root/ 2>/dev/null || true
sync && echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || echo "[preflight] WARNING: drop_caches failed (need root)"

# Warn if swap is active (should be disabled on host: sudo swapoff -a && sudo sed -i '/swap/s/^/#/' /etc/fstab)
if grep -q 'swap' /proc/swaps 2>/dev/null; then
    SWAP_USED=$(awk 'NR>1{s+=$4} END{printf "%.0f", s/1024}' /proc/swaps 2>/dev/null)
    echo "[preflight] WARNING: swap active (${SWAP_USED}MB used) — disable on host: sudo swapoff -a"
fi


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
        train/diffusion_finet \
        $DATA_DIR \
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
            ${SSH_CMD} "cd /home/cg666/Code && bash start-worker.sh --daemon"
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

    # Detect GeneT5 code path on worker (differs from master mount layout)
    WORKER_CODE_DIR=$(${SSH_CMD} "docker exec ${CONTAINER} bash -c 'for d in /workspace/Code/GeneT5 /workspace/GeneT5; do [ -f \$d/train/diffusion_finet ] && echo \$d && break; done'" 2>/dev/null)
    if [[ -z "$WORKER_CODE_DIR" ]]; then
        echo "ERROR: Cannot find GeneT5 code directory on worker"
        exit 1
    fi
    echo "[worker] Code directory: ${WORKER_CODE_DIR}"

    # Sync requirements on worker (idempotent, picks up webdataset etc.)
    echo "[worker] Syncing pip requirements..."
    ${SSH_CMD} "docker exec ${CONTAINER} pip install -q -r ${WORKER_CODE_DIR}/requirements.txt" 2>/dev/null || true

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
    ENV_STR+="export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8; "
    ENV_STR+="export PYTHONPATH=${WORKER_CODE_DIR}; "
    # Worker detects its own CX7 interface + HCA (may differ from master)
    ENV_STR+="for iface in enP2p1s0f1np1 enP2p1s0f0np0 enp1s0f1np1 enp1s0f0np0; do "
    ENV_STR+="  if [ -d /sys/class/net/\$iface ] && python3 -c \"import socket,struct,fcntl;s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);socket.inet_ntoa(fcntl.ioctl(s.fileno(),0x8915,struct.pack('256s',b'\$iface'))[20:24])\" 2>/dev/null; then "
    ENV_STR+="    export NCCL_SOCKET_IFNAME=\$iface; export GLOO_SOCKET_IFNAME=\$iface; "
    ENV_STR+="    tmp=\${iface#en}; export NCCL_IB_HCA=roce\${tmp%np*}; break; "
    ENV_STR+="  fi; "
    ENV_STR+="done; "
    ENV_STR+="cd ${WORKER_CODE_DIR}; "
    # Pre-flight cleanup on worker (pycache, inductor, page cache)
    ENV_STR+="find ${WORKER_CODE_DIR} -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true; "
    ENV_STR+="rm -rf /tmp/torchinductor_root/ 2>/dev/null || true; "
    ENV_STR+="sync && echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true; "

    WORKER_CMD="${ENV_STR}$(build_torchrun_cmd 1)"

    MASTER_PID=""
    WORKER_PID=""
    CLEANED_UP=0

    SSH_BASE="ssh -F /dev/null -o StrictHostKeyChecking=no -o IdentityFile=~/.ssh/id_ed25519_spark -o IdentitiesOnly=yes"

    # Kill a process and all its descendants with SIGKILL
    kill_tree() {
        local pid=$1
        local children
        children=$(pgrep -P "$pid" 2>/dev/null) || true
        for child in $children; do
            kill_tree "$child"
        done
        kill -9 "$pid" 2>/dev/null || true
    }

    kill_local() {
        # Kill master process tree
        if [ -n "$MASTER_PID" ] && kill -0 "$MASTER_PID" 2>/dev/null; then
            kill_tree "$MASTER_PID"
        fi
        # Kill SSH tunnel to worker
        if [ -n "$WORKER_PID" ] && kill -0 "$WORKER_PID" 2>/dev/null; then
            kill_tree "$WORKER_PID"
        fi
        # Belt-and-suspenders: pattern kill (catches orphans + data loader workers)
        pkill -9 -f 'torchrun.*finet' 2>/dev/null || true
        pkill -9 -f 'python.*train/diffusion_finet' 2>/dev/null || true
    }

    kill_remote() {
        # Try docker exec first (fast path), then docker restart as fallback
        # Use background + wait to avoid blocking on unresponsive worker
        (
            timeout 10 ${SSH_BASE} "${WORKER_USER}@${WORKER_IP}" \
                "docker exec ${CONTAINER} bash -c 'pkill -9 -f torchrun; pkill -9 -f python.*train/diffusion_finet'" \
                2>/dev/null \
            || timeout 15 ${SSH_BASE} "${WORKER_USER}@${WORKER_IP}" \
                "docker restart ${CONTAINER}" \
                2>/dev/null \
            || echo "[cleanup] Worker unreachable, run on worker: docker restart ${CONTAINER}"
        ) &
        local kill_pid=$!
        # Wait up to 20s total, then give up
        local waited=0
        while kill -0 $kill_pid 2>/dev/null && [ $waited -lt 20 ]; do
            sleep 1
            waited=$((waited + 1))
        done
        kill -9 $kill_pid 2>/dev/null || true
        wait $kill_pid 2>/dev/null || true
    }

    verify_clean() {
        for attempt in 1 2 3; do
            if ! pgrep -f 'python.*train/diffusion_finet' >/dev/null 2>&1 && \
               ! pgrep -f 'torchrun.*finet' >/dev/null 2>&1; then
                return 0
            fi
            echo "[cleanup] Attempt $attempt/3: killing remaining processes..."
            pkill -9 -f 'torchrun' 2>/dev/null || true
            pkill -9 -f 'python.*finet' 2>/dev/null || true
            sleep 1
        done
        if pgrep -f 'python.*finet' >/dev/null 2>&1; then
            echo "[cleanup] WARNING: processes survived all kill attempts:"
            pgrep -af 'python.*finet' || true
        fi
    }

    cleanup() {
        [ "$CLEANED_UP" -eq 1 ] && return
        CLEANED_UP=1
        trap - EXIT INT TERM HUP PIPE
        echo ""
        echo "[cleanup] Shutting down both nodes..."
        kill_local
        kill_remote
        wait "$MASTER_PID" 2>/dev/null || true
        wait "$WORKER_PID" 2>/dev/null || true
        verify_clean
        echo "[cleanup] Done"
    }
    trap cleanup EXIT
    trap 'cleanup; exit 130' INT
    trap 'cleanup; exit 143' TERM PIPE
    trap '' HUP

    # Launch worker via SSH (background)
    echo ""
    echo "[worker] Launching on ${WORKER_IP}..."
    ${SSH_BASE} "${WORKER_USER}@${WORKER_IP}" \
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

    # Monitor loop: check both processes every 2s, exit as soon as either dies
    MASTER_EXIT=0
    WORKER_EXIT=0
    while true; do
        MASTER_ALIVE=true
        WORKER_ALIVE=true
        kill -0 "$MASTER_PID" 2>/dev/null || MASTER_ALIVE=false
        kill -0 "$WORKER_PID" 2>/dev/null || WORKER_ALIVE=false

        if ! $MASTER_ALIVE; then
            wait "$MASTER_PID" 2>/dev/null; MASTER_EXIT=$?
            echo ""
            echo "[master] Exited (code=$MASTER_EXIT), stopping worker..."
            kill_remote
            kill_tree "$WORKER_PID" 2>/dev/null || true
            wait "$WORKER_PID" 2>/dev/null; WORKER_EXIT=$?
            break
        fi

        if ! $WORKER_ALIVE; then
            wait "$WORKER_PID" 2>/dev/null; WORKER_EXIT=$?
            echo ""
            echo "[worker] Exited (code=$WORKER_EXIT), stopping master..."
            kill_tree "$MASTER_PID" 2>/dev/null || true
            wait "$MASTER_PID" 2>/dev/null; MASTER_EXIT=$?
            kill_remote
            break
        fi

        sleep 2
    done

    # Final verification (cleanup trap also runs on exit, but idempotent)
    verify_clean
    trap - EXIT INT TERM HUP PIPE
    CLEANED_UP=1

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
