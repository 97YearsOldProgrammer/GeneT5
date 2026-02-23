#!/bin/bash
#
# Distributed entropy model training across two DGX Sparks
#
# One-click (from master):
#   entropy/entropy.sh <raw_dir> --output <dir> --worker <WORKER_IP> [flags]
#
# All extra flags (--epochs, --lr, --batch_size, etc.) passed to entropy/train_entropy
#

set -e

MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29501}"
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
            local_tmp="${iface#en}"
            CX7_HCA="roce${local_tmp%np*}"
            break
        fi
    fi
done

# Parse args — first positional is data_dir, rest are flags
DATA_DIR=""
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
            fi
            shift
            ;;
    esac
done

if [ -z "$DATA_DIR" ]; then
    echo "Usage: $0 <raw_dir> --output <dir> [--worker <IP>] [flags]"
    echo ""
    echo "  raw_dir: raw FASTA directory with species subdirectories"
    echo ""
    echo "One-click:"
    echo "  --worker IP         Worker node IP"
    echo "  --worker-user USER  SSH user (default: cg666)"
    echo "  --container NAME    Docker container (default: gt5)"
    echo ""
    echo "All other flags passed to entropy/train_entropy (--epochs, --lr, --batch_size, etc.)"
    exit 1
fi

# ── Auto-detect master IP when using --worker ──
if [[ -n "$WORKER_IP" && "$MASTER_ADDR" == "localhost" ]]; then
    WORKER_SUBNET=$(echo "$WORKER_IP" | grep -oP '^\d+\.\d+\.\d+\.')
    for ip in $(hostname -I 2>/dev/null); do
        if [[ "$ip" == ${WORKER_SUBNET}* ]]; then
            MASTER_ADDR="$ip"
            break
        fi
    done
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

for env in "${NCCL_ENVS[@]}"; do
    export "$env"
done
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8"
# Auto-detect code directory (master vs worker may differ)
if [ -d "/workspace/Code/GeneT5" ]; then
    CODEDIR="/workspace/Code/GeneT5"
elif [ -d "/workspace/GeneT5" ]; then
    CODEDIR="/workspace/GeneT5"
else
    CODEDIR="$(pwd)"
fi
export PYTHONPATH="$CODEDIR"


####################
#####  Header  #####
####################


echo "========================================"
echo "  DNA Entropy Model - Distributed"
echo "========================================"
echo "Master:          $MASTER_ADDR:$MASTER_PORT"
echo "Nodes:           $NNODES (this is node $NODE_RANK)"
echo "Procs per node:  $NPROC_PER_NODE"
echo "Network iface:   ${CX7_IF:-auto}"
if [[ -n "$WORKER_IP" ]]; then
    echo "Mode:            one-click (auto-launching worker)"
    echo "Worker:          ${WORKER_USER}@${WORKER_IP} (container: ${CONTAINER})"
fi
echo "Raw dir:         $DATA_DIR"
echo "Extra args:      ${EXTRA_ARGS[*]}"
echo "========================================"


##################################
#####  Pre-flight Cleanup   #####
##################################


echo "[preflight] Clearing stale caches..."
find /workspace/Code/GeneT5 -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
rm -rf /tmp/torchinductor_root/ 2>/dev/null || true
sync && echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || echo "[preflight] WARNING: drop_caches failed"


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
        entropy/train_entropy \
        $DATA_DIR \
        ${EXTRA_ARGS[*]}"
}


##############################
#####  One-Click Launch  #####
##############################


if [[ -n "$WORKER_IP" ]]; then
    SSH_CMD="ssh -F /dev/null -o StrictHostKeyChecking=no -o IdentityFile=~/.ssh/id_ed25519_spark -o IdentitiesOnly=yes ${WORKER_USER}@${WORKER_IP}"
    WORKER_STATE=$(${SSH_CMD} "docker inspect -f '{{.State.Status}}' ${CONTAINER} 2>/dev/null" || echo "missing")

    case "$WORKER_STATE" in
        running)
            echo "[worker] Container already running"
            ;;
        exited|created)
            echo "[worker] Container stopped, restarting..."
            ${SSH_CMD} "docker start ${CONTAINER}"
            ;;
        *)
            echo "[worker] ERROR: Container not found"
            exit 1
            ;;
    esac

    # Build env string — worker detects its own CX7 interface
    ENV_STR=""
    for env in "${NCCL_ENVS[@]}"; do
        case "$env" in
            NCCL_SOCKET_IFNAME=*|GLOO_SOCKET_IFNAME=*|NCCL_IB_HCA=*) ;;
            *) ENV_STR+="export $env; " ;;
        esac
    done
    ENV_STR+="export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8; "
    # Worker auto-detects code directory
    ENV_STR+="if [ -d /workspace/Code/GeneT5 ]; then CODEDIR=/workspace/Code/GeneT5; "
    ENV_STR+="elif [ -d /workspace/GeneT5 ]; then CODEDIR=/workspace/GeneT5; "
    ENV_STR+="else CODEDIR=/workspace; fi; "
    ENV_STR+="export PYTHONPATH=\$CODEDIR; "
    # Worker detects its own CX7 interface + HCA
    ENV_STR+="for iface in enP2p1s0f1np1 enP2p1s0f0np0 enp1s0f1np1 enp1s0f0np0; do "
    ENV_STR+="  if [ -d /sys/class/net/\$iface ] && python3 -c \"import socket,struct,fcntl;s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);socket.inet_ntoa(fcntl.ioctl(s.fileno(),0x8915,struct.pack('256s',b'\$iface'))[20:24])\" 2>/dev/null; then "
    ENV_STR+="    export NCCL_SOCKET_IFNAME=\$iface; export GLOO_SOCKET_IFNAME=\$iface; "
    ENV_STR+="    tmp=\${iface#en}; export NCCL_IB_HCA=roce\${tmp%np*}; break; "
    ENV_STR+="  fi; "
    ENV_STR+="done; "
    ENV_STR+="cd \$CODEDIR; "
    # Pre-flight cleanup on worker
    ENV_STR+="find \$CODEDIR -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true; "
    ENV_STR+="rm -rf /tmp/torchinductor_root/ 2>/dev/null || true; "
    ENV_STR+="sync && echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true; "

    WORKER_CMD="${ENV_STR}$(build_torchrun_cmd 1)"

    MASTER_PID=""
    WORKER_PID=""
    CLEANED_UP=0

    SSH_BASE="ssh -F /dev/null -o StrictHostKeyChecking=no -o IdentityFile=~/.ssh/id_ed25519_spark -o IdentitiesOnly=yes"

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
        if [ -n "$MASTER_PID" ] && kill -0 "$MASTER_PID" 2>/dev/null; then
            kill_tree "$MASTER_PID"
        fi
        if [ -n "$WORKER_PID" ] && kill -0 "$WORKER_PID" 2>/dev/null; then
            kill_tree "$WORKER_PID"
        fi
        pkill -9 -f 'torchrun.*train_entropy' 2>/dev/null || true
        pkill -9 -f 'python.*train_entropy' 2>/dev/null || true
    }

    kill_remote() {
        (
            timeout 10 ${SSH_BASE} "${WORKER_USER}@${WORKER_IP}" \
                "docker exec ${CONTAINER} bash -c 'pkill -9 -f torchrun; pkill -9 -f train_entropy'" \
                2>/dev/null \
            || timeout 15 ${SSH_BASE} "${WORKER_USER}@${WORKER_IP}" \
                "docker restart ${CONTAINER}" \
                2>/dev/null \
            || echo "[cleanup] Worker unreachable"
        ) &
        local kill_pid=$!
        local waited=0
        while kill -0 $kill_pid 2>/dev/null && [ $waited -lt 20 ]; do
            sleep 1
            waited=$((waited + 1))
        done
        kill -9 $kill_pid 2>/dev/null || true
        wait $kill_pid 2>/dev/null || true
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
        echo "[cleanup] Done"
    }
    trap cleanup EXIT
    trap 'cleanup; exit 130' INT
    trap 'cleanup; exit 143' TERM HUP PIPE

    # Launch worker via SSH (background)
    echo ""
    echo "[worker] Launching on ${WORKER_IP}..."
    ${SSH_BASE} "${WORKER_USER}@${WORKER_IP}" \
        "docker exec ${CONTAINER} bash -c '${WORKER_CMD}'" \
        > >(while IFS= read -r line; do echo "[worker] $line"; done) \
        2>&1 &
    WORKER_PID=$!

    sleep 3

    # Launch master locally
    echo "[master] Launching locally..."
    echo ""
    set -m
    eval "$(build_torchrun_cmd 0)" &
    MASTER_PID=$!
    set +m

    # Monitor loop
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
    fi

    eval "$(build_torchrun_cmd $NODE_RANK)"
fi
