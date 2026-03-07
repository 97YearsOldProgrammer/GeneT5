#!/bin/bash
#
# Run DDP tuning benchmark across both DGX Spark nodes
#
# Usage: bash tests/run_ddp_tuning.sh [--channels N]
#
# Reuses the same NCCL setup as train/sft.sh

set -e

NCHANNELS="${NCHANNELS:-2}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --channels) NCHANNELS="$2"; shift 2 ;;
        *) shift ;;
    esac
done

# Source NCCL config from sft.sh-style detection
CX7_IF=""
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

export NCCL_DEBUG=WARN
export NCCL_IB_HCA="${CX7_HCA:-roceP2p1s0f1}"
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=13
export NCCL_IB_MERGE_NICS=0
export NCCL_CROSS_NIC=0
export NCCL_MAX_NCHANNELS="$NCHANNELS"
export NCCL_NET_GDR_LEVEL=0
export NCCL_NET_GDR_READ=0
export NCCL_TIMEOUT=1800
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8"
export TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas"
export TRITON_CACHE_DIR="/workspace/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="/workspace/.cache/torchinductor"
export TORCHINDUCTOR_FX_GRAPH_CACHE=1

if [ -n "$CX7_IF" ]; then
    export NCCL_SOCKET_IFNAME="$CX7_IF"
    export GLOO_SOCKET_IFNAME="$CX7_IF"
fi

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$CODE_DIR"

DATA_DIR="/workspace/baked/GeneT5/feb17_s51_v1_e1"
MODEL_PATH="/workspace/model/GeneT5/init/init_upcycle_moe16_topk2"

echo "========================================"
echo "  DDP Tuning Benchmark"
echo "  NCCL_MAX_NCHANNELS=$NCHANNELS"
echo "  Data: $DATA_DIR"
echo "  Model: $MODEL_PATH"
echo "========================================"

torchrun \
    --nnodes=2 \
    --nproc_per_node=1 \
    --node_rank="${NODE_RANK:-0}" \
    --master_addr="${MASTER_ADDR:-192.168.100.10}" \
    --master_port="${MASTER_PORT:-29501}" \
    tests/test_tmp_ddp_tuning.py \
    "$DATA_DIR" "$MODEL_PATH"
