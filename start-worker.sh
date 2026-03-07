#!/bin/bash
#
# Start the gt5 worker container on Spark 2
#
# Usage:
#   bash start-worker.sh          # interactive shell (default)
#   bash start-worker.sh --daemon # detached, stays running for remote jobs

MODE="interactive"
if [[ "$1" == "--daemon" || "$1" == "-d" ]]; then
    MODE="daemon"
fi

CONTAINER="gt5"

# If container exists, check state
if docker inspect "$CONTAINER" > /dev/null 2>&1; then
    if [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER")" == "true" ]]; then
        if [[ "$MODE" == "daemon" ]]; then
            echo "Container $CONTAINER already running"
            exit 0
        fi
        echo "Container $CONTAINER running, attaching..."
        docker exec -it "$CONTAINER" bash
        exit 0
    fi
    # Stopped but exists — restart (packages survive)
    echo "Container $CONTAINER stopped, restarting..."
    docker start "$CONTAINER"
    if [[ "$MODE" == "daemon" ]]; then
        echo "Container $CONTAINER restarted (daemon)"
        exit 0
    fi
    docker exec -it "$CONTAINER" bash
    exit 0
fi

# Fresh container — install packages
echo "Creating new container $CONTAINER..."

RUN_FLAGS="--gpus all --name $CONTAINER --network=host --ipc=host --privileged --ulimit memlock=-1:-1"

if [[ "$MODE" == "daemon" ]]; then
    RUN_FLAGS="-d $RUN_FLAGS"
else
    RUN_FLAGS="-it $RUN_FLAGS"
fi

INIT_CMD="pip install -q -r /workspace/GeneT5/requirements.txt 2>/dev/null"
if [[ "$MODE" == "daemon" ]]; then
    INIT_CMD="$INIT_CMD; sleep infinity"
else
    INIT_CMD="$INIT_CMD; exec bash"
fi

docker run $RUN_FLAGS \
    -v /home/cg666/Code/GeneT5:/workspace/GeneT5 \
    -v /home/cg666/Data/genome/raw:/workspace/raw \
    -v /home/cg666/Data/genome/baked:/workspace/baked \
    -v /home/cg666/Data/logs:/workspace/logs \
    -v /home/cg666/Data/model:/workspace/model \
    -e NCCL_SOCKET_IFNAME=enp1s0f1np1 \
    -e GLOO_SOCKET_IFNAME=enp1s0f1np1 \
    -e NCCL_DEBUG=INFO \
    -e NCCL_IB_HCA=rocep1s0f1 \
    -e NCCL_IB_GID_INDEX=3 \
    -e NCCL_IB_TIMEOUT=22 \
    -e NCCL_IB_RETRY_CNT=7 \
    -e NCCL_IB_ROCE_VERSION_NUM=2 \
    -e NCCL_IB_ADDR_FAMILY=AF_INET \
    -e NCCL_IB_TC=106 \
    -e NCCL_IB_MERGE_NICS=1 \
    -e NCCL_NET_GDR_LEVEL=0 \
    -e NCCL_NET_GDR_READ=0 \
    -e PYTHONPATH=/workspace/GeneT5 \
    -w /workspace \
    nvcr.io/nvidia/pytorch:25.12-py3 \
    bash -c "$INIT_CMD"
