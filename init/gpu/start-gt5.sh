#!/bin/bash
#
# GeneT5 per-project container (master node)
#
# Usage: bash init/gpu/start-gt5.sh [--fresh]

CONTAINER="gt5"
IMAGE="nvcr.io/nvidia/pytorch:26.02-py3"

if [[ "$1" == "--fresh" ]]; then
    echo "Removing old container..."
    docker rm -f "$CONTAINER" 2>/dev/null
fi

# Reuse existing container
if docker inspect "$CONTAINER" > /dev/null 2>&1; then
    if [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER")" == "true" ]]; then
        echo "Container $CONTAINER running, attaching..."
        docker exec -it "$CONTAINER" bash
        exit 0
    fi
    echo "Container $CONTAINER stopped, restarting..."
    docker start "$CONTAINER"
    docker exec -it "$CONTAINER" bash
    exit 0
fi

echo "Creating container $CONTAINER from $IMAGE..."

docker run --gpus all -it \
    --name "$CONTAINER" \
    --hostname "$(hostname)" \
    --network=host \
    --ipc=host \
    --privileged \
    --ulimit memlock=-1:-1 \
    -v /home/cg666/Code/GeneT5:/workspace/code \
    -v /home/cg666/Data/genome/raw:/workspace/data/raw \
    -v /home/cg666/Data/genome/baked/GeneT5:/workspace/data/baked \
    -v /home/cg666/Data/model/GeneT5:/workspace/data/model \
    -v /home/cg666/Data/logs:/workspace/data/logs \
    -v /home/cg666/.ssh:/tmp/host-ssh:ro \
    -e PYTHONPATH=/workspace/code \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8 \
    -e CUDA_MODULE_LOADING=LAZY \
    -w /workspace/code \
    "$IMAGE" \
    bash -c '
        cp -r /tmp/host-ssh /root/.ssh && chmod 700 /root/.ssh && chmod 600 /root/.ssh/* && chmod 644 /root/.ssh/*.pub /root/.ssh/known_hosts 2>/dev/null
        echo "[setup] Installing pip dependencies..."
        pip install -q -r /workspace/code/requirements.txt 2>/dev/null
        echo "[setup] Done. Container ready."
        exec bash
    '
