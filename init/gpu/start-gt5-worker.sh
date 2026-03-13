#!/bin/bash
#
# GeneT5 per-project container (worker node)
#
# Usage:
#   bash init/gpu/start-gt5-worker.sh          # interactive
#   bash init/gpu/start-gt5-worker.sh --daemon  # detached for remote jobs

CONTAINER="gt5"
IMAGE="nvcr.io/nvidia/pytorch:26.02-py3"

MODE="interactive"
[[ "$1" == "--daemon" || "$1" == "-d" ]] && MODE="daemon"

# Reuse existing container
if docker inspect "$CONTAINER" > /dev/null 2>&1; then
    if [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER")" == "true" ]]; then
        [[ "$MODE" == "daemon" ]] && echo "Already running" && exit 0
        docker exec -it "$CONTAINER" bash
        exit 0
    fi
    docker start "$CONTAINER"
    [[ "$MODE" == "daemon" ]] && echo "Restarted (daemon)" && exit 0
    docker exec -it "$CONTAINER" bash
    exit 0
fi

echo "Creating container $CONTAINER from $IMAGE..."

RUN_FLAGS="--gpus all --name $CONTAINER --hostname $(hostname) --network=host --ipc=host --privileged --ulimit memlock=-1:-1"
[[ "$MODE" == "daemon" ]] && RUN_FLAGS="-d $RUN_FLAGS" || RUN_FLAGS="-it $RUN_FLAGS"

INIT_CMD='
echo "[setup] Installing pip dependencies..."
pip install -q -r /workspace/code/requirements.txt 2>/dev/null
echo "[setup] Done."'

[[ "$MODE" == "daemon" ]] && INIT_CMD="$INIT_CMD; sleep infinity" || INIT_CMD="$INIT_CMD; exec bash"

docker run $RUN_FLAGS \
    -v /home/cg666/Code/GeneT5:/workspace/code \
    -v /home/cg666/Data/genome/raw:/workspace/data/raw \
    -v /home/cg666/Data/genome/baked/GeneT5:/workspace/data/baked \
    -v /home/cg666/Data/model/GeneT5:/workspace/data/model \
    -v /home/cg666/Data/logs:/workspace/data/logs \
    -e PYTHONPATH=/workspace/code \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8 \
    -e CUDA_MODULE_LOADING=LAZY \
    -w /workspace/code \
    "$IMAGE" \
    bash -c "$INIT_CMD"
