#!/bin/bash

CONTAINER="gt5"

# If container exists, check state
if docker inspect "$CONTAINER" > /dev/null 2>&1; then
    if [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER")" == "true" ]]; then
        echo "Container $CONTAINER running, attaching..."
        docker exec -it "$CONTAINER" bash
        exit 0
    fi
    # Stopped but exists — restart and attach (packages already installed)
    echo "Container $CONTAINER stopped, restarting..."
    docker start "$CONTAINER"
    docker exec -it "$CONTAINER" bash
    exit 0
fi

# Fresh container — install packages
echo "Creating new container $CONTAINER..."

[ ! -f ~/.claude.json ] && echo '{}' > ~/.claude.json

docker run --gpus all -it \
    --name $CONTAINER \
    --hostname $(hostname) \
    --network=host \
    --ipc=host \
    --privileged \
    --ulimit memlock=-1:-1 \
    -v /home/cg666/Code:/workspace/Code \
    -v /home/cg666/Data/genome/raw:/workspace/raw \
    -v /home/cg666/Data/genome/baked:/workspace/baked \
    -v /home/cg666/Data/logs:/workspace/logs \
    -v /home/cg666/Data/model:/workspace/model \
    -v ~/.gitconfig:/root/.gitconfig \
    -v ~/.ssh:/tmp/host-ssh:ro \
    -v ~/.claude:/root/.claude \
    -v ~/.claude.json:/root/.claude.json \
    -v ~/.nvm:/root/.nvm \
    -v ~/.local:/root/.local \
    -v ~/.npm:/root/.npm \
    -e ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}" \
    -e GITHUB_PAT="${GITHUB_PAT}" \
    -e NVM_DIR=/root/.nvm \
    -e PYTHONPATH=/workspace/GeneT5 \
    -e PATH="/root/.local/bin:/root/.nvm/versions/node/v24.13.0/bin:$PATH" \
    -w /workspace \
    nvcr.io/nvidia/pytorch:25.12-py3 \
    bash -c "
        cp -r /tmp/host-ssh /root/.ssh && chmod 700 /root/.ssh && chmod 600 /root/.ssh/* && chmod 644 /root/.ssh/*.pub /root/.ssh/known_hosts 2>/dev/null
        mkdir -p /home/cg666/.local && ln -sf /root/.local/share /home/cg666/.local/share
        apt-get update -qq && apt-get install -y -qq tmux > /dev/null 2>&1
        pip install -q -r /workspace/GeneT5/requirements.txt 2>/dev/null
        exec bash
    "
