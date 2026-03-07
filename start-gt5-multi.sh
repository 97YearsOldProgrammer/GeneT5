#!/bin/bash

# Multi-node DGX Spark launcher
# Usage:
#   Spark 1 (master): ./start-gt5-multi.sh
#   Spark 2 (worker): ./start-gt5-multi.sh

docker rm -f gt5 2>/dev/null

[ ! -f ~/.claude.json ] && echo '{}' > ~/.claude.json

docker run --gpus all -it \
    --name gt5 \
    --network=host \
    --ipc=host \
    --privileged \
    --ulimit memlock=-1:-1 \
    -v /home/cg666/Code/GeneT5:/workspace/GeneT5 \
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
    -e PATH="/root/.local/bin:/root/.nvm/versions/node/v24.13.0/bin:$PATH" \
    -w /workspace \
    nvcr.io/nvidia/pytorch:25.12-py3 \
    bash -c "cp -r /tmp/host-ssh /root/.ssh && chmod 700 /root/.ssh && chmod 600 /root/.ssh/* && chmod 644 /root/.ssh/*.pub /root/.ssh/known_hosts 2>/dev/null; pip install -q -r /workspace/GeneT5/requirements.txt 2>/dev/null; mkdir -p /home/cg666/.local && ln -sf /root/.local/share /home/cg666/.local/share; ssh -f -N -L 11001:localhost:11000 worker 2>/dev/null; exec bash"
