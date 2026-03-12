#!/bin/bash
# Sync baked data + code + model to RunPod network volume via SSH
#
# Usage: bash sync_to_runpod.sh <RUNPOD_SSH> [--data] [--code] [--model] [--all]
#
# Examples:
#   bash sync_to_runpod.sh root@205.x.x.x:22222 --all
#   bash sync_to_runpod.sh root@205.x.x.x:22222 --code --data
#   bash sync_to_runpod.sh "ssh -p 22222 root@205.x.x.x" --data
#
# Run from inside gt5 container on DGX Spark master node

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <user@host:port OR 'ssh -p PORT user@host'> [--data] [--code] [--model] [--all]"
    exit 1
fi

RUNPOD_SSH="$1"
shift

# Parse SSH target
if [[ "$RUNPOD_SSH" == *" "* ]]; then
    SSH_CMD="$RUNPOD_SSH"
    SSH_HOST=$(echo "$RUNPOD_SSH" | awk '{print $NF}')
    SCP_PORT=$(echo "$RUNPOD_SSH" | grep -oP '(?<=-p\s)\d+' || echo "22")
else
    SSH_HOST=$(echo "$RUNPOD_SSH" | cut -d: -f1)
    SSH_PORT=$(echo "$RUNPOD_SSH" | cut -d: -f2 -s)
    SSH_PORT=${SSH_PORT:-22}
    SSH_CMD="ssh -p ${SSH_PORT} ${SSH_HOST}"
    SCP_PORT="${SSH_PORT}"
fi

# Paths — auto-detect host vs container paths
LOCAL_CODE="/home/cg666/Code/GeneT5"
REMOTE_BASE="/workspace"

# Baked data: try container path first, then host path
LOCAL_BAKED=""
for d in /workspace/baked/GeneT5/mar06_s51_w20k_p18k \
         /home/cg666/Data/genome/baked/GeneT5/mar06_s51_w20k_p18k; do
    [ -d "$d" ] && LOCAL_BAKED="$d" && break
done

# Model init: try container path, then Code/Data path, then host Data path
LOCAL_MODEL=""
for d in /workspace/model/GeneT5 \
         /home/cg666/Code/GeneT5/Data/model/GeneT5 \
         /home/cg666/Data/model/GeneT5; do
    [ -d "$d" ] && LOCAL_MODEL="$d" && break
done

SYNC_DATA=false
SYNC_CODE=false
SYNC_MODEL=false

for arg in "$@"; do
    case $arg in
        --data)  SYNC_DATA=true ;;
        --code)  SYNC_CODE=true ;;
        --model) SYNC_MODEL=true ;;
        --all)   SYNC_DATA=true; SYNC_CODE=true; SYNC_MODEL=true ;;
    esac
done

echo "=== Sync to RunPod ==="
echo "Target: ${SSH_HOST} (port ${SCP_PORT})"
echo "Sync data: ${SYNC_DATA}, code: ${SYNC_CODE}, model: ${SYNC_MODEL}"
echo "Local code:  ${LOCAL_CODE}"
echo "Local baked: ${LOCAL_BAKED:-NOT FOUND}"
echo "Local model: ${LOCAL_MODEL:-NOT FOUND}"

# Install rsync if missing (much better than scp for large transfers)
if ! command -v rsync &>/dev/null; then
    echo "Installing rsync..."
    apt-get update -qq && apt-get install -y -qq rsync
fi

# Check rsync on remote
${SSH_CMD} "command -v rsync" &>/dev/null || {
    echo "Installing rsync on remote..."
    ${SSH_CMD} "apt-get update -qq && apt-get install -y -qq rsync"
}

RSYNC_OPTS="-avz --progress --compress-level=1 -e 'ssh -p ${SCP_PORT}'"

# Sync code (small, fast)
if $SYNC_CODE; then
    echo ""
    echo "--- Syncing code (~5MB) ---"
    eval rsync ${RSYNC_OPTS} \
        --exclude='.git' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='.worker_pubkey' \
        --exclude='tests/' \
        --exclude='docs/' \
        --exclude='Data/' \
        --exclude='output/' \
        "${LOCAL_CODE}/" "${SSH_HOST}:${REMOTE_BASE}/GeneT5/"
    echo "Code sync complete"
fi

# Sync baked data (96GB - the big one)
if $SYNC_DATA; then
    if [ -z "$LOCAL_BAKED" ]; then
        echo "ERROR: Baked data not found at any known path"
        exit 1
    fi
    echo ""
    echo "--- Syncing baked data (~96GB) ---"
    echo "Source: ${LOCAL_BAKED}"
    echo "This will take a while. rsync supports resume if interrupted."
    ${SSH_CMD} "mkdir -p ${REMOTE_BASE}/baked/GeneT5/mar06_s51_w20k_p18k"

    # Transfer species directories in parallel (4 at a time)
    SPECIES_DIRS=$(ls -d ${LOCAL_BAKED}/*/ 2>/dev/null)
    TOTAL=$(echo "$SPECIES_DIRS" | wc -l)
    COUNT=0

    for species_dir in $SPECIES_DIRS; do
        species=$(basename "$species_dir")
        COUNT=$((COUNT + 1))
        echo "[${COUNT}/${TOTAL}] ${species}"
        eval rsync ${RSYNC_OPTS} \
            "${species_dir}" \
            "${SSH_HOST}:${REMOTE_BASE}/baked/GeneT5/mar06_s51_w20k_p18k/" &

        # Limit parallel transfers to 4
        if (( COUNT % 4 == 0 )); then
            wait
        fi
    done
    wait

    # Sync non-directory files (validation.bin, eval.json, etc.)
    eval rsync ${RSYNC_OPTS} \
        --exclude='*/' \
        "${LOCAL_BAKED}/" \
        "${SSH_HOST}:${REMOTE_BASE}/baked/GeneT5/mar06_s51_w20k_p18k/"

    echo "Data sync complete"
fi

# Sync model checkpoint
if $SYNC_MODEL; then
    if [ -z "$LOCAL_MODEL" ]; then
        echo "ERROR: Model directory not found at any known path"
        exit 1
    fi
    echo ""
    echo "--- Syncing model checkpoints ---"
    echo "Source: ${LOCAL_MODEL}"
    ${SSH_CMD} "mkdir -p ${REMOTE_BASE}/model/GeneT5"

    # Sync all init models (may have multiple variants)
    if [ -d "${LOCAL_MODEL}/init" ]; then
        for init_dir in ${LOCAL_MODEL}/init/*/; do
            [ -d "$init_dir" ] || continue
            INIT_NAME=$(basename "$init_dir")
            INIT_SIZE=$(du -sh "$init_dir" | cut -f1)
            echo "Syncing init model: ${INIT_NAME} (${INIT_SIZE})..."
            eval rsync ${RSYNC_OPTS} \
                "${init_dir}" \
                "${SSH_HOST}:${REMOTE_BASE}/model/GeneT5/init/${INIT_NAME}/"
        done
    fi

    # Sync latest experiment (most recent by date)
    LATEST=$(ls -dt ${LOCAL_MODEL}/mar*/ 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        LATEST_NAME=$(basename "$LATEST")
        echo "Syncing latest experiment: ${LATEST_NAME}"
        eval rsync ${RSYNC_OPTS} \
            "${LATEST}/" \
            "${SSH_HOST}:${REMOTE_BASE}/model/GeneT5/${LATEST_NAME}/"
    fi

    echo "Model sync complete"
fi

echo ""
echo "=== Sync complete ==="
echo "Remote layout:"
${SSH_CMD} "ls -la ${REMOTE_BASE}/GeneT5/ 2>/dev/null; echo '---'; du -sh ${REMOTE_BASE}/baked/ 2>/dev/null; du -sh ${REMOTE_BASE}/model/ 2>/dev/null" || true
