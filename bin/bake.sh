#!/bin/bash
#
# One-click distributed data baking across two DGX Sparks
#
# Usage:
#   bin/bake.sh --worker <IP> --tokenizer ../model/base [bake_data args...]
#
# What it does:
#   1. SSHes to worker, launches bake_data --node_rank 1 inside the gt5 container
#   2. Runs bake_data --node_rank 0 locally
#   3. Waits for both, prints combined summary
#
# Species are sorted by genome size and round-robin assigned so both nodes
# get roughly equal work (largest genome -> node 0, second -> node 1, etc.)
#
# Prerequisites:
#   - Both machines running the gt5 Docker container (start-gt5-multi.sh)
#   - SSH key access from host -> worker (passwordless)
#   - Same mount layout on both (/workspace/raw, /workspace/baked, etc.)
#

set -e

WORKER_IP=""
WORKER_USER="cg666"
CONTAINER="gt5"
BAKE_ARGS=()

# Parse our args, pass the rest through to bake_data
while [[ $# -gt 0 ]]; do
    case $1 in
        --worker)
            WORKER_IP="$2"
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
        *)
            BAKE_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$WORKER_IP" ]]; then
    echo "Usage: $0 --worker <WORKER_IP> --tokenizer <path> [bake_data args...]"
    echo ""
    echo "Options:"
    echo "  --worker IP          Worker node IP (required)"
    echo "  --worker-user USER   SSH user on worker (default: cg666)"
    echo "  --container NAME     Docker container name (default: gt5)"
    echo ""
    echo "All other args are passed to bin/bake_data"
    echo ""
    echo "Example:"
    echo "  bin/bake.sh --worker 192.168.100.11 \\"
    echo "      --tokenizer ../model/base \\"
    echo "      --output_dir ../baked/w20k \\"
    echo "      --window_size 20000 \\"
    echo "      --species_parallel 3 \\"
    echo "      --canonical_only"
    exit 1
fi


echo "============================================================"
echo "        GeneT5 Distributed Data Baker (2 nodes)"
echo "============================================================"
echo "  Host (node 0):   $(hostname) (local)"
echo "  Worker (node 1):  ${WORKER_USER}@${WORKER_IP}"
echo "  Container:        ${CONTAINER}"
echo "  Bake args:        ${BAKE_ARGS[*]}"
echo "============================================================"
echo ""

# ── Ensure worker container is running ──
SSH_CMD="ssh -F /dev/null -o StrictHostKeyChecking=no -o IdentityFile=~/.ssh/id_ed25519_spark -o IdentitiesOnly=yes ${WORKER_USER}@${WORKER_IP}"

WORKER_RUNNING=$(${SSH_CMD} "docker inspect -f '{{.State.Running}}' ${CONTAINER} 2>/dev/null" || echo "false")
if [[ "$WORKER_RUNNING" != "true" ]]; then
    echo "[node 1] Container not running, starting in daemon mode..."
    ${SSH_CMD} "cd /home/cg666/Code/GeneT5 && bash start-worker.sh --daemon"
    echo "[node 1] Waiting for container setup (pip install)..."
    for i in $(seq 1 30); do
        if ${SSH_CMD} "docker exec ${CONTAINER} python -c 'import liger_kernel' 2>/dev/null"; then
            break
        fi
        sleep 2
    done
fi

# Build the bake_data command
BAKE_CMD="cd /workspace/GeneT5 && PYTHONPATH=/workspace/GeneT5 python bin/bake_data ${BAKE_ARGS[*]}"

# ── Launch worker (node 1) via SSH ──
echo "[node 1] Launching on ${WORKER_IP}..."
ssh -F /dev/null -o StrictHostKeyChecking=no -o IdentityFile=~/.ssh/id_ed25519_spark -o IdentitiesOnly=yes "${WORKER_USER}@${WORKER_IP}" \
    "docker exec ${CONTAINER} bash -c '${BAKE_CMD} --nnodes 2 --node_rank 1'" \
    > >(while IFS= read -r line; do echo "[node 1] $line"; done) \
    2>&1 &
WORKER_PID=$!

# Small delay so worker prints its header first
sleep 2

# ── Launch host (node 0) locally ──
echo "[node 0] Launching locally..."
PYTHONPATH=/workspace/GeneT5 python bin/bake_data ${BAKE_ARGS[*]} \
    --nnodes 2 --node_rank 0 \
    > >(while IFS= read -r line; do echo "[node 0] $line"; done) \
    2>&1 &
HOST_PID=$!

# ── Wait for both ──
echo ""
echo "Both nodes running. Waiting for completion..."
echo ""

HOST_EXIT=0
WORKER_EXIT=0

wait $HOST_PID   || HOST_EXIT=$?
wait $WORKER_PID || WORKER_EXIT=$?

echo ""
echo "============================================================"
echo "  Distributed Bake Complete"
echo "============================================================"
echo "  Node 0 (host):   exit=${HOST_EXIT}"
echo "  Node 1 (worker): exit=${WORKER_EXIT}"

# ── Combined summary ──
OUTPUT_DIR=""
for i in "${!BAKE_ARGS[@]}"; do
    if [[ "${BAKE_ARGS[$i]}" == "--output_dir" ]]; then
        OUTPUT_DIR="${BAKE_ARGS[$((i+1))]}"
        break
    fi
done

if [[ -n "$OUTPUT_DIR" && -d "$OUTPUT_DIR" ]]; then
    echo ""
    echo "  Combined output:"
    TRAIN_COUNT=$(find "$OUTPUT_DIR" -name "training.bin" | wc -l)
    VAL_COUNT=$(find "$OUTPUT_DIR" -name "validation.bin" | wc -l)
    TOTAL_SIZE=$(du -sh "$OUTPUT_DIR" 2>/dev/null | cut -f1)
    echo "    Training files:   ${TRAIN_COUNT}"
    echo "    Validation files: ${VAL_COUNT}"
    echo "    Total size:       ${TOTAL_SIZE}"
fi

echo "============================================================"

if [[ $HOST_EXIT -ne 0 || $WORKER_EXIT -ne 0 ]]; then
    exit 1
fi
