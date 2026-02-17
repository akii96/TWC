#!/bin/bash
###############################################################################
# stress_test_sglang.sh
#
# Stress-tests an SGLang model serving inside Docker to catch hangs during
# CUDA graph build or decode.
#
# Usage:
#   ./stress_test_sglang.sh <docker_image> <num_loops>
#
# Example:
#   ./stress_test_sglang.sh lmsysorg/sglang:v0.5.8-rocm700-mi30x 20
###############################################################################

set -euo pipefail

# ── Args ────────────────────────────────────────────────────────────────────
if [ $# -lt 2 ]; then
    echo "Usage: $0 <docker_image> <num_loops>"
    echo "Example: $0 lmsysorg/sglang:v0.5.8-rocm700-mi30x 20"
    exit 1
fi

DOCKER_IMAGE="$1"
NUM_LOOPS="$2"

# ── HF Token ────────────────────────────────────────────────────────────────
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo ""
    echo "ERROR: HF_TOKEN environment variable is not set."
    echo ""
    echo "  Set it before running this script:"
    echo ""
    echo "    export HF_TOKEN='hf_your_token_here'"
    echo "    $0 <docker_image> <num_loops>"
    echo ""
    echo "  Or inline:"
    echo ""
    echo "    HF_TOKEN='hf_your_token_here' $0 <docker_image> <num_loops>"
    echo ""
    echo "  You can generate a token at: https://huggingface.co/settings/tokens"
    echo ""
    exit 1
fi

# ── Config ──────────────────────────────────────────────────────────────────
CONTAINER_TIMEOUT=900           # 15 minutes — hard safety net (most aiter modules prebuilt)
SERVER_PORT=30000
SERVER_STARTUP_TIMEOUT=600      # 10 minutes max to wait for server readiness
PROMPT_TIMEOUT=120              # 2 minutes max per prompt request
WORKSPACE_DIR="/home/${USER:-root}"

# Sanitise image name for folder naming (replace / and : with _)
IMAGE_SLUG=$(echo "$DOCKER_IMAGE" | sed 's/[\/:]/_/g')
RUN_DIR="$WORKSPACE_DIR/sglang_stress_${IMAGE_SLUG}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

SUMMARY_LOG="$RUN_DIR/summary.log"

# ── Counters ────────────────────────────────────────────────────────────────
SUCCESS_COUNT=0
FAIL_COUNT=0

# ── Prompt payload ──────────────────────────────────────────────────────────
PROMPT_PAYLOAD='{
  "model": "zai-org/GLM-4.7-FP8",
  "messages": [
    {
      "role": "user",
      "content": "only yes or no. is there winter in afirca?"
    }
  ],
  "stream": false,
  "top_p": 0.95,
  "temperature": 0.9,
  "repetition_penalty": 1.05,
  "max_tokens": 3000,
  "chat_template_kwargs": { "enable_thinking": false }
}'

# ── Server launch command (runs inside the container) ───────────────────────
SERVER_CMD="ROCM_QUICK_REDUCE_QUANTIZATION=INT4 \
HSA_NO_SCRATCH_RECLAIM=1 \
SGLANG_USE_AITER=1 \
SAFETENSORS_FAST_GPU=1 \
SGLANG_AITER_MLA_PERSIST=0 \
python3 -m sglang.launch_server \
  --model-path zai-org/GLM-4.7-FP8 \
  --tp-size 8 \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --mem-fraction-static 0.8 \
  --model-loader-extra-config '{\"enable_multithread_load\": true, \"num_threads\": 8}' \
  --port ${SERVER_PORT}"

# ── Helper: write to summary and stdout ─────────────────────────────────────
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$SUMMARY_LOG"
}

# ── Helper: kill container if still running ─────────────────────────────────
cleanup_container() {
    local name="$1"
    if docker ps -q -f name="$name" 2>/dev/null | grep -q .; then
        log "  Stopping container $name ..."
        docker rm -f "$name" >/dev/null 2>&1 || true
    fi
    # Also kill the watchdog if it's still around
    if [ -n "${WATCHDOG_PID:-}" ] && kill -0 "$WATCHDOG_PID" 2>/dev/null; then
        kill "$WATCHDOG_PID" 2>/dev/null || true
        wait "$WATCHDOG_PID" 2>/dev/null || true
    fi
}

###############################################################################
# MAIN LOOP
###############################################################################
log "============================================================"
log "SGLang Stress Test"
log "  Docker image : $DOCKER_IMAGE"
log "  Loops        : $NUM_LOOPS"
log "  Timeout      : ${CONTAINER_TIMEOUT}s ($(( CONTAINER_TIMEOUT / 60 )) min)"
log "  Results dir  : $RUN_DIR"
log "============================================================"

for (( i=1; i<=NUM_LOOPS; i++ )); do
    ITER_START=$(date +%s)
    ITER_TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
    CONTAINER_NAME="sglang_stress_iter${i}_$$"
    ITER_LOG="$RUN_DIR/iter_${i}_${ITER_TIMESTAMP}.log"
    ITER_STATUS="FAIL"  # assume fail until proven otherwise

    log ""
    log "────────────────────────────────────────────────────────────"
    log "  Iteration $i / $NUM_LOOPS   (container: $CONTAINER_NAME)"
    log "────────────────────────────────────────────────────────────"

    # Make sure no leftover container with same name
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

    # ── Launch Docker container in detached mode ────────────────────────────
    log "  Starting Docker container ..."
    docker run \
        -d \
        --cap-add=SYS_PTRACE \
        --cap-add=CAP_SYS_ADMIN \
        --security-opt seccomp=unconfined \
        --user root \
        --device=/dev/kfd \
        --device=/dev/dri \
        --group-add video \
        --ulimit memlock=999332768:999332768 \
        --ipc=host \
        --name "$CONTAINER_NAME" \
        --shm-size=128G \
        --hostname "STRESS-$(echo "${HOSTNAME:-$(hostname)}" | cut -f 1 -d .)" \
        -v "$WORKSPACE_DIR:/workspace/" \
        -v "$WORKSPACE_DIR/.cache:/workspace/.cache" \
        -v "$WORKSPACE_DIR/data:/workspace/data" \
        -e HF_HOME="/workspace/.cache/huggingface" \
        -e "HF_TOKEN=$HF_TOKEN" \
        --workdir /workspace/ \
        --network host \
        "$DOCKER_IMAGE" \
        bash -c "$SERVER_CMD" \
        2>&1 | tee -a "$ITER_LOG"

    # ── Watchdog: auto-kill container after CONTAINER_TIMEOUT ───────────────
    (
        sleep "$CONTAINER_TIMEOUT"
        if docker ps -q -f name="$CONTAINER_NAME" 2>/dev/null | grep -q .; then
            echo "[WATCHDOG] Timeout reached (${CONTAINER_TIMEOUT}s). Killing $CONTAINER_NAME" >> "$ITER_LOG"
            docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
        fi
    ) &
    WATCHDOG_PID=$!

    # ── Stream container logs to iteration log file ─────────────────────────
    docker logs -f "$CONTAINER_NAME" > "$ITER_LOG" 2>&1 &
    LOGS_PID=$!

    # ── Wait for server readiness ───────────────────────────────────────────
    log "  Waiting for server to become ready (up to ${SERVER_STARTUP_TIMEOUT}s) ..."
    SERVER_READY=false
    ELAPSED=0
    while [ $ELAPSED -lt $SERVER_STARTUP_TIMEOUT ]; do
        # Check if container is still running
        if ! docker ps -q -f name="$CONTAINER_NAME" 2>/dev/null | grep -q .; then
            log "  Container died before server became ready."
            break
        fi

        # Try the health endpoint
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
            --max-time 5 \
            "http://localhost:${SERVER_PORT}/health" 2>/dev/null) || true

        if [ "$HTTP_CODE" = "200" ]; then
            SERVER_READY=true
            log "  Server is ready! (took ~${ELAPSED}s)"
            break
        fi

        sleep 5
        ELAPSED=$(( ELAPSED + 5 ))
    done

    if [ "$SERVER_READY" = false ]; then
        log "  FAIL: Server did not become ready within ${SERVER_STARTUP_TIMEOUT}s or container died."
        cleanup_container "$CONTAINER_NAME"
        # Wait for log streaming to finish
        kill "$LOGS_PID" 2>/dev/null || true; wait "$LOGS_PID" 2>/dev/null || true
        FAIL_COUNT=$(( FAIL_COUNT + 1 ))
        ITER_END=$(date +%s)
        log "  Iteration $i finished in $(( ITER_END - ITER_START ))s — FAIL (server not ready)"
        continue
    fi

    # ── Send 10 prompts ─────────────────────────────────────────────────────
    PROMPTS_OK=true
    ALL_YES=true
    for p in 1 2 3 4 5 6 7 8 9 10; do
        log "  Sending prompt $p/10 ..."

        RESPONSE=$(curl -s --max-time "$PROMPT_TIMEOUT" \
            -X POST "http://localhost:${SERVER_PORT}/v1/chat/completions" \
            -H "accept: */*" \
            -H "Content-Type: application/json" \
            -d "$PROMPT_PAYLOAD" 2>/dev/null) || RESPONSE=""

        if [ -z "$RESPONSE" ]; then
            log "  FAIL: Prompt $p/10 — no response (timeout or connection error)."
            PROMPTS_OK=false
            break
        fi

        # Extract the content field from the response
        CONTENT=$(echo "$RESPONSE" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d['choices'][0]['message']['content'])
except Exception as e:
    print(f'PARSE_ERROR: {e}')
" 2>/dev/null) || CONTENT="PARSE_ERROR"

        log "  Prompt $p response: $CONTENT"

        # Save raw response to log
        echo "--- Prompt $p response ---" >> "$ITER_LOG"
        echo "$RESPONSE" >> "$ITER_LOG"
        echo "" >> "$ITER_LOG"

        # Check if the answer contains "yes" (case-insensitive)
        if ! echo "$CONTENT" | grep -qi "yes"; then
            ALL_YES=false
            log "  WARNING: Prompt $p/10 answer does not contain 'yes'."
        fi
    done

    # ── Check for HSA error in logs ─────────────────────────────────────────
    HSA_ERROR=false
    # Give logs a moment to flush
    sleep 2
    if grep -q "HSA_STATUS_ERROR_EXCEPTION" "$ITER_LOG" 2>/dev/null; then
        HSA_ERROR=true
        log "  FAIL: Found HSA_STATUS_ERROR_EXCEPTION in logs."
    fi

    # ── Determine success/failure ───────────────────────────────────────────
    if [ "$PROMPTS_OK" = true ] && [ "$ALL_YES" = true ] && [ "$HSA_ERROR" = false ]; then
        ITER_STATUS="SUCCESS"
        SUCCESS_COUNT=$(( SUCCESS_COUNT + 1 ))
    else
        ITER_STATUS="FAIL"
        FAIL_COUNT=$(( FAIL_COUNT + 1 ))
    fi

    # ── Tear down ───────────────────────────────────────────────────────────
    cleanup_container "$CONTAINER_NAME"
    kill "$LOGS_PID" 2>/dev/null || true; wait "$LOGS_PID" 2>/dev/null || true

    # Rename log file to include status
    FINAL_LOG="$RUN_DIR/iter_${i}_${ITER_TIMESTAMP}_${ITER_STATUS}.log"
    mv "$ITER_LOG" "$FINAL_LOG"

    ITER_END=$(date +%s)
    log "  Iteration $i finished in $(( ITER_END - ITER_START ))s — $ITER_STATUS"
done

###############################################################################
# SUMMARY
###############################################################################
log ""
log "============================================================"
log "  STRESS TEST COMPLETE"
log "============================================================"
log "  Docker image : $DOCKER_IMAGE"
log "  Total runs   : $NUM_LOOPS"
log "  Successes    : $SUCCESS_COUNT"
log "  Failures     : $FAIL_COUNT"
log "  Pass rate    : $(( SUCCESS_COUNT * 100 / NUM_LOOPS ))%"
log "  Results dir  : $RUN_DIR"
log "============================================================"

# Exit with non-zero if any failures
if [ "$FAIL_COUNT" -gt 0 ]; then
    exit 1
fi
exit 0
