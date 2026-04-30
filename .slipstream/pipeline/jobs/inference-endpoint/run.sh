#!/bin/bash
# Inference endpoint server: parallelism math → engine install → launch + health-poll.
#
# /repo is the inference-endpoint repo root (mounted via shark.toml [job] repo_dir).
# All Python code lives at /repo/src/ — same layout as tails-mpt/number-generator.

set -euo pipefail

export PATH="$HOME/bin:$PATH"
MICROMAMBA="$HOME/bin/micromamba"
ENV_NAME="inference-endpoint"
PYTHON() { "$MICROMAMBA" run -n "$ENV_NAME" python "$@"; }
PIP_IN_ENV() { "$MICROMAMBA" run -n "$ENV_NAME" pip "$@"; }

# Schema [[input]] env vars come in via baby-shark; defaults here are fallbacks
# only for the case where someone runs this job directly outside slipstream.
ENGINE="${ENGINE:-sglang}"
PORT="${PORT:-8000}"

echo "============================================================"
echo "Inference endpoint"
echo "  Engine:        $ENGINE"
echo "  Target model:  ${TARGET_MODEL:-<unset>}"
echo "  Accelerators:  ${ACCELERATORS:-<unset>}"
echo "  Port:          $PORT"
echo "  Spec enabled:  ${SPEC_ENABLED:-false}"
echo "============================================================"

# 1. Compute TP/DP/EP and validate fit. Writes /tmp/parallelism.json.
echo "=== Computing parallelism ==="
PYTHONPATH=/repo/src PYTHON /repo/src/parallelism.py
echo

# 2. Install the requested engine. Boring approach: one pip install per launch,
# accepting the cold-start cost (model download dwarfs it anyway).
echo "=== Installing engine: $ENGINE ==="
case "$ENGINE" in
  sglang)
    PIP_IN_ENV install --quiet "sglang==0.5.6"
    ;;
  vllm)
    PIP_IN_ENV install --quiet vllm
    ;;
  *)
    echo "Error: ENGINE must be 'sglang' or 'vllm', got: $ENGINE" >&2
    exit 2
    ;;
esac
echo

# 3. Launch the server in the background so we can health-poll it before
# declaring the job up. wait at the end keeps the job alive while the server runs.
LOG_FILE="$HOME/server.log"
echo "=== Launching server (logs: $LOG_FILE) ==="
PYTHONPATH=/repo/src PYTHON /repo/src/launch.py > "$LOG_FILE" 2>&1 &
SERVER_PID=$!

# 4. Wait for /health. Both engines expose it on the same path.
echo "Waiting for server to be ready..."
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-900}"  # Big models take time to load.
ELAPSED=0
INTERVAL=5
while [ $ELAPSED -lt $HEALTH_TIMEOUT ]; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Error: server process died." >&2
        echo "--- last 60 lines of $LOG_FILE ---"
        tail -60 "$LOG_FILE"
        exit 1
    fi
    if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "Server ready after ${ELAPSED}s."
        break
    fi
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
done

if [ $ELAPSED -ge $HEALTH_TIMEOUT ]; then
    echo "Error: server didn't become healthy in ${HEALTH_TIMEOUT}s." >&2
    tail -60 "$LOG_FILE"
    kill "$SERVER_PID" 2>/dev/null || true
    exit 1
fi

# 5. Connection info.
EXTERNAL_IP=$(curl -sf https://ifconfig.me 2>/dev/null || echo "<external-ip>")
SERVED_NAME="${SERVED_MODEL_NAME:-$(basename "${TARGET_MODEL}")}"
echo
echo "============================================================"
echo "Endpoint live at http://${EXTERNAL_IP}:${PORT}"
echo
echo "  curl http://${EXTERNAL_IP}:${PORT}/health"
echo
echo "  curl -s http://${EXTERNAL_IP}:${PORT}/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\":\"${SERVED_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}],\"max_tokens\":50}'"
echo "============================================================"

# Stay alive until the job is cancelled.
wait "$SERVER_PID"
