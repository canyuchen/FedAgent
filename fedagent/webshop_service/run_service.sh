#!/bin/bash
# Launch the WebShop remote env service in the verl-agent-webshop conda env.
# Env vars: WEBSHOP_PORT (default 8080), WEBSHOP_POOL_SIZE (default 4).
set -e
source /software/miniconda3/4.10.3/etc/profile.d/conda.sh
conda activate verl-agent-webshop

HERE="$(cd "$(dirname "$0")" && pwd)"          # .../fedagent/fedagent/webshop_service
REPO_ROOT="$(cd "$HERE/../.." && pwd)"          # .../fedagent (repo root)
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"      # so `import fedagent.webshop_service.server` resolves
PORT="${WEBSHOP_PORT:-8080}"
export WEBSHOP_POOL_SIZE="${WEBSHOP_POOL_SIZE:-4}"

echo "[webshop-service] python=$(which python) port=$PORT pool=$WEBSHOP_POOL_SIZE root=$REPO_ROOT"
exec uvicorn fedagent.webshop_service.server:app --host 0.0.0.0 --port "$PORT" --log-level warning
