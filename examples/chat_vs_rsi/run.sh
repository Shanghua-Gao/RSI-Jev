#!/usr/bin/env bash
# The whole comparison on one CUDA GPU, one model process at a time.
#
#   bash examples/chat_vs_rsi/run.sh [MODEL]        # MODEL: v4.0-vl-2b (default), a Hub id or a checkpoint dir
#
# Writes rsi.json, chat.json, results.json, results.md and chat-vs-rsi.gif next to this script.
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
MODEL=${1:-v4.0-vl-2b}
PORT=${PORT:-8000}
PY=${PYTHON:-python}

# (a) RSI-Jev v4.0-VL through `rsi-jev serve`
rsi-jev serve "$MODEL" --port "$PORT" > "$D/serve.log" 2>&1 &
SP=$!
trap 'kill $SP 2>/dev/null || true' EXIT
for _ in $(seq 1 120); do
  curl -sf "localhost:$PORT/health" >/dev/null && break
  kill -0 $SP 2>/dev/null || { echo "server exited; see $D/serve.log"; exit 1; }
  sleep 5
done
$PY "$D/rsi_client.py" "http://127.0.0.1:$PORT"
kill $SP; wait $SP 2>/dev/null || true
trap - EXIT

# (b) Qwen3.5-2B chat, non-thinking (the server is stopped first: one model at a time)
$PY "$D/chat_baseline.py"

$PY "$D/summarize.py"
$PY "$D/make_gif.py" "$D/chat-vs-rsi.gif"
