#!/bin/bash
set -euxo pipefail

BUCKET="minimal-llm-fabimur"
RUN_NAME="aws-g5-run1"
IMAGE="public.ecr.aws/d5g0x1k7/minimal-llm-train:latest"
DATA_DIR="/opt/dlami/nvme/artifacts"

# torch.compile cache, namespaced by instance type since Inductor/Triton artifacts
# are tied to GPU arch + CUDA/driver version (g5=A10G, g6=L4 — not interchangeable).
INSTANCE_TYPE=$(curl -s http://169.254.169.254/latest/meta-data/instance-type)
CACHE_DIR="$DATA_DIR/torch-cache"
CACHE_S3_PREFIX="s3://$BUCKET/torch-cache/$INSTANCE_TYPE/"

mkdir -p "$DATA_DIR/checkpoints/$RUN_NAME"
mkdir -p "$CACHE_DIR"

docker pull "$IMAGE"

aws s3 cp "s3://$BUCKET/data/train.bin" "$DATA_DIR/train.bin"
aws s3 cp "s3://$BUCKET/data/val.bin" "$DATA_DIR/val.bin"
aws s3 cp "s3://$BUCKET/data/meta.json" "$DATA_DIR/meta.json"

aws s3 sync "s3://$BUCKET/checkpoints/$RUN_NAME/" "$DATA_DIR/checkpoints/$RUN_NAME/" || true
aws s3 sync "$CACHE_S3_PREFIX" "$CACHE_DIR/" || true

RESUME_ARGS=()
if [ -f "$DATA_DIR/checkpoints/$RUN_NAME/latest.pt" ]; then
  RESUME_ARGS=(--resume "artifacts/checkpoints/$RUN_NAME/latest.pt")
fi

( while true; do
    aws s3 sync "$DATA_DIR/checkpoints/" "s3://$BUCKET/checkpoints/" --exclude "*" --include "*.pt"
    aws s3 sync "$CACHE_DIR/" "$CACHE_S3_PREFIX"
    sleep 60
  done ) &
SYNC_PID=$!

docker run --gpus all -v "$DATA_DIR:/app/artifacts" \
  -e TORCHINDUCTOR_CACHE_DIR=/app/artifacts/torch-cache/inductor \
  -e TORCHINDUCTOR_FX_GRAPH_CACHE=1 \
  -e TRITON_CACHE_DIR=/app/artifacts/torch-cache/triton \
  "$IMAGE" \
  --run_name "$RUN_NAME" --save_interval 50 "${RESUME_ARGS[@]}"

kill "$SYNC_PID" || true
aws s3 sync "$DATA_DIR/checkpoints/" "s3://$BUCKET/checkpoints/" --exclude "*" --include "*.pt"
aws s3 sync "$CACHE_DIR/" "$CACHE_S3_PREFIX"
