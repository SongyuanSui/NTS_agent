#!/usr/bin/env bash
set -euo pipefail

# End-to-end inference smoke test for SKAB anomaly data.

cd /data1/zx57/NTS_agent

export PYTHONUNBUFFERED=1

LOG_FILE="logs/inference_skab.log"
mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Starting SKAB anomaly inference"
echo "Logging to $LOG_FILE"

/data1/zx57/.conda/envs/nts_agent/bin/python -u scripts/run_pipeline.py --mode inference -- \
  --pipeline-config configs/pipelines/end2end_anomaly.yaml \
  --task-config configs/tasks/anomaly_window.yaml \
  --config configs/data/anomaly.yaml \
  --max-samples-per-split 50 \
  --max-test-samples 5 \
  --save-json outputs/evaluations/inference_skab.json