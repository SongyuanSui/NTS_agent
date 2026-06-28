#!/usr/bin/env bash
set -euo pipefail

# End-to-end inference smoke test for SKAB anomaly data.

cd /data1/zx57/NTS_agent

/data1/zx57/.conda/envs/nts_agent/bin/python scripts/run_pipeline.py --mode inference -- \
  --pipeline-config configs/pipelines/end2end_anomaly.yaml \
  --task-config configs/tasks/anomaly_window.yaml \
  --config configs/data/anomaly.yaml \
  --max-samples-per-split 5 \
  --max-test-samples 1 \
  --save-json outputs/evaluations/inference_smoke_skab_via_wrapper.json