#!/usr/bin/env bash
set -euo pipefail

# Stable end-to-end inference smoke test.
# Uses the multivariate pipeline config and stat retrieval override for ERing.

cd /data1/zx57/NTS_agent

/data1/zx57/.conda/envs/nts_agent/bin/python scripts/run_pipeline.py --mode inference -- \
  --pipeline-config configs/pipelines/end2end_multivariate.yaml \
  --config configs/data/ering.yaml \
  --max-samples-per-split 20 \
  --max-test-samples 1 \
  --retrieval-agent-override retrieval_agent_stat \
  --save-json outputs/evaluations/inference_smoke_ering_via_wrapper.json
