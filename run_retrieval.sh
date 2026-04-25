# PYTHONPATH=src python scripts/run_stat_feature_retrieval.py \
#   --dataset-loader ucr2015 \
#   --dataset ECG200 \
#   --channel-id 0 \
#   --feature-groups statistics.json spectral_frequency.json \
#   --distance cosine \
#   --normalize zscore \
#   --k 5 \
#   --persist-memory \
#   --experiment-name stat_spectral_retrieval

PYTHONPATH=src python scripts/run_stat_feature_retrieval.py \
  --dataset-loader uea \
  --dataset BasicMotions \
  --channel-id 1 \
  --max-samples-per-split 20 \
  --distance cosine \
  --normalize zscore \
  --k 5 \
  --persist-memory \
  --experiment-name stat_spectral_retrieval_UEA_simpleTest_channel1