PYTHONPATH=src python scripts/run_stat_feature_retrieval.py \
  --dataset ECG200 \
  --feature-groups statistics.json spectral_frequency.json \
  --distance cosine \
  --normalize zscore \
  --k 5 \
  --persist-memory \
  --experiment-name stat_spectral_retrieval