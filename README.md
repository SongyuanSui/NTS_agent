# NTS Agent

New Time Series Agent

ts_agent/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
│
├── datasets/
│   ├── classification/<dataset_name>
│   ├── prediction/<dataset_name>
│   └── anomaly/<dataset_name>
│
├── configs/
│   ├── defaults.yaml
│   ├── data/
│   │   ├── classification.yaml
│   │   ├── anomaly.yaml
│   │   └── prediction.yaml
│   ├── tasks/
│   │   ├── classification.yaml
│   │   ├── prediction.yaml
│   │   └── anomaly_window.yaml
│   ├── agents/
│   │   ├── channel_selector.yaml
│   │   ├── channel_decomposer.yaml
│   │   ├── representation_agent_ts.yaml
│   │   ├── representation_agent_summary.yaml
│   │   ├── representation_agent_statistic.yaml
│   │   ├── retrieval_agent_ts.yaml
│   │   ├── retrieval_agent_text.yaml
│   │   ├── retrieval_agent_stat.yaml
│   │   ├── retrieval_agent_hybrid.yaml
│   │   ├── reasoner_agent.yaml
│   │   └── aggregation_agent.yaml
│   ├── retrieval/
│   │   ├── ts_dtw.yaml
│   │   ├── ts_euclidean.yaml
│   │   ├── text_embedding.yaml
│   │   ├── stat_distance.yaml
│   │   └── hybrid_weighted.yaml
│   ├── pipelines/
│   │   ├── memory_build.yaml
│   │   ├── end2end_multivariate.yaml
│   │   └── end2end_univariate.yaml
│   └── experiments/
│       ├── exp_classification.yaml
│       ├── exp_prediction.yaml
│       ├── exp_anomaly_window.yaml
│       └── ablations.yaml
│
├── scripts/
│   ├── run_channel_selector.py
│   ├── build_memory.py
│   ├── run_pipeline.py
│   ├── run_agent.py
│   ├── run_ablation.py
│   ├── evaluate_pipeline.py
│   └── evaluate_agent.py
│
├── src/
│   ├── __init__.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── enums.py
│   │   ├── types.py
│   │   ├── schemas.py
│   │   ├── interfaces.py
│   │   ├── exceptions.py
│   │   ├── registry.py
│   │   ├── factories.py
│   │   └── constants.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── schemas.py
│   │   ├── dataset_base.py
│   │   ├── dataset_registry.py
│   │   ├── split.py
│   │   ├── transforms.py
│   │   ├── windowing.py
│   │   ├── collate.py
│   │   ├── loaders/
│   │   │   ├── __init__.py
│   │   │   ├── classification_univariate_loader.py
│   │   │   ├── classification_multivariate_loader.py
│   │   │   ├── anomaly_loader.py
│   │   │   └── prediction_loader.py
│   │   └── adapters/
│   │       ├── __init__.py
│   │       ├── univariate_adapter.py
│   │       └── multivariate_adapter.py
│   │
│   ├── tasks/
│   │   ├── __init__.py
│   │   ├── task_base.py
│   │   ├── task_registry.py
│   │   ├── classification.py
│   │   ├── prediction.py
│   │   ├── anomaly_window.py
│   │   ├── prompt_targets.py
│   │   ├── output_parsers.py
│   │   └── label_space.py
│   │
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── schemas.py
│   │   ├── memory_bank.py
│   │   ├── memory_builder.py
│   │   ├── memory_store.py
│   │   ├── indexing.py
│   │   ├── filters.py
│   │   └── artifacts.py
│   │
│   ├── representations/
│   │   ├── stat_feature/
│   │   │   ├── calculation.py
│   │   │   └── selector.py 
│   │   ├── __init__.py
│   │   ├── rep_base.py
│   │   ├── schemas.py
│   │   ├── raw_series.py
│   │   ├── text_summary.py
│   │   ├── statistics.py
│   │   ├── bundle.py
│   │   └── normalizers.py
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── retriever_base.py
│   │   ├── schemas.py
│   │   ├── raw_retrievers.py
│   │   ├── text_retrievers.py
│   │   ├── stat_retrievers.py
│   │   ├── hybrid_retriever.py
│   │   ├── scoring.py
│   │   ├── fusion.py
│   │   └── postprocess.py
│   │
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── prompt_base.py
│   │   ├── builders.py
│   │   ├── formatters.py
│   │   └── templates/
│   │       ├── summary.py
│   │       ├── classification.py
│   │       ├── prediction.py
│   │       └── anomaly_window.py
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── client_base.py
│   │   ├── openai_client.py
│   │   ├── deepseek_client.py
│   │   ├── qwen_client.py
│   │   ├── response_parser.py
│   │   └── retry.py
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── agent_base.py
│   │   ├── schemas.py
│   │   ├── channel_selector.py
│   │   ├── channel_decomposer.py
│   │   ├── representation_agent_ts.py
│   │   ├── representation_agent_summary.py
│   │   ├── representation_agent_statistic.py
│   │   ├── retrieval_agent_ts.py
│   │   ├── retrieval_agent_text.py
│   │   ├── retrieval_agent_stat.py
│   │   ├── retrieval_agent_hybrid.py
│   │   ├── reasoner_agent.py
│   │   └── aggregation_agent.py
│   │
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── pipeline_base.py
│   │   ├── memory_build_pipeline.py
│   │   ├── inference_pipeline.py
│   │   ├── agent_graph.py
│   │   ├── execution_context.py
│   │   └── hooks.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics_base.py
│   │   ├── classification_metrics.py
│   │   ├── prediction_metrics.py
│   │   ├── anomaly_metrics.py
│   │   ├── retrieval_metrics.py
│   │   ├── agent_metrics.py
│   │   └── evaluators.py
│   │
│   ├── experiments/
│   │   ├── __init__.py
│   │   ├── experiment_base.py
│   │   ├── experiment_config.py
│   │   ├── single_agent.py
│   │   ├── end_to_end.py
│   │   └── ablation.py
│   │
│   ├── logging/
│   │   ├── __init__.py
│   │   ├── logger.py
│   │   ├── trackers.py
│   │   └── event_log.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── io.py
│       ├── seed.py
│       ├── time.py
│       ├── json_utils.py
│       ├── math_utils.py
│       └── validation.py
│
├── tests/
│   ├── unit/
│   │   ├── test_task_specs.py
│   │   ├── test_channel_selector.py
│   │   ├── test_channel_decomposer.py
│   │   ├── test_representation_agent_summary.py
│   │   ├── test_representation_agent_statistic.py
│   │   ├── test_retrieval_agent_ts.py
│   │   ├── test_retrieval_agent_text.py
│   │   ├── test_retrieval_agent_stat.py
│   │   ├── test_retrieval_agent_hybrid.py
│   │   ├── test_reasoner.py
│   │   └── test_aggregator.py
│   └── integration/
│       ├── test_pipeline_univariate.py
│       ├── test_pipeline_multivariate.py
│       ├── test_run_agent_script.py
│       └── test_run_ablation_script.py
│
└── outputs/
    ├── memory/
    │   └── <dataset_name>_<experiment_name>/
    │       ├── selected_channels.json
    │       ├── memory_bank.jsonl / parquet / pkl
    │       ├── index_ts.pkl
    │       ├── index_text.pkl
    │       └── index_stat.pkl
    ├── predictions/
    ├── agent_outputs/
    ├── evaluations/
    ├── experiments/
    └── logs/

## Running tests

Tests are organized under `tests/{unit,component,integration}`. The `test.sh` wrapper invokes `pytest` with `PYTHONPATH=src` already set. It uses the `python` on `PATH`, so activate the project's conda env first (e.g. `conda activate nts_agent`).

```bash
./test.sh                                                      # all tests
./test.sh unit                                                 # only unit
./test.sh component                                            # only component
./test.sh integration                                          # only integration
./test.sh tests/integration/test_reasoner_llm_integration.py   # a specific file
```

Note: the shortcut keywords (`unit`/`component`/`integration`/`all`) consume `$1` and ignore any extra arguments. To pass pytest flags like `-m`, pass a path instead (the script then forwards all arguments verbatim) — see the skip example below.

### Tests that need a local LLM server

Tests marked with `@pytest.mark.requires_llm` (e.g. `tests/integration/test_reasoner_llm_integration.py`) hit a local LLM endpoint at `http://localhost:30000/v1` and will fail if it is not running.

**To run them locally**, start sglang first (in a separate terminal — it's a long-running server):

```bash
./run_sglang_simple.sh
# wait until the server reports it is serving, then check:
curl -s http://localhost:30000/v1/models
```

Then run the tests as usual. `test.sh` does not pass `-m`, so `requires_llm` tests are included by default.

**To skip these tests locally** when sglang is not available:

```bash
./test.sh tests/integration -m "not requires_llm"
```

**CI**: the GitHub Actions workflow already runs `pytest tests/ -q -m "not requires_llm"`, so `requires_llm` tests are skipped automatically in CI ([.github/workflows/tests.yml](.github/workflows/tests.yml)).