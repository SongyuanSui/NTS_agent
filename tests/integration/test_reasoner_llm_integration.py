
import sys

import pytest

pytestmark = pytest.mark.requires_llm


def test_reasoner_uses_llm():
    # Ensure repo sources are importable and rely on the real local LLM server
    sys.path.insert(0, "src")
    sys.path.insert(0, "scripts")

    import common
    import tasks.classification  # register task
    import agents.channel_decomposer  # register agent
    import agents.retrieval_agent_stat  # register agent
    import agents.reasoner_agent  # register agent
    import agents.aggregation_agent  # register agent
    import pipelines.inference_pipeline  # register pipeline

    from core.factories import build_agent, build_pipeline, build_task
    from pipelines.memory_build_pipeline import MemoryBuildPipeline
    from representations.raw_series import RawSeriesRepresentation
    from representations.statistics import StatisticsRepresentation, compute_statistics_for_sample
    from representations.text_summary import TextSummaryRepresentation
    from core.enums import TaskType

    repo = common.REPO_ROOT

    pipeline_cfg = dict(common.load_config_stack(repo / 'configs/pipelines/end2end_multivariate.yaml')['pipeline'])
    data_cfg = common.load_config_stack(repo / 'configs/data/ering.yaml')
    task_cfg = common.load_config_stack(repo / 'configs/tasks/classification.yaml')['task']
    reasoner_cfg = common.load_config_stack(repo / 'configs/agents/reasoner_agent.yaml')['agent']
    retrieval_cfg = common.load_config_stack(repo / 'configs/agents/retrieval_agent_stat.yaml')['agent']
    channel_decomposer_cfg = common.load_config_stack(repo / 'configs/agents/channel_decomposer.yaml')['agent']
    aggregation_cfg = common.load_config_stack(repo / 'configs/agents/aggregation_agent.yaml')['agent']

    # small dataset for speed
    bundle = common.load_dataset_from_config(data_cfg, max_samples_per_split=2)

    memory_pipeline = MemoryBuildPipeline(
        components={
            'ts_representation': RawSeriesRepresentation(),
            'summary_representation': TextSummaryRepresentation(config={'style': 'statistical'}),
            'statistic_representation': StatisticsRepresentation(),
        },
        config={'task_type': TaskType.CLASSIFICATION},
    )

    memory_bank = memory_pipeline.build_memory_bank(
        samples=bundle.train.samples,
        task_type=TaskType.CLASSIFICATION,
        channel_ids=[0],
        context={'representation_metadata': {'summary': {'style': 'statistical'}, 'statistic': {'feature_groups': None}}},
    ).memory_bank

    components = {
        'task': build_task(task_cfg),
        'channel_decomposer': build_agent(channel_decomposer_cfg),
        'retrieval_agent': build_agent(retrieval_cfg),
        'reasoner_agent': build_agent(reasoner_cfg),
        'aggregator_agent': build_agent(aggregation_cfg),
    }

    # Ensure the reasoner config explicitly enables LLM path
    try:
        components['reasoner_agent'].config['use_llm'] = True
    except Exception:
        pass

    pipeline = build_pipeline(pipeline_cfg, components=components)

    # Diagnostic prints
    print("pipeline llm config:", pipeline.get_config('llm'))
    try:
        print("reasoner use_llm:", components['reasoner_agent'].get_config('use_llm'))
    except Exception as e:
        print("reasoner config inspect failed:", e)

    sample = bundle.test.samples[0]
    query_stat_dict = compute_statistics_for_sample(sample, channel_id=0, feature_groups=None)

    # Build an LLM client from the pipeline config and inject into context
    from core.factories import build_llm_client
    llm_cfg = pipeline_cfg.get('params', {}).get('llm', {})
    llm_client = build_llm_client(llm_cfg)

    result = pipeline.run(sample, context={
        'memory_bank': memory_bank,
        'dataset_name': bundle.dataset_name,
        'selected_channel_ids': [0],
        'query_stat_dict': query_stat_dict,
        'llm_client': llm_client,
    })

    rd = result.intermediates['reasoner_output'].channel_decisions[0]

    assert rd.metadata is not None
    assert rd.metadata.get('llm_used') is True, f"Expected llm_used True, got {rd.metadata}"
    assert 'llm_raw_response' in rd.metadata and '"label"' in rd.metadata['llm_raw_response']
