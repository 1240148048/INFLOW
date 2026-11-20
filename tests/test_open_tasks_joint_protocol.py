import json
import math
from collections import Counter

import pytest

from rig.io import read_jsonl, write_jsonl
from rig.open_tasks.pipeline import prepare, records, select, retrieve
from rig.open_tasks.paper_baselines import paper_contexts
from test_open_tasks import FakeRuntime, make_fixture


def test_shared_initial_is_used_by_all_retrieval_methods_but_only_js_calls_judge(tmp_path):
    data, output, config = make_fixture(tmp_path)
    samples, partitions = [], []
    class Tracking(FakeRuntime):
        def sample(self, chat, count, seed, settings):
            samples.append((count, seed))
            return super().sample(chat, count, seed, settings)
        def partition_answers(self, chat, max_prompt_tokens):
            partitions.append(len(json.loads(chat[1]['content'])['answers']))
            return super().partition_answers(chat, max_prompt_tokens)
    prepare(config, data, output, runtime_factory=Tracking)
    assert Counter(count for count, _ in samples) == {1: 6, 5: 30}
    assert Counter(partitions) == {6: 6}
    initial = records(output / 'initial.jsonl')
    for row in initial.values():
        assert len(row['responses']) == 1
        assert row['certainty'] == pytest.approx(math.exp(-row['initial_predictive_entropy']))
    for name in ['full', 'neuron', 'mutual', 'js']:
        path = output / ('teacher.jsonl' if name == 'js' else f'teacher.{name}.jsonl')
        for row in records(path).values():
            assert row['initial_response'] == initial[row['uid']]['initial_response']
            assert len(row['retrieved_responses']) == 5
            if name == 'js':
                assert len(row['joint_responses']) == 6
                assert row['initial_distribution'] == {'class:0': 1.}
            else:
                assert row['joint_responses'] is None
                assert row['clustering'] is None
                assert row['initial_distribution'] is None
                assert 'js' not in row['scores']
            assert row['uses_shared_initial']
    select(config, output, ['full', 'redundancy', 'consistency', 'neuron', 'mutual', 'js'])
    for row in read_jsonl(output / 'subsets/js.jsonl'):
        assert len(row['joint_class_ids']) == 6
        assert 'responses' not in row


def test_consistency_runs_without_initial_cache_and_uses_own_coqa_history(tmp_path):
    data, output, config = make_fixture(tmp_path)
    source = [{'uid': f'coqa:{i}', 'task': 'coqa', 'split': 'train', 'question': f'Original Q{i}',
               'context': 'Passage', 'query': f'Q{i}', 'history_uids': [f'coqa:{j}' for j in range(i)],
               'conversation_id': 'c'} for i in range(3)]
    write_jsonl(data / 'train.jsonl', source)
    class Independent(FakeRuntime):
        def sample(self, chat, count, seed, settings):
            assert count == 5
            assert len(chat) == 2
            assert 'SHARED_POISON' not in str(chat)
            if 'Question: Q1' in chat[-1]['content']:
                assert 'PRIVATE' in chat[-1]['content']
            return ['PRIVATE'] * count, [[1, 2]] * count, [1]
    # A malformed shared cache must not even be read by Consistency.
    output.mkdir(parents=True)
    (output / 'initial.jsonl').write_text('SHARED_POISON is not JSON')
    prepare(config, data, output, runtime_factory=Independent, methods=['consistency'])
    select(config, output, ['consistency'])
    for row in read_jsonl(output / 'subsets/consistency.jsonl'):
        assert row['initial_response'] is None
        assert row['retrieved_responses'] is None
        assert row['independent_responses'] == ['PRIVATE'] * 5
        assert row['joint_responses'] is None
        assert row['joint_clusters'] is None
        assert not row['demo_ids'] and not row['uses_shared_initial']
        assert set(row['scores']) == {'consistency'}


def test_retrieval_uses_single_answer_certainty_and_all_qa_tasks():
    rows = [{'uid': 'a', 'task': 'hotpotqa', 'trajectory': [[1, 0]], 'certainty': .5},
            {'uid': 'b', 'task': 'squad', 'trajectory': [[1, 0]], 'certainty': .9},
            {'uid': 'c', 'task': 'coqa', 'trajectory': [[1, 0]], 'certainty': .1}]
    config = {'selection': {'context_count': 1}, 'representation': {}}
    assert retrieve(rows, config)['a'] == ['b']
    mutual = paper_contexts(rows, 1, 'mutual')
    assert mutual['a'][0] == ['b']
    assert mutual['a'][1] == pytest.approx(-.5 * math.log(1 - .9))


@pytest.mark.parametrize('method', ['full', 'redundancy', 'consistency', 'neuron', 'mutual'])
def test_baseline_can_run_when_semantic_judge_is_unavailable(tmp_path, method):
    data, output, config = make_fixture(tmp_path)
    class NoJudge(FakeRuntime):
        def partition_answers(self, *args, **kwargs):
            raise AssertionError(f'{method} must never call an LLM for clustering')
    prepare(config, data, output, runtime_factory=NoJudge, methods=[method])
    select(config, output, [method])
    assert list(read_jsonl(output / 'subsets' / f'{method}.jsonl'))
