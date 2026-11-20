from collections import Counter

import pytest

from rig.open_tasks.beliefs import joint_cluster, weighted_distribution
from rig.open_tasks.pipeline import prepare, experiment, select
from rig.io import write_jsonl, read_jsonl
from rig.open_tasks.data import dump
from rig.open_tasks.runtime import messages
from test_open_tasks import FakeRuntime, make_fixture


def test_js_selects_global_top_scores_without_task_quotas(tmp_path):
    data, output, config = make_fixture(tmp_path)
    config['selection']['fraction'] = .2
    rows, teacher = [], []
    for task, size in [('hotpotqa', 200), ('coqa', 150), ('squad', 150)]:
        for i in range(size):
            uid = f'{task}:{i:03}'
            rows.append({'uid': uid, 'task': task, 'split': 'train', 'question': uid})
            teacher.append({'uid': uid, 'scores': {'js': (.5 if task == 'hotpotqa' else 0) + i / 2000},
                            'demo_ids': [], 'self_feedback': None})
    for record in teacher:
        record.update(question=record['uid'], initial_response='A', retrieved_responses=['A'] * 5, initial_predictive_entropy=1.,
                      retrieved_predictive_entropies=[1.] * 5, initial_distribution={'class:0': 1.},
                      joint_responses=['A'] * 6, joint_class_ids=['class:0'] * 6,
                      joint_clusters=[[0, 1, 2, 3, 4, 5]], distribution={'class:0': 1.})
    write_jsonl(data / 'train.jsonl', rows)
    experiment(config, data, output)
    write_jsonl(output / 'initial.jsonl', rows)
    write_jsonl(output / 'teacher.jsonl', teacher)
    select(config, output, ['js'])
    chosen = list(read_jsonl(output / 'subsets/js.jsonl'))
    assert len(chosen) == 100
    assert Counter(row['task'] for row in chosen) == {'hotpotqa': 100}
    assert {row['uid'] for row in chosen} == {f'hotpotqa:{i:03}' for i in range(100, 200)}
    select(config, output, ['js'])
    assert list(read_jsonl(output / 'subsets/js.jsonl')) == chosen


def test_new_qa_prompt_does_not_change_common_evaluation_prompt():
    for task, phrase in [('squad', 'shortest passage span'), ('coqa', 'yes/no question')]:
        row = {'task': task, 'question': 'Passage and question'}
        original = messages(row, {})
        optimized = messages(row, {}, prompt_version='task_qa_v5')
        assert phrase in optimized[0]['content']
        assert phrase not in original[0]['content']
        assert messages(row, {}) == original
        assert "Do not output intermediate reasoning" in optimized[0]["content"]
        assert "Do not output intermediate reasoning" in original[0]["content"]
    row = {'task': 'code', 'question': 'Implement f'}
    assert messages(row, {}) == messages(row, {}, prompt_version='task_qa_v5')


def test_changed_prompt_cannot_reuse_old_generated_cache(tmp_path):
    data, output, config = make_fixture(tmp_path)
    prepare(config, data, output, runtime_factory=FakeRuntime, methods=['js'])
    with pytest.raises(ValueError, match='different data/config'):
        prepare({**config, 'preparation_prompt_version': 'task_qa_v5'}, data, output,
                runtime_factory=FakeRuntime, methods=['js'])
