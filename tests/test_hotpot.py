import json
import sys
from types import SimpleNamespace

import pytest

from rig.io import read_jsonl, write_jsonl
from rig.open_tasks.data import dump, mix_data, read_pool
from rig.open_tasks.hotpot import medium_rows, answer_metrics, load_training_json
from rig.open_tasks.evaluation import score_task
from rig.open_tasks.runtime import messages


def raw_rows():
    return [{'id': str(i), 'level': 'medium' if i < 420 else 'hard',
             'question': f'Who directed film {i}?', 'answer': 'GOLD_SECRET',
             'type': 'bridge' if i % 3 else 'comparison',
             'context': {'title': ['Relevant', 'Distractor'], 'sentences': [['Some fact.'], ['Other fact.']]},
             'supporting_facts': {'title': ['Relevant'], 'sent_id': [0]}}
            for i in range(440)]


def test_medium_holdout_is_disjoint_deterministic_and_unlabeled():
    source = raw_rows()
    source.append({**source[0], 'id': 'duplicate'})
    train, evaluation, available = medium_rows(source, 200, 200, 42)
    assert len(train) == len(evaluation) == 200
    assert available == 420
    assert {row['uid'].split(':')[-1] for row in train}.isdisjoint(row['uid'].split(':')[-1] for row in evaluation)
    assert all(int(row['uid'].split(':')[-1]) < 420 for row in train + evaluation)
    assert 'GOLD_SECRET' not in json.dumps(train)
    assert all('supporting_facts' not in row for row in train + evaluation)
    assert all('[Distractor]\nOther fact.' in row['question'] for row in train + evaluation)
    assert medium_rows(source, 200, 200, 42) == (train, evaluation, available)
    with pytest.raises(ValueError, match='Requested'):
        medium_rows(source[:300], 200, 200, 42)


def test_official_json_keeps_sampling_identical_and_bypasses_hub(monkeypatch, tmp_path):
    from rig.open_tasks.__main__ import main
    source = raw_rows()
    official = []
    for row in source:
        converted = {**row, '_id': row['id'], 'context': list(zip(row['context']['title'], row['context']['sentences']))}
        del converted['id']
        official.append(converted)
    local = tmp_path / 'train.json'
    local.write_text(json.dumps(official), encoding='utf-8')
    assert medium_rows(load_training_json(local), 200, 200, 42) == medium_rows(source, 200, 200, 42)
    def forbidden(*args, **kwargs):
        raise AssertionError('Local JSON must not call the Hub')
    monkeypatch.setitem(sys.modules, 'datasets', SimpleNamespace(load_dataset=forbidden))
    monkeypatch.setenv('HOTPOT_TRAIN_JSON', str(local))
    dest = tmp_path / 'prepared'
    main(['data', '--task', 'hotpotqa', '--output', str(dest)])
    assert len(read_pool(dest)) == 200
    assert len(list(read_jsonl(dest / 'eval.jsonl'))) == 200


def test_hotpot_cli_preparation_cache_and_mixed_recipe(monkeypatch, tmp_path):
    from rig.open_tasks.__main__ import main
    calls = []
    def load(name, config, **kwargs):
        assert config == 'distractor' and kwargs['split'] == 'train'
        calls.append(name)
        return raw_rows()
    monkeypatch.setitem(sys.modules, 'datasets', SimpleNamespace(load_dataset=load))
    hotpot = tmp_path / 'hotpot'
    args = ['data', '--task', 'hotpotqa', '--output', str(hotpot)]
    main(args)
    main(args)
    assert len(calls) == 1
    assert len(read_pool(hotpot)) == 200
    manifest = json.loads((hotpot / 'data_manifest.json').read_text())
    assert manifest['source']['level'] == 'medium'
    assert 'custom disjoint holdout' in manifest['source']['evaluation_split']
    components = {'hotpotqa': hotpot}
    for task in ['coqa', 'squad']:
        dest = tmp_path / task
        rows = [{'uid': f'{task}:{i}', 'task': task, 'split': 'train', 'question': f'Q{i}'} for i in range(150)]
        write_jsonl(dest / 'train.jsonl', rows)
        write_jsonl(dest / 'eval.jsonl', [{**row, 'uid': 'test:' + row['uid'], 'split': 'test', 'references': ['A']} for row in rows])
        dump(dest / 'data_manifest.json', {'task': task})
        components[task] = dest
    mixed = tmp_path / 'mixed'
    mix_args = ['mix', '--output', str(mixed)]
    for task, dest in components.items():
        mix_args.extend(['--' + task, str(dest)])
    main(mix_args)
    assert len(read_pool(mixed)) == 500
    assert len(list(read_jsonl(mixed / 'eval.jsonl'))) == 500


def test_hotpot_scoring_and_answer_only_prompt():
    assert answer_metrics('The New York', ['New York'])['exact'] == 1
    assert answer_metrics('yes indeed', ['yes'])['f1'] == 0
    assert answer_metrics('New York', ['New York City'])['f1'] == pytest.approx(.8)
    rows = [{'uid': 'x', 'references': ['New York']}]
    predictions = [{'uid': 'x', 'sample_index': 0, 'text': 'New York'}, {'uid': 'x', 'sample_index': 1, 'text': 'Boston'}]
    score = score_task('hotpotqa', rows, predictions, 2, None)
    assert score['exact_match'] == score['token_f1'] == 50
    prompt = messages({'task': 'hotpotqa', 'question': 'Q'}, {}, prompt_version='task_qa_v5')
    assert 'Combine information across passages' in prompt[0]['content']
    assert 'Do not output intermediate reasoning' in prompt[0]['content']
