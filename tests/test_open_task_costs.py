import json

from rig.io import write_jsonl
from rig.open_tasks.costs import export
from rig.open_tasks.data import dump


def test_stage_totals_exempt_consistency_from_initial_and_mark_unmeasured_budget_scores(tmp_path):
    dump(tmp_path / 'resolved_config.json', {'model': {'name': 'test-model'}, 'training': {'updates': 250}})
    dump(tmp_path / 'experiment.json', {'training_rows': [{'uid': 'q'}]})
    dump(tmp_path / 'preparation_loading.json', {'seconds': 1})
    write_jsonl(tmp_path / 'initial.jsonl', [{'uid': 'q', 'timing': {
        'generation_seconds': 1, 'belief_seconds': 2, 'feature_seconds': 3}}])
    dump(tmp_path / 'selection_manifest.json', {'full': {'seconds': 8}, 'consistency': {'seconds': 8}})
    for method in ['full', 'consistency']:
        write_jsonl(tmp_path / f'teacher.{method}.jsonl', [{'uid': 'q', 'timing': {
            'generation_seconds': 4, 'belief_seconds': 5, 'feature_seconds': 0 if method == 'consistency' else 6}}])
        dump(tmp_path / f'retrieval.{method}.timing.json', {'seconds': 0 if method == 'consistency' else 7})
        target = tmp_path / 'checkpoints' / method
        dump(target / 'training_metrics.json', {'seconds': 20})
        dump(target / 'updates/step-000250/ready_time.json', {'startup_seconds': 5, 'distillation_seconds': 20})
        dump(target / 'updates/step-000050/ready_time.json', {'startup_seconds': 5, 'distillation_seconds': 10})
    rows = {row['method']: row for row in export(tmp_path, budget_minutes=.9)}
    assert rows['full']['adaptation_seconds'] == 62
    assert rows['consistency']['adaptation_seconds'] == 43
    assert rows['consistency']['initial_generation_seconds'] == 0
    assert rows['consistency']['initial_entropy_seconds'] == 0
    assert rows['js']['adaptation_seconds'] is None
    assert rows['js']['teacher_scoring_seconds'] is None
    import csv
    with (tmp_path / 'fixed_budget_summary.csv').open() as handle:
        budget = {row['method']: row for row in csv.DictReader(handle)}
    assert budget['full']['optimizer_update'] == '50'
    assert budget['full']['status'] == 'needs evaluation'
    assert budget['full']['overall_accuracy'] == ''
    assert budget['consistency']['optimizer_update'] == '250'
    assert (tmp_path / 'cost_breakdown.tex').read_text().startswith('\\begin{tabular}')
