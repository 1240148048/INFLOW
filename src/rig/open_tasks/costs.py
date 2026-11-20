"""Export measured adaptation-stage costs without loading a model."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

METHODS = ('original', 'full', 'redundancy', 'consistency', 'neuron', 'mutual', 'js')
STAGES = ('preparation_loading_seconds', 'initial_generation_seconds', 'initial_entropy_seconds',
          'features_and_retrieval_seconds', 'teacher_generation_seconds', 'teacher_scoring_seconds',
          'selection_seconds', 'training_startup_seconds', 'distillation_seconds')


def read(path, default=None):
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else default


def records(path):
    if not path.is_file():
        return []
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sum_stage(rows, key):
    if not rows or any(key not in row.get('timing', {}) for row in rows):
        return None
    return sum(row['timing'][key] for row in rows)


def write_csv(path, fields, rows):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export(output, budget_minutes=None):
    output = Path(output)
    config = read(output / 'resolved_config.json')
    if not config:
        raise ValueError('Missing resolved_config.json; point --output at the experiment root')
    expected = len(read(output / 'experiment.json', {}).get('training_rows', [])) or 500
    initial = records(output / 'initial.jsonl')
    selections = read(output / 'selection_manifest.json', {})
    loading = read(output / 'preparation_loading.json', {}).get('seconds')
    cap = config['training']['updates']
    result, budget_rows = [], []
    for method in METHODS:
        evaluation = read(output / 'metrics' / f'{method}.json', {})
        if method == 'original':
            row = {'method': method, 'status': 'evaluated' if evaluation else 'not evaluated',
                   'teacher_questions': 0, **dict.fromkeys(STAGES, 0.), 'adaptation_seconds': 0.,
                   'evaluation_seconds': evaluation.get('generation_seconds')}
            result.append(row)
            if budget_minutes is not None:
                budget_rows.append({'method': method, 'budget_minutes': budget_minutes, 'optimizer_update': 0,
                    'adaptation_minutes': 0., 'overall_accuracy': evaluation.get('overall_accuracy'), 'status': row['status']})
            continue
        family = 'full' if method == 'redundancy' else method
        teacher = records(output / ('teacher.jsonl' if family == 'js' else f'teacher.{family}.jsonl'))
        source = [] if method == 'consistency' else initial
        retrieval = read(output / f'retrieval.{family}.timing.json', {}).get('seconds')
        feature_source = 0. if method == 'consistency' else sum_stage(source, 'feature_seconds')
        feature_teacher = sum_stage(teacher, 'feature_seconds')
        features = (retrieval + feature_source + feature_teacher
                    if all(v is not None for v in (retrieval, feature_source, feature_teacher)) else None)
        target = output / 'checkpoints' / method
        training = read(target / 'training_metrics.json', {})
        ready = read(target / 'updates' / f'step-{cap:06d}' / 'ready_time.json', {})
        startup = ready.get('startup_seconds')
        if startup is None and training:
            startup = training.get('model_loading_seconds', 0.) + training.get('adapter_setup_seconds', 0.)
        row = {'method': method, 'teacher_questions': len(teacher),
            'preparation_loading_seconds': loading,
            'initial_generation_seconds': 0. if method == 'consistency' else sum_stage(source, 'generation_seconds'),
            'initial_entropy_seconds': 0. if method == 'consistency' else sum_stage(source, 'belief_seconds'),
            'features_and_retrieval_seconds': features,
            'teacher_generation_seconds': sum_stage(teacher, 'generation_seconds'),
            'teacher_scoring_seconds': sum_stage(teacher, 'belief_seconds'),
            'selection_seconds': selections.get(method, {}).get('seconds'),
            'training_startup_seconds': startup,
            'distillation_seconds': ready.get('distillation_seconds', training.get('seconds')),
            'evaluation_seconds': evaluation.get('generation_seconds')}
        prep_fields = STAGES[:-2]
        prep_complete = len(teacher) == expected and (method == 'consistency' or len(initial) == expected)
        prep_complete = prep_complete and all(row[key] is not None for key in prep_fields)
        complete = prep_complete and bool(training) and all(row[key] is not None for key in STAGES)
        row['status'] = 'complete' if complete else f'partial: teacher {len(teacher)}/{expected}'
        row['adaptation_seconds'] = sum(row[key] for key in STAGES) if complete else None
        result.append(row)
        if budget_minutes is not None:
            eligible = []
            if prep_complete:
                prep_seconds = sum(row[key] for key in prep_fields)
                for path in (target / 'updates').glob('step-*/ready_time.json'):
                    step = int(path.parent.name.split('-')[-1])
                    clock = read(path)
                    elapsed = prep_seconds + clock['startup_seconds'] + clock['distillation_seconds']
                    if step <= cap and elapsed <= budget_minutes * 60:
                        eligible.append((step, elapsed))
            step, elapsed = max(eligible, default=(0, 0.))
            metrics_path = output / ('metrics' if step == cap else f'curve/step-{step:06d}/metrics') / f'{method}.json'
            metrics = read(metrics_path, {}) if step else {}
            status = ('evaluated' if metrics else 'needs evaluation') if step else ('no checkpoint within budget' if prep_complete else 'preparation incomplete')
            budget_rows.append({'method': method, 'budget_minutes': budget_minutes, 'optimizer_update': step,
                'adaptation_minutes': elapsed / 60 if step else None, 'overall_accuracy': metrics.get('overall_accuracy'), 'status': status})
    fields = ['method', 'status', 'teacher_questions', *STAGES, 'adaptation_seconds', 'evaluation_seconds']
    write_csv(output / 'cost_breakdown.csv', fields, result)
    tex_fields = ['initial_generation_seconds', 'initial_entropy_seconds', 'features_and_retrieval_seconds',
                  'teacher_generation_seconds', 'teacher_scoring_seconds', 'selection_seconds',
                  'distillation_seconds', 'adaptation_seconds']
    tex = [r'\begin{tabular}{lrrrrrrrr}', r'\toprule',
           r'Method & Init. gen. & Init. entropy & Feat./retr. & Teacher gen. & Scoring & Select & Distill & Total \\', r'\midrule']
    for row in result:
        values = ['--' if row[key] is None else f'{row[key]/60:.2f}' for key in tex_fields]
        tex.append(row['method'] + ' & ' + ' & '.join(values) + r' \\')
    tex.extend([r'\bottomrule', r'\end{tabular}',
                '% Minutes. Total includes preparation loading and training startup, listed separately in CSV.'])
    (output / 'cost_breakdown.tex').write_text('\n'.join(tex)+'\n', encoding='utf-8')
    note = {'model': config['model']['name'], 'update_cap': cap, 'teacher_scoring':
        'Predictive entropy and answer scoring; joint LLM semantic clustering only for JS.',
        'accounting': 'Standalone adaptation reconstructed from recorded successful-stage timings. Shared initial preparation is charged to each method requiring it; Consistency is exempt. Includes recorded model loading/startup; excludes data download, evaluation, failed attempts and unrecorded restart/orchestration overhead. Partial totals remain blank.',
        'unit': 'CSV seconds; TeX minutes', 'budget_minutes': budget_minutes}
    (output / 'cost_breakdown_notes.json').write_text(json.dumps(note, indent=2)+'\n', encoding='utf-8')
    if budget_minutes is not None:
        write_csv(output / 'fixed_budget_summary.csv', ['method','budget_minutes','optimizer_update','adaptation_minutes','overall_accuracy','status'], budget_rows)
    print(f'Cost files saved to {output}', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--budget-minutes', type=float)
    args = parser.parse_args()
    if args.budget_minutes is not None and args.budget_minutes <= 0:
        parser.error('--budget-minutes must be positive')
    export(args.output, args.budget_minutes)

if __name__ == '__main__':
    main()
