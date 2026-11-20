"""HotpotQA train-medium held-out sampling and official-style answer scoring."""
from __future__ import annotations

import json
from pathlib import Path

from .data import stratified
from .qa import normalize_answer, qa_metrics


def load_training_json(path):
    """Normalize the official JSON into the same row shape as the HF dataset."""
    with Path(path).open(encoding='utf-8') as handle:
        rows = json.load(handle)
    for row in rows:
        context = row['context']
        yield {**row, 'id': row.get('id', row.get('_id')),
               'context': context if isinstance(context, dict) else {
                   'title': [pair[0] for pair in context],
                   'sentences': [pair[1] for pair in context]}}


def medium_rows(dataset, train_count, test_count, seed):
    if train_count < 1 or test_count < 1:
        raise ValueError('HotpotQA medium requires positive train/test counts')
    raw, candidates, seen = {}, [], set()
    for row in dataset:
        if row['level'] != 'medium':
            continue
        text_key = ' '.join(row['question'].split()).casefold()
        if text_key in seen:
            continue
        seen.add(text_key)
        uid = row['id']
        raw[uid] = row
        candidates.append({'uid': uid, 'domain': row['type']})
    selected = stratified(candidates, train_count + test_count, seed)
    held_out = stratified(selected, test_count, seed + 1)
    test_ids = {row['uid'] for row in held_out}

    def convert(item, split):
        source = raw[item['uid']]
        context = source['context']
        paragraphs = ['[' + title + ']\n' + ' '.join(sentences)
                      for title, sentences in zip(context['title'], context['sentences'])]
        row = {'uid': f"hotpotqa:{split}:{item['uid']}", 'task': 'hotpotqa', 'split': split,
               'domain': item['domain'],
               'question': 'Passages:\n' + '\n\n'.join(paragraphs) + '\nQuestion: ' + source['question']}
        if split == 'test':
            row['references'] = [source['answer']]
        return row

    train = [convert(row, 'train') for row in selected if row['uid'] not in test_ids]
    evaluation = [convert(row, 'test') for row in held_out]
    return train, evaluation, len(candidates)


def answer_metrics(text, references):
    # HotpotQA gives zero F1 for mismatched yes/no/noanswer predictions.
    values = qa_metrics(text, references)
    predicted = normalize_answer(text)
    if predicted in {'yes', 'no', 'noanswer'} or any(normalize_answer(ref) in {'yes', 'no', 'noanswer'} for ref in references):
        values['f1'] = values['exact']
    return values
