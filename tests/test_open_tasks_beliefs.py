import json
import math

import pytest

from rig.open_tasks.beliefs import (belief_scores, joint_cluster, predictive_certainty,
                                    validate_partition, weighted_distribution)


def test_joint_six_answer_partition_and_entropy_weighted_beliefs():
    answers = ['New York City', 'NYC', 'Boston', 'New York City', 'Boston', 'Chicago']
    class Judge:
        calls = 0
        def partition_answers(self, prompt, max_prompt_tokens):
            self.calls += 1
            data = json.loads(prompt[1]['content'])
            assert [item['text'] for item in data['answers']] == answers
            assert 'not correctness' in prompt[0]['content']
            assert 'invalid label' in prompt[0]['content']
            return '[[0,1,3],[2,4],[5]]'
    judge = Judge()
    result = joint_cluster(judge, {'uid': 'q', 'task': 'squad', 'question': 'Where?'}, answers)
    assert judge.calls == 1
    assert result['keys'] == ['class:0', 'class:0', 'class:1', 'class:0', 'class:1', 'class:2']
    initial = weighted_distribution(result['keys'][:1], [2.])
    assert initial == {'class:0': 1.}
    retrieved = weighted_distribution(result['keys'][1:], [0, math.log(3), 0, math.log(3), 0])
    assert retrieved == pytest.approx({'class:0': 6/11, 'class:1': 2/11, 'class:2': 3/11})
    scores = belief_scores({'keys': result['keys'][:1], 'distribution': initial, 'certainty': math.exp(-2)},
                          {'keys': result['keys'][1:], 'distribution': retrieved})
    assert 0 < scores['js'] < math.log(2)
    assert predictive_certainty(2) == math.exp(-2)


@pytest.mark.parametrize('partition', ['[[0,1],[2,3,4]]', '[[0,1,1],[2,3,4,5]]', '[[true,1],[2,3,4,5]]'])
def test_incomplete_duplicated_or_inconsistent_partition_is_rejected(partition):
    with pytest.raises(ValueError):
        validate_partition(partition, ['Japan'] * 6)


def test_valid_short_answers_never_become_invalid_and_js_endpoints():
    same = validate_partition('[[0,1,2,3,4,5]]', ['Japan'] * 6)
    assert set(same['keys']) == {'class:0'}
    p = {'keys': ['class:0'], 'distribution': {'class:0': 1.}, 'certainty': .8}
    assert belief_scores(p, {'keys': ['class:0'] * 5, 'distribution': {'class:0': 1.}})['js'] == 0
    assert belief_scores(p, {'keys': ['class:1'] * 5, 'distribution': {'class:1': 1.}})['js'] == pytest.approx(math.log(2))


def test_malformed_partition_is_retried_jointly_instead_of_greedy_fallback():
    class Judge:
        calls = 0
        def partition_answers(self, prompt, max_prompt_tokens):
            self.calls += 1
            assert len(json.loads(prompt[1]['content'])['answers']) == 6
            return '[[0,1]]' if self.calls == 1 else '[[0,1,2,3,4,5]]'
    result = joint_cluster(Judge(), {'uid': 'q', 'task': 'coqa', 'question': 'Q'}, ['yes'] * 6)
    assert result['attempts'] == 2


def test_reported_identical_answer_split_is_repaired_without_retry():
    class Judge:
        calls = 0
        def partition_answers(self, prompt, max_prompt_tokens):
            self.calls += 1
            return '[[0,2,3,4,5],[1]]'
    judge = Judge()
    result = joint_cluster(judge, {'uid': 'coqa:train:2756:9', 'task': 'coqa', 'question': 'Q'}, ['Japan'] * 6)
    assert judge.calls == result['attempts'] == 1
    assert result['groups'] == [[0,1,2,3,4,5]]
    assert result['identical_answer_groups_merged']
    assert result['raw_groups'] == [[0,2,3,4,5],[1]]


def test_exact_answer_links_preserve_other_distinct_classes():
    answers = ['Paris', 'Boston', ' paris ', 'Boston', 'Rome', 'Rome']
    result = validate_partition('[[0],[1,3],[2],[4,5]]', answers)
    assert result['groups'] == [[0,2],[1,3],[4,5]]
    assert result['keys'] == ['class:0','class:1','class:0','class:1','class:2','class:2']


def test_quoted_answer_indices_have_exactly_the_same_partition():
    answers = ['Paris', 'Paris', 'Boston', 'Boston', 'Paris', 'Boston']
    quoted = validate_partition('[["0", "1", "4"], ["2", "3", "5"]]', answers)
    integers = validate_partition('[[0,1,4],[2,3,5]]', answers)
    assert quoted == integers


@pytest.mark.parametrize('text', ['[["0",1,2,3,4,"6"]]', '[["0",1,2,3,4,"1.5"]]', '[[true,1,2,3,4,5]]'])
def test_invalid_indices_remain_rejected_after_string_conversion(text):
    with pytest.raises(ValueError):
        validate_partition(text, ['Japan'] * 6)


@pytest.mark.parametrize('wrapped', [
    '```json\n[[0,1,2,3,4,5]]\n```',
    '```\n[[0,1,2,3,4,5]]\n```',
    '```JSON\r\n[["0","1","2","3","4","5"]]\r\n```',
    'Here is the partition:\n[[0,1,2,3,4,5]]',
    '[[0,1,2,3,4,5]]\nAll answers are equivalent.',
    '\ufeff[[0,1,2,3,4,5]]',
    '```json\n[[0,1,2,3,4,5]]',
    '```json\n[[0,1,2,3,4,5]]\n```\nAll answers are equivalent.',
])
def test_valid_json_wrappers_preserve_the_partition(wrapped):
    assert validate_partition(wrapped, ['Japan'] * 6) == validate_partition('[[0,1,2,3,4,5]]', ['Japan'] * 6)


def test_reported_code_fence_is_accepted_in_the_first_judge_call():
    class Judge:
        calls = 0
        def partition_answers(self, chat, max_prompt_tokens):
            self.calls += 1
            return '```json\n[[0,1,2,3,4,5]]\n```'
    judge = Judge()
    result = joint_cluster(judge, {'uid': 'coqa:train:1898:0', 'task': 'coqa', 'question': 'Q'}, ['yes'] * 6)
    assert judge.calls == result['attempts'] == 1


@pytest.mark.parametrize('broken', [
    '', 'no partition', '```json\n[[0,1,2,3,4,5]',
    '[[0,1,2,3,4,5]] [[0,1,2,3,4,5]]',
    '[[0,1,2,3,4,5]]]',
    '```json\n[[0,1,2,3,4]]\n```',
])
def test_wrapping_does_not_hide_missing_indices_or_ambiguous_output(broken):
    with pytest.raises(ValueError):
        validate_partition(broken, ['Japan'] * 6)
