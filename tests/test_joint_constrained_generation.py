import json

import pytest

from rig.open_tasks.beliefs import validate_partition
from rig.open_tasks.runtime import Runtime, partition_label_vectors, partition_token_trie


class CharTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, ids, skip_special_tokens=True):
        return ''.join(chr(int(index)) for index in ids if int(index))


def test_constraint_preserves_every_six_answer_partition():
    vectors = partition_label_vectors()
    assert len(vectors) == len(set(vectors)) == 203
    trie, longest = partition_token_trie(CharTokenizer())
    assert longest == 13
    for labels in vectors:
        node = trie
        for token in CharTokenizer().encode(json.dumps(labels, separators=(',', ':'))):
            node = node[token]
        assert node[None]
        result = validate_partition(json.dumps(labels), ['a', 'b', 'c', 'd', 'e', 'f'])
        assert sorted(index for group in result['groups'] for index in group) == list(range(6))
    short = trie
    for token in CharTokenizer().encode('[0'):
        short = short[token]
    assert ord(']') not in short


def test_real_generate_cannot_stop_at_one_label():
    torch = pytest.importorskip('torch')
    from transformers import GPT2Config, GPT2LMHeadModel
    runtime = Runtime.__new__(Runtime)
    runtime.tokenizer = CharTokenizer()
    runtime.model = GPT2LMHeadModel(GPT2Config(vocab_size=128, n_positions=32, n_embd=8,
        n_layer=1, n_head=1, eos_token_id=0, pad_token_id=0)).eval()
    runtime.encode = lambda chat, budget: {'input_ids': torch.tensor([[1, 2]]), 'attention_mask': torch.ones((1, 2), dtype=torch.long)}
    result = json.loads(runtime.partition_answers([], 32))
    assert len(result) == 6
    assert tuple(result) in partition_label_vectors()


def test_old_groups_and_new_label_vectors_mean_the_same_thing():
    answers = ['a', 'b', 'c', 'd', 'e', 'f']
    assert validate_partition('[0,1,0,1,1,1]', answers) == validate_partition('[[0,2],[1,3,4,5]]', answers)
    with pytest.raises(ValueError, match='exactly 6'):
        validate_partition('[0]', answers)
