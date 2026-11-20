from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
from transformers import GPT2Config, GPT2LMHeadModel

from rig.open_tasks.training import initialize_adapters, on_policy_loss


@pytest.mark.parametrize('task,selector', [('code', 'js'), ('squad', 'js'), ('squad', 'full')])
def test_fresh_student_rollouts_shared_prefix_and_frozen_ema_teacher(task, selector):
    class Runtime:
        def __init__(self):
            torch.manual_seed(4)
            self.model = GPT2LMHeadModel(GPT2Config(vocab_size=24, n_positions=32, n_embd=16, n_layer=1, n_head=2, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0))
            self.device = torch.device("cpu")
            self.calls = 0

        def sample(self, chat, count, seed, settings):
            assert len(chat) == 2  # system + target, no retrieved demonstrations
            if task == 'squad':
                assert 'shortest passage span' in chat[0]['content']
                assert 'Do not output intermediate reasoning' in chat[0]['content']
            assert self.model.active_adapter == "student"
            self.calls += 1
            return ["unused"], [[4 + self.calls, 6, 2]], [1, 2]

        def encode(self, chat, maximum):
            assert len(chat) == 4  # system + demonstration pair + target
            if task == 'squad':
                assert 'shortest passage span' in chat[0]['content']
            return {"input_ids": torch.tensor([[1, 9, 3]])}

    runtime = Runtime()
    config = {"seed": 42, "rollout": {}, "training": {"lora_rank": 2, "lora_alpha": 4, "lora_targets": ["c_attn"], "gradient_checkpointing": False, "max_sequence_tokens": 32, "kl_chunk_tokens": 2}}
    config['preparation_prompt_version'] = 'task_qa_v5'
    initialize_adapters(runtime, config)
    calls = []
    def capture(module, args, kwargs):
        calls.append(kwargs["input_ids"].detach().tolist()[0])
    handle = runtime.model.get_base_model().register_forward_pre_hook(capture, with_kwargs=True)
    row = {"task": task, "selector": selector, "question": "Implement f", "responses": ["must not be used as training prefix"]}
    demonstrations = [{"task": task, "question": "Other question", "demonstration": "Example answer"}]
    loss, count = on_policy_loss(runtime, row, demonstrations, {}, config, 42)
    loss.backward()
    assert count == 3
    assert calls[:2] == [[1, 9, 3, 5, 6], [1, 2, 5, 6]]
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for name, p in runtime.model.named_parameters() if ".student." in name)
    assert all(p.grad is None for name, p in runtime.model.named_parameters() if ".teacher." in name)
    on_policy_loss(runtime, row, demonstrations, {}, config, 43)
    assert runtime.calls == 2
    assert calls[-2:] == [[1, 9, 3, 6, 6], [1, 2, 6, 6]]
    handle.remove()


def test_training_checkpoint_restores_ema_optimizer_and_data_order(tmp_path):
    import json
    from rig.io import write_jsonl
    from rig.open_tasks.data import dump
    from rig.open_tasks.training import train

    class Tokenizer:
        def save_pretrained(self, directory):
            (directory / "tokenizer_config.json").write_text("{}")

    class Runtime:
        fail_second_update = False

        def __init__(self, config):
            torch.manual_seed(4)
            self.model = GPT2LMHeadModel(GPT2Config(vocab_size=24, n_positions=32, n_embd=16, n_layer=1, n_head=2, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0))
            self.device = torch.device("cpu")
            self.tokenizer = Tokenizer()

        def sample(self, chat, count, seed, settings):
            if self.fail_second_update and seed == 1_000_043:
                raise RuntimeError("simulated interruption")
            return ["unused"], [[5 + seed % 2, 6, 2]], [1, 2]

        def encode(self, chat, maximum):
            return {"input_ids": torch.tensor([[1, 9, 3]])}

    data = tmp_path / "data"
    rows = [{"uid": str(i), "task": "code", "split": "train", "question": f"Task {i}", "domain": "code"} for i in range(2)]
    write_jsonl(data / "train.jsonl", rows)
    dump(data / "schema.json", {})
    config = {"seed": 42, "rollout": {}, "training": {"lora_rank": 2, "lora_alpha": 4, "lora_targets": ["c_attn"], "gradient_checkpointing": False, "max_sequence_tokens": 32, "kl_chunk_tokens": 2, "updates": 2, "batch_size": 1, "learning_rate": 0.01, "warmup_ratio": 0, "ema_retention": 0.9, "checkpoint_every": 1}}
    def setup(directory):
        write_jsonl(directory / "initial.jsonl", [{**row, "demonstration": "def f(): return 1"} for row in rows])
        write_jsonl(directory / "subsets" / "js.jsonl", [{**rows[0], "selector": "js", "demo_ids": ["1"]}])
    uninterrupted, resumed = tmp_path / "full", tmp_path / "resumed"
    setup(uninterrupted)
    setup(resumed)
    train(config, data, uninterrupted, "js", Runtime)
    Runtime.fail_second_update = True
    with pytest.raises(RuntimeError, match="simulated"):
        train(config, data, resumed, "js", Runtime)
    Runtime.fail_second_update = False
    train(config, data, resumed, "js", Runtime)
    full_state = torch.load(uninterrupted / "checkpoints/js/training_state.pt", weights_only=False)
    resumed_state = torch.load(resumed / "checkpoints/js/training_state.pt", weights_only=False)
    assert full_state["completed"] == resumed_state["completed"] == 2
    assert full_state["tokens"] == resumed_state["tokens"] == 6
    for name in full_state["adapters"]:
        torch.testing.assert_close(full_state["adapters"][name], resumed_state["adapters"][name])
    metrics = json.loads((resumed / "checkpoints/js/training_metrics.json").read_text())
    assert metrics["updates"] == 2
    for update in (1, 2):
        checkpoint = resumed / "checkpoints/js/updates" / f"step-{update:06d}"
        assert (checkpoint / "student/adapter_model.safetensors").is_file()
        assert json.loads((checkpoint / "checkpoint.json").read_text())["optimizer_update"] == update


def test_actual_runtime_local_generation_and_layer_trajectory(tmp_path, monkeypatch):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    from rig.open_tasks.runtime import Runtime

    vocabulary = {word: i for i, word in enumerate(["[UNK]", "[PAD]", "[EOS]", "system", "user", "assistant", "return", "code", "def", "f", "x", "1", "2", "3", "4", "5"])}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")
    tokenizer.chat_template = "{% for message in messages %}{{message['role'] + ' ' + message['content'] + ' '}}{% endfor %}assistant "
    tokenizer.save_pretrained(tmp_path)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=len(vocabulary), n_positions=32, n_embd=16, n_layer=2, n_head=2, pad_token_id=1, eos_token_id=2))
    model.generation_config.suppress_tokens = [1, 2]
    model.save_pretrained(tmp_path)
    config = {"seed": 42, "model": {"name": str(tmp_path), "dtype": "float32", "device": "cpu"}, "representation": {"projection_dim": 3}}
    runtime = Runtime(config)
    settings = {"max_prompt_tokens": 20, "max_new_tokens": 3, "temperature": 0.6, "top_p": 0.95}
    texts, tokens, prompt = runtime.sample([{"role": "user", "content": "return code"}], 2, 42, settings)
    assert len(texts) == len(tokens) == 2
    assert len(tokens[0]) == 3
    entropies = runtime.predictive_entropies(prompt, tokens)
    assert len(entropies) == 2 and all(0 < h <= 2.773 for h in entropies)
    trajectory = torch.tensor(runtime.trajectory(prompt, tokens[0]))
    assert trajectory.shape == (2, 3)
    torch.testing.assert_close(trajectory.norm(dim=1), torch.ones(2))
    sequential = {**settings, "sample_batch_size": 1}
    texts, tokens, _ = runtime.sample([{"role": "user", "content": "return code"}], 2, 42, sequential)
    assert len(texts) == len(tokens) == 2
    calls = []
    original_generate = runtime.model.generate
    def tracked_generate(**kwargs):
        calls.append(kwargs["num_return_sequences"])
        return original_generate(**kwargs)
    monkeypatch.setattr(runtime.model, "generate", tracked_generate)
    monkeypatch.setenv("OPEN_SAMPLE_BATCH_SIZE", "5")
    texts, tokens, _ = runtime.sample([{"role": "user", "content": "return code"}], 5, 42, sequential)
    assert len(texts) == len(tokens) == 5
    assert calls == [5]
