from __future__ import annotations

import json
import os


def output_contract(row):
    return "short_answer_qa" if row["task"] in {"coqa", "squad", "hotpotqa"} else row["task"]


def partition_label_vectors(count=6):
    """One canonical label vector for every partition, with no semantic restriction."""
    def extend(prefix):
        if len(prefix) == count:
            yield prefix
            return
        for label in range(max(prefix) + 2):
            yield from extend((*prefix, label))
    return list(extend((0,)))


def partition_token_trie(tokenizer):
    trie, longest = {}, 0
    for labels in partition_label_vectors():
        tokens = tokenizer.encode(json.dumps(labels, separators=(",", ":")), add_special_tokens=False)
        longest = max(longest, len(tokens))
        node = trie
        for token in tokens:
            node = node.setdefault(token, {})
        node[None] = True
    return trie, longest


def messages(row: dict, schema: dict, demonstrations=(), *, self_feedback=None, prompt_version=None):
    def user_text(example):
        if example["task"] == "topv2" and example.get("domain"):
            return f"Domain: {example['domain']}\nUtterance: {example['question']}"
        return example["question"]

    if row["task"] == "topv2":
        instruction = (
            "Parse the user request into a TOPv2 bracketed intent/slot tree. "
            "Output one complete bracketed tree with no explanation. Use one root [IN:...] "
            "and wrap exact source spans in [SL:... span]. Copy each input word once, in order, "
            "including the words outside slots. Do not paraphrase or invent words. "
            "Use the narrowest relevant labels and preserve necessary nesting. "
            "Syntax pattern (placeholder labels only): [IN:INTENT text [SL:SLOT copied value ] text ]. "
            "Allowed labels: " + ", ".join(schema["labels"])
        )
    elif row["task"] == "squad" and prompt_version in {"task_qa_v4", "task_qa_v5"}:
        instruction = (
            "Answer the question using the passage. Return the shortest passage span that fully "
            "answers the question. Copy names, dates and numbers exactly from the passage. "
            "Output the answer only, without explanation or an introductory phrase."
        )
    elif row["task"] == "coqa" and prompt_version in {"task_qa_v4", "task_qa_v5"}:
        instruction = (
            "Answer the current question using the passage and conversation history. Return a short "
            "answer that directly resolves the question. For a yes/no question, return yes or no. "
            "Use unknown when the information is unavailable. Output the answer only."
        )
    elif row["task"] == "hotpotqa":
        instruction = (
            "Answer the question using the provided passages. Combine information across passages "
            "when necessary. Return only the shortest complete final answer. For a yes/no question, "
            "return yes or no."
        )
    elif row["task"] in {"coqa", "squad"}:
        instruction = "Read the passage and answer the current question. Use the conversation history when present. Return only a concise answer, without explanations. If the answer is unknown, return unknown."
    else:
        instruction = (
            "Implement the requested Python function. Return a complete Python module "
            "including the required function definition and any imports. "
            "Return code only, without explanations or Markdown fences."
        )
    if row["task"] in {"hotpotqa", "coqa", "squad"}:
        instruction += (
            " Output only the final answer itself. Do not output intermediate reasoning, analysis, "
            "explanations, evidence, an answer prefix, Markdown or any other text."
        )
    result = [{"role": "system", "content": instruction}]
    for demo in demonstrations:
        if output_contract(demo) != output_contract(row):
            raise ValueError("A demonstration must use the target task's output contract")
        result.extend([{"role": "user", "content": user_text(demo)}, {"role": "assistant", "content": demo["demonstration"]}])
    content = user_text(row)
    if self_feedback is not None:
        content += ("\n\nSelf-check your draft below. The draft was chosen by agreement among your "
                    "own independent answers. Verify it against the original request and correct any "
                    "errors. Return only the final answer in the task format.\nDraft:\n" + self_feedback)
    result.append({"role": "user", "content": content})
    return result


class Runtime:
    def __init__(self, config, adapter=None, device=None):
        import torch
        from ..modeling import load_causal_model, load_tokenizer, torch_dtype
        self.config = config
        self.device = torch.device(device or config["model"].get("device", "cuda:0"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("This experiment requires CUDA; CPU unit tests do not constitute a benchmark run")
        name = config["model"]["name"]
        self.tokenizer = load_tokenizer(name, config["model"].get("trust_remote_code", False))
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = load_causal_model(name, torch_dtype(config["model"]["dtype"]), config["model"].get("trust_remote_code", False)).to(self.device)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter).to(self.device)
        self.model.eval()
        self.projection = None

    def encode(self, chat, max_tokens):
        from ..chat_encoding import encode_messages
        encoded = encode_messages(self.tokenizer, chat, 2**30, self.device, enable_thinking=False)
        count = encoded["input_ids"].shape[1]
        if count > max_tokens:
            raise ValueError(f"Prompt has {count} tokens but budget is {max_tokens}; increase the budget or reduce context length. Prompts are not silently truncated.")
        return encoded

    def sample(self, chat, count, seed, settings):
        import torch
        from transformers import set_seed
        from .training import trim_tokens
        set_seed(seed)
        encoded = self.encode(chat, settings["max_prompt_tokens"])
        batch_size = int(os.environ.get("OPEN_SAMPLE_BATCH_SIZE", settings.get("sample_batch_size", count)))
        if batch_size < 1:
            raise ValueError("sample_batch_size must be positive")
        generated = []
        with torch.inference_mode():
            for offset in range(0, count, batch_size):
                output = self.model.generate(
                **encoded, do_sample=settings["temperature"] > 0,
                temperature=settings["temperature"] if settings["temperature"] > 0 else None,
                top_p=settings["top_p"] if settings["temperature"] > 0 else None,
                num_return_sequences=min(batch_size, count-offset), max_new_tokens=settings["max_new_tokens"],
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id, use_cache=True,
                )
                generated.extend(output)
        width = encoded["input_ids"].shape[1]
        tokens = [trim_tokens(row[width:].tolist(), self.tokenizer.eos_token_id, self.tokenizer.pad_token_id) for row in generated]
        texts = [self.tokenizer.decode(row, skip_special_tokens=True) for row in tokens]
        return texts, tokens, encoded["input_ids"][0].tolist()

    def choose_label(self, chat, labels, max_prompt_tokens):
        import torch
        from ..opsd_train import response_logits
        label_ids = [self.tokenizer.encode(label, add_special_tokens=False) for label in labels]
        if any(len(ids) != 1 for ids in label_ids) or len({ids[0] for ids in label_ids}) != len(labels):
            raise ValueError("Judge labels must be distinct single tokens for this tokenizer")
        with torch.inference_mode():
            logits = response_logits(self.model, self.encode(chat, max_prompt_tokens), 1)
            index = logits[0, -1, [ids[0] for ids in label_ids]].argmax().item()
        return labels[index]

    def partition_answers(self, chat, max_prompt_tokens):
        import torch
        encoded = self.encode(chat, max_prompt_tokens)
        if getattr(self, "_partition_trie", None) is None:
            self._partition_trie, self._partition_max_tokens = partition_token_trie(self.tokenizer)
        width = encoded["input_ids"].shape[1]
        eos = self.tokenizer.eos_token_id
        endings = eos if isinstance(eos, list) else [eos]
        def allowed_tokens(batch_id, input_ids):
            node = self._partition_trie
            for token in input_ids[width:].tolist():
                node = node[token]
            allowed = [token for token in node if token is not None]
            if None in node:
                allowed.extend(endings)
            return allowed
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded, do_sample=False, num_beams=1, num_return_sequences=1,
                max_new_tokens=self._partition_max_tokens + 1,
                prefix_allowed_tokens_fn=allowed_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id, use_cache=True)
        return self.tokenizer.decode(generated[0, width:], skip_special_tokens=True).strip()

    def predictive_entropies(self, prompt_ids, responses):
        """Mean full-vocabulary entropy at sampled answer prefixes, before sampling warpers."""
        import torch
        from ..opsd_train import response_logits

        result = []
        with torch.inference_mode():
            for response in responses:
                # The prefix at position t predicts response token t, including EOS if sampled.
                ids = torch.tensor([prompt_ids + response[:-1]], device=self.device)
                logits = response_logits(self.model, {"input_ids": ids, "attention_mask": torch.ones_like(ids)}, len(response))
                total = torch.zeros((), device=self.device)
                for chunk in logits[0].split(32):
                    log_probs = chunk.float().log_softmax(dim=-1)
                    total -= (log_probs.exp() * log_probs).sum()
                result.append(max(0.0, total.item() / len(response)))
                del logits
        return result

    def trajectory(self, prompt_ids, response):
        import torch
        import torch.nn.functional as F
        eos = self.tokenizer.eos_token_id
        excluded = set(eos if isinstance(eos, list) else [eos]) | {self.tokenizer.pad_token_id}
        content = [token for token in response if token not in excluded]
        ids = torch.tensor([prompt_ids + content], device=self.device)
        with torch.inference_mode():
            try:
                output = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True, use_cache=False, logits_to_keep=1)
            except TypeError:
                output = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True, use_cache=False)
            hidden = torch.stack([layer[0, -1].float() for layer in output.hidden_states[1:]])
            if self.projection is None:
                generator = torch.Generator(device="cpu").manual_seed(self.config["seed"])
                projection = torch.randint(0, 2, (hidden.shape[1], self.config["representation"]["projection_dim"]), generator=generator).float() * 2 - 1
                self.projection = projection.to(self.device)
            projected = F.normalize(hidden @ self.projection, dim=1)
        return projected.cpu().tolist()

    def neuron_features(self, prompt_ids, response):
        import torch
        from ..modeling import language_backbone, lm_head, mlp_down_projections
        from ..opsd_prepare import ResponseActivationCapture, neuron_set
        ids = torch.tensor([prompt_ids + response[:-1]], device=self.device)
        positions = list(range(len(prompt_ids) - 1, ids.shape[1]))
        settings = self.config.get("neuron", {})
        with torch.inference_mode(), ResponseActivationCapture(self.model, positions) as capture:
            language_backbone(self.model)(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            return neuron_set(capture.values, mlp_down_projections(self.model), lm_head(self.model),
                              response, settings.get("per_layer_topk", 256), settings.get("global_topk", 5000))


def contract_messages(row, count):
    return [{"role": "user", "content": (
        "Define one Python function interface for the following programming task and "
        f"exactly {count} diverse valid example inputs for that interface. "
        "Do not solve the task and do not provide expected outputs. Inputs must use JSON "
        "types only. Output exactly a JSON object with entry_point, signature (one line "
        "starting with def and ending with a colon), and probes (list of objects with "
        "args: list and kwargs: object). Use built-in input types; no custom classes.\nTask: " + row["question"]
    )}]
