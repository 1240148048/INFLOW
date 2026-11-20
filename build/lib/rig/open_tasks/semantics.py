from __future__ import annotations

import ast
import math
import re
from collections import Counter
import unicodedata


def extract_code(text: str) -> str:
    """One fixed extraction rule, shared by training and evaluation."""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S | re.I)
    return (blocks[0] if blocks else text).strip() + "\n"


def parse_tree(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[^\n]*\n|\n```$", "", text).strip()
    tokens = re.findall(r"\[|\]|[^\s\[\]]+", text)
    position = 0

    def node():
        nonlocal position
        if position >= len(tokens) or tokens[position] != "[":
            raise ValueError("Expected '['")
        position += 1
        if position >= len(tokens) or not re.fullmatch(r"(?:IN|SL):[A-Z0-9_]+", tokens[position]):
            raise ValueError("Invalid intent/slot label")
        label = tokens[position]
        position += 1
        children = []
        while position < len(tokens) and tokens[position] != "]":
            if tokens[position] == "[":
                children.append(node())
            else:
                children.append(tokens[position])
                position += 1
        if position == len(tokens):
            raise ValueError("Unclosed tree")
        position += 1
        return label, tuple(children)

    tree = node()
    if position != len(tokens) or not tree[0].startswith("IN:"):
        raise ValueError("Expected exactly one root intent")
    return tree


def tree_metrics(prediction: str, reference: str) -> dict:
    gold = parse_tree(reference)
    try:
        pred = parse_tree(prediction)
    except ValueError:
        pred = None

    strict_exact = int(pred == gold)

    def normalize_leaf_spans(tree, *, labeled_only=False, inside_slot=False):
        if tree is None:
            return None
        label, children = tree
        inside_slot = inside_slot or label.startswith("SL:")
        normalized = []
        span = []
        def flush():
            if span and (not labeled_only or inside_slot):
                value = "".join(span)
                normalized.append(unicodedata.normalize("NFKC", value).casefold() if labeled_only else value)
            span.clear()
        for child in children:
            if isinstance(child, tuple):
                flush()
                normalized.append(normalize_leaf_spans(child, labeled_only=labeled_only, inside_slot=inside_slot))
            else:
                span.append(child)
        flush()
        return label, tuple(normalized)

    surface_exact = int(normalize_leaf_spans(pred) == normalize_leaf_spans(gold))
    pred = normalize_leaf_spans(pred, labeled_only=True)
    gold = normalize_leaf_spans(gold, labeled_only=True)

    def slots(tree, ancestors=()):
        result = Counter()
        if tree is None:
            return result
        label, children = tree
        path = ancestors + (label,)
        # Full subtree + ancestor labels preserves nesting and slot type.
        if label.startswith("SL:"):
            result[(path, tree)] += 1
        for child in children:
            if isinstance(child, tuple):
                result.update(slots(child, path))
        return result

    predicted, actual = slots(pred), slots(gold)
    return {
        "exact": int(pred == gold), "surface_exact": surface_exact,
        "strict_exact": strict_exact, "valid": int(pred is not None),
        "intent": int(pred is not None and pred[0] == gold[0]),
        "slot_tp": sum((predicted & actual).values()),
        "slot_pred": sum(predicted.values()), "slot_gold": sum(actual.values()),
    }


def entropy(probabilities) -> float:
    return -sum(p * math.log(p) for p in probabilities if p > 0)


def distribution(keys: list[str | None], tag: str) -> dict[str, float]:
    if not keys:
        raise ValueError("At least one response is required")
    # Failures do not constitute evidence that two programs/trees agree.
    values = [key if key is not None else f"invalid:{tag}:{i}" for i, key in enumerate(keys)]
    counts = Counter(values)
    return {key: count / len(values) for key, count in counts.items()}


def certainty(keys: list[str | None]) -> float:
    if len(keys) < 2:
        raise ValueError("Certainty needs at least two samples")
    valid_fraction = sum(key is not None for key in keys) / len(keys)
    concentrated = 1 - entropy(distribution(keys, "certainty").values()) / math.log(len(keys))
    return valid_fraction * max(0.0, min(1.0, concentrated))


def js_divergence(left: dict, right: dict) -> float:
    keys = set(left) | set(right)
    value = 0.0
    for key in keys:
        p, q = left.get(key, 0.0), right.get(key, 0.0)
        middle = (p + q) / 2
        if p:
            value += 0.5 * p * math.log(p / middle)
        if q:
            value += 0.5 * q * math.log(q / middle)
    return max(0.0, min(math.log(2), value))


def selection_scores(initial: list[str | None], contextual: list[str | None]) -> dict:
    p, q = distribution(initial, "initial"), distribution(contextual, "contextual")
    # Lexicographic tie handling is explicit and independent of correctness.
    mode = lambda d: min(d, key=lambda key: (-d[key], key))
    return {
        "js": js_divergence(p, q), "certainty": certainty(initial),
        "entropy": entropy(p.values()), "agreement": max(q.values()),
        "hard_flip": int(mode(p) != mode(q)),
        "initial_valid": sum(x is not None for x in initial),
        "contextual_valid": sum(x is not None for x in contextual),
    }


def topv2_keys(texts: list[str]) -> list[str | None]:
    result = []
    for text in texts:
        try:
            result.append("tree:" + repr(parse_tree(text)))
        except ValueError:
            result.append(None)
    return result


def validate_contract(text: str, count: int) -> dict:
    import json
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[^\n]*\n|\n```$", "", raw).strip()
    contract = json.loads(raw)
    signature = contract["signature"].strip()
    entry = contract["entry_point"]
    if not isinstance(entry, str) or not entry.isidentifier() or entry.startswith("_"):
        raise ValueError("Invalid entry point")
    if "\n" in signature or not signature.startswith("def ") or not signature.endswith(":"):
        raise ValueError("Expected a one-line function signature")
    tree = ast.parse(signature + "\n    pass\n")
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or tree.body[0].name != entry:
        raise ValueError("Signature and entry point disagree")
    probes = contract["probes"]
    if not isinstance(probes, list) or len(probes) != count:
        raise ValueError(f"Expected exactly {count} probes")
    for probe in probes:
        if set(probe) != {"args", "kwargs"} or not isinstance(probe["args"], list) or not isinstance(probe["kwargs"], dict):
            raise ValueError("Probes must contain JSON args and kwargs only")
    # Re-serialization rejects non-JSON objects and nonfinite floats.
    json.dumps(probes, allow_nan=False)
    return {"entry_point": entry, "signature": signature, "probes": probes}
