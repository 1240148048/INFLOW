from __future__ import annotations

import random
import re
import string
from collections import Counter


def normalize_answer(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def answer_keys(texts):
    return ["answer:" + normalize_answer(text) if normalize_answer(text) else None for text in texts]


def qa_metrics(text, references):
    predicted = normalize_answer(text)
    exact, f1 = 0.0, 0.0
    for reference in references:
        gold = normalize_answer(reference)
        exact = max(exact, float(predicted == gold))
        left, right = predicted.split(), gold.split()
        overlap = sum((Counter(left) & Counter(right)).values())
        value = float(left == right) if not left or not right else 2 * overlap / (len(left) + len(right))
        f1 = max(f1, value)
    return {"exact": exact, "f1": f1, "valid": int(bool(predicted))}


def question(context, query, history=()):
    result = "Passage:\n" + context + "\n"
    if history:
        result += "Conversation history (model-generated answers):\n"
        result += "\n".join(f"Q: {q}\nA: {a}" for q, a in history) + "\n"
    return result + "Question: " + query


def coqa_rows(dataset, split, count, seed):
    """Count QA turns, retain chronological prefixes of shuffled conversations."""
    order = list(range(len(dataset)))
    random.Random(seed).shuffle(order)
    result = []
    for index in order:
        conversation = dataset[index]
        previous = []
        for turn, query in enumerate(conversation["questions"]):
            uid = f"coqa:{split}:{index}:{turn}"
            row = {"uid": uid, "task": "coqa", "split": split, "question": question(conversation["story"], query), "context": conversation["story"], "query": query, "history_uids": list(previous), "conversation_id": f"coqa:{split}:{index}", "domain": conversation["source"]}
            if split == "test":
                row["references"] = [conversation["answers"]["input_text"][turn]]
            result.append(row)
            previous.append(uid)
            if count and len(result) == count:
                return result
    if count and len(result) < count:
        raise ValueError("Not enough CoQA turns")
    return result
