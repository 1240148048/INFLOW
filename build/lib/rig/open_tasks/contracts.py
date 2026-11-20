"""Generate input-only code probes with validation feedback and completion."""
from __future__ import annotations

import json
import re

from ..io import append_jsonl
from .runtime import contract_messages
from .semantics import validate_contract


def generate_contract(runtime, source, config, index, output):
    count = config["code"]["probe_count"]
    settings = dict(config["contract_generation"])
    settings["max_new_tokens"] = max(4096, settings.get("max_new_tokens", 2048))
    accepted = None
    seen = set()
    feedback = ""
    for attempt in range(6):
        remaining = count - len(accepted["probes"]) if accepted else count
        prompt = contract_messages(source, remaining)
        if accepted:
            prompt[0]["content"] += (
                "\nKeep exactly this entry_point and signature: "
                + json.dumps({k: accepted[k] for k in ("entry_point", "signature")})
                + "\nGenerate additional distinct inputs. Do not repeat these inputs: "
                + json.dumps(accepted["probes"], ensure_ascii=False)
            )
        if feedback:
            prompt[0]["content"] += "\nPrevious response validation: " + feedback
        texts, _, _ = runtime.sample(prompt, 1, config["seed"] + index * 10 + attempt, settings)
        raw = texts[0].strip()
        try:
            if raw.startswith("```"):
                raw = re.sub(r"^```[^\n]*\n|\n```$", "", raw).strip()
            value = json.loads(raw)
            # Validate every returned input before combining partial responses.
            probes = value["probes"]
            if not isinstance(probes, list) or not probes:
                raise ValueError("probes must be a nonempty list")
            candidate = validate_contract(json.dumps(value), len(probes))
            if accepted and any(candidate[k] != accepted[k] for k in ("entry_point", "signature")):
                raise ValueError("Keep the previously specified entry_point and signature")
            if accepted is None:
                accepted = {**candidate, "probes": []}
            for probe in candidate["probes"]:
                key = json.dumps(probe, sort_keys=True, allow_nan=False)
                if key not in seen:
                    accepted["probes"].append(probe)
                    seen.add(key)
            if len(accepted["probes"]) >= count:
                accepted["probes"] = accepted["probes"][:count]
                result = validate_contract(json.dumps(accepted), count)
                append_jsonl(output / "contract_attempts.jsonl", {"uid": source["uid"], "attempt": attempt + 1, "status": "complete", "response": texts[0]})
                return result
            feedback = f"Still need {count - len(accepted['probes'])} additional distinct inputs. Return complete JSON only."
        except (ValueError, KeyError, TypeError, AttributeError, SyntaxError) as exc:
            feedback = f"{type(exc).__name__}: {exc}. Return complete JSON only, without commentary or expected outputs."
        append_jsonl(output / "contract_attempts.jsonl", {"uid": source["uid"], "attempt": attempt + 1, "status": "retry", "feedback": feedback, "response": texts[0]})
        print(f"contract retry {attempt + 1}/6: {source['uid']}: {feedback}", flush=True)
    raise ValueError(f"Cannot complete {count} valid input probes for {source['uid']}; inspect {output / 'contract_attempts.jsonl'}. Existing preparation caches are preserved.")
