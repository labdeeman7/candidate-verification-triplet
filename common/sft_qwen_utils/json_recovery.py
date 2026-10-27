import json
import re
from typing import Any, Dict, List, Optional, Tuple


def _find_json_object_end(text: str, start: int) -> Optional[int]:
    """Return the end index for a JSON object starting at start."""
    if start < 0 or start >= len(text) or text[start] != "{":
        return None

    depth = 0
    in_string = False
    escaped = False

    for idx in range(start, len(text)):
        char = text[idx]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return idx + 1

    return None


def _last_prediction_id_before(text: str, end: int) -> Optional[str]:
    matches = list(re.finditer(r'"id"\s*:\s*"([^"]+)"', text[:end]))
    if not matches:
        return None
    return matches[-1].group(1)


def recover_reasoning_predictions_from_text(text: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Recover id/final_triplet pairs from malformed reasoning JSON.

    This intentionally ignores the free-form reasoning fields. Those fields are
    where LLM JSON usually breaks, while the final_triplet objects are often
    still valid enough for evaluation/LabelMe conversion.
    """
    recovered = []
    errors = []
    pos = 0

    while True:
        key_pos = text.find('"final_triplet"', pos)
        if key_pos < 0:
            break

        pred_id = _last_prediction_id_before(text, key_pos)
        object_start = text.find("{", key_pos)
        object_end = _find_json_object_end(text, object_start)

        if pred_id is None or object_start < 0 or object_end is None:
            errors.append(f"Could not recover final_triplet near character {key_pos}")
            pos = key_pos + len('"final_triplet"')
            continue

        fragment = text[object_start:object_end]
        try:
            final_triplet = json.loads(fragment)
        except json.JSONDecodeError as exc:
            errors.append(f"Could not parse final_triplet for {pred_id}: {exc}")
            pos = object_end
            continue

        required_keys = {"instrument", "action", "target"}
        if not required_keys.issubset(final_triplet):
            errors.append(f"Recovered final_triplet for {pred_id} is missing keys")
            pos = object_end
            continue

        recovered.append(
            {
                "id": pred_id,
                "reasoning": {"recovered_from_malformed_json": True},
                "final_triplet": final_triplet,
            }
        )
        pos = object_end

    return recovered, errors


def build_recovered_reasoning_payload(text: str) -> Optional[Dict[str, Any]]:
    recovered, errors = recover_reasoning_predictions_from_text(text)
    if not recovered:
        return None

    return {
        "reasoning_predictions": recovered,
        "parse_failed": True,
        "recovered_from_parse_failed": True,
        "recovery_errors": errors,
        "raw_output": text,
    }
