"""Pure-logic helpers for eval_sft_multiturn_decomposition.py.

Deliberately kept free of any torch/PIL/transformers dependency (unlike the eval
script itself, which needs all three to actually load and run the model) so this
can be imported and tested on its own, e.g. from a bare environment with no ML
packages installed.

`safe_json_load` here is a deliberate small duplicate of the one in
eval_sft_with_overlays.py (same logic, same json_recovery.py dependency) rather
than an import from it, specifically to avoid pulling in that script's
module-level torch/PIL/transformers imports just to reach one JSON-parsing
function.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .json_recovery import build_recovered_reasoning_payload
from .ontology import options_for_turn


def safe_json_load(text: str) -> tuple[dict[str, Any], bool, str | None]:
    """Parse model JSON output, with a few recovery paths for common LLM mistakes."""

    raw_text = text.strip()
    raw_text = re.sub(r"```json", "", raw_text)
    raw_text = re.sub(r"```", "", raw_text).strip()

    try:
        return json.loads(raw_text), True, None
    except Exception as e:
        first_error = str(e)

    try:
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            return json.loads(match.group(0)), True, "recovered_from_extraction"
    except Exception:
        pass

    try:
        cleaned = re.sub(r",(\s*[}\]])", r"\1", raw_text)
        return json.loads(cleaned), True, "recovered_trailing_commas"
    except Exception:
        pass

    recovered_reasoning = build_recovered_reasoning_payload(raw_text)
    if recovered_reasoning is not None:
        return recovered_reasoning, True, "recovered_reasoning_final_triplets"

    return {"parse_failed": True, "raw_output": raw_text}, False, first_error


def extract_instance_ids_and_true_instruments(
    final_turn_gt_text: str,
) -> tuple[list[str], dict[str, str]]:
    """Recover instance ids and their given/fixed instrument from the final turn's GT.

    Instrument identity is supplied by Stage 1 and never revised by Stage 2 (per the
    paper's problem formulation), so options that depend on "which instrument" should
    always use this given value, not the model's own instrument-turn guess (candidate_
    verification's instrument turn is scaffolding for the dialogue, not something whose
    correctness should gate downstream option computation). In the predicted-grounding
    jsonl this slot holds the Stage-1 predicted class instead of a human label, so the
    same logic generalizes to both grounding settings unchanged.
    """

    gt_obj = json.loads(final_turn_gt_text)
    ids = [p["id"] for p in gt_obj["predictions"]]
    true_instrument_by_id = {p["id"]: p["instrument"] for p in gt_obj["predictions"]}
    return ids, true_instrument_by_id


def parse_field_by_id(raw_text: str, field: str) -> dict[str, str]:
    """Best-effort parse of a generated turn's response into {id: field_value}.

    Malformed or partial generations degrade gracefully to an empty mapping rather than
    raising, since a downstream turn's option lookup must still proceed for the ids it can
    resolve and simply omit options for the ones it can't.
    """

    parsed, ok, _ = safe_json_load(raw_text)
    if not ok or "predictions" not in parsed:
        return {}
    result = {}
    for p in parsed["predictions"]:
        if isinstance(p, dict) and "id" in p and field in p:
            result[p["id"]] = p[field]
    return result


def build_turn_user_text(
    turn,
    instrument_names: list[str],
    true_instrument_by_id: dict[str, str],
    generated_action_by_id: dict[str, str],
) -> str:
    """Render one turn's instruction, computing options from confirmed state, not GT.

    `true_instrument_by_id` is legitimate given information (see
    `extract_instance_ids_and_true_instruments`); `generated_action_by_id` comes from the
    model's own just-generated action-turn response, not the dataset's teacher-forced
    ground truth, so error propagation across turns is realistic at inference time.
    """

    options_text = ""
    if turn.show_options:
        lines = []
        for instance_id in instrument_names:
            instrument = true_instrument_by_id.get(instance_id)
            if instrument is None:
                continue
            if turn.key == "action":
                options = options_for_turn("action", instrument)
            elif turn.key == "target":
                action = generated_action_by_id.get(instance_id)
                if action is None:
                    continue
                options = options_for_turn("target", instrument, action)
            else:
                continue
            lines.append(f"{instance_id}: {options}")
        options_text = "\n".join(lines)
    return turn.user_template.format(instrument_list=instrument_names, options=options_text)
