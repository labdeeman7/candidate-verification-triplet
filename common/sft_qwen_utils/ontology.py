"""Structured valid-combination ontology derived from the official triplet registry.

Built once from `TripletSegmentationVariables.categories["triplet"]` (the same
100-triplet registry used for AP evaluation and by `triplet_validator.py`), so
this stays authoritative and in sync with evaluation.
"""

from functools import lru_cache

# Supports being imported either as `common.sft_qwen_utils.ontology` (repo root
# on sys.path) or as a bare sibling package — try both rather than assuming one
# invocation style.
try:
    from common.tripletseg_dataset_utils.dataset_variables import TripletSegmentationVariables
except ModuleNotFoundError:
    from tripletseg_dataset_utils.dataset_variables import TripletSegmentationVariables

from .converters import (
    labelme_instrument_to_sft,
    labelme_verb_to_sft,
    labelme_target_to_sft,
)


@lru_cache(maxsize=1)
def _load_registry() -> tuple[tuple[str, str, str], ...]:
    """Parse the official triplet registry into SFT-format (instrument, action, target) tuples."""

    triplets = []
    for labelme_triplet in TripletSegmentationVariables.categories["triplet"].values():
        inst_lm, verb_lm, target_lm = labelme_triplet.split(",")
        triplets.append(
            (
                labelme_instrument_to_sft(inst_lm),
                labelme_verb_to_sft(verb_lm),
                labelme_target_to_sft(target_lm),
            )
        )
    return tuple(triplets)


def valid_actions_for_instrument(instrument: str) -> list[str]:
    """Sorted SFT-format actions that are valid for a given SFT-format instrument."""

    actions = {verb for inst, verb, _ in _load_registry() if inst == instrument}
    return sorted(actions)


def valid_targets_for_instrument_action(instrument: str, action: str) -> list[str]:
    """Sorted SFT-format targets that are valid for a given (instrument, action) pair."""

    targets = {
        target
        for inst, verb, target in _load_registry()
        if inst == instrument and verb == action
    }
    return sorted(targets)


NOT_AN_INSTRUMENT = "not an instrument"


def options_for_turn(turn_key: str, instrument: str, action: str | None = None) -> list[str]:
    """Valid options for a Candidate Verification turn, given fields confirmed so far.

    Shared between dataset construction (using ground-truth confirmed fields
    under teacher forcing) and inference (using the model's own parsed
    earlier-turn output), so both stay consistent.

    NOT_AN_INSTRUMENT is a pseudo-class used by the Stage-1-noise-aligned training
    variant (V2): a Stage-1 detection with no matching GT instance at all (a real
    false positive). It has no entry in the triplet registry (it's not a real
    instrument), so the registry lookups below would silently return an empty
    options list for it -- special-cased here instead, so the shown options
    correctly read "no action"/"no target" rather than an unhelpful empty list.
    """

    if instrument == NOT_AN_INSTRUMENT:
        return ["no action"] if turn_key == "action" else ["no target"]
    if turn_key == "action":
        return valid_actions_for_instrument(instrument)
    if turn_key == "target":
        if action is None:
            raise ValueError("action must be known before computing valid targets")
        return valid_targets_for_instrument_action(instrument, action)
    raise ValueError(f"No options defined for turn key: {turn_key}")
