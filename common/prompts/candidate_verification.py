from .types import DecompositionTurn, MultiTurnDecompositionPrompt

# The richer decomposition from the user's research diary (papers/20th June
# 2026.pptx, "Proposed Direction 2: Candidate Verification"): instrument ->
# action (shown the valid actions for that instrument) -> target (shown the
# valid targets for that instrument+action) -> final triplet. The "shown
# options" text is filled per-instance at generation/inference time from
# utils.sft_qwen_utils.ontology, not hardcoded here.
PROMPT = MultiTurnDecompositionPrompt(
    name="candidate_verification",
    system=(
        "You are a surgical scene understanding assistant. Return valid JSON only."
    ),
    turns=[
        DecompositionTurn(
            key="instrument",
            user_template=(
                "For each labelled instrument instance in this image, identify the "
                "instrument type. Return JSON only.\n"
                "Instrument instances to label (from the overlay): {instrument_list}"
            ),
        ),
        DecompositionTurn(
            key="action",
            user_template=(
                "For each labelled instrument instance, identify the action being "
                "performed. The valid actions for each instance's instrument are:\n"
                "{options}\n"
                "Return JSON only."
            ),
            show_options=True,
        ),
        DecompositionTurn(
            key="target",
            user_template=(
                "For each labelled instrument instance, identify the target being "
                "acted upon, given the action you just identified. The valid targets "
                "for each instance's instrument and action are:\n"
                "{options}\n"
                "Return JSON only."
            ),
            show_options=True,
        ),
        DecompositionTurn(
            key="final",
            user_template=(
                "For each labelled instrument instance, predict the final "
                "instrument-action-target triplet. Return JSON only."
            ),
        ),
    ],
)
