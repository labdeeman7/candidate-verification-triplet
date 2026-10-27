from .baseline import PROMPT as BASELINE
from .candidate_verification import PROMPT as CANDIDATE_VERIFICATION

PROMPT_REGISTRY = {
    "baseline": BASELINE,
    "candidate_verification": CANDIDATE_VERIFICATION,
}


def get_prompt(name: str):
    try:
        return PROMPT_REGISTRY[name]
    except KeyError as e:
        raise KeyError(
            f"Unknown prompt '{name}'. Available: {list(PROMPT_REGISTRY.keys())}"
        ) from e
