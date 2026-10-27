from dataclasses import dataclass

@dataclass
class Prompt:
    name: str
    system: str
    user: str


@dataclass(frozen=True)
class DecompositionTurn:
    """One turn in a multi-turn triplet-decomposition dialogue.

    `key` identifies which field(s) of the per-instance prediction this
    turn's assistant response supplies: "instrument", "action", "target", or
    "final" (the complete triplet). `user_template` is the turn's user
    instruction text; when `show_options` is True the template must contain
    an `{options}` placeholder, filled per-instance at generation time from
    the ontology's valid label combinations given the fields already
    confirmed by earlier turns (see `utils.sft_qwen_utils.ontology`).
    """

    key: str
    user_template: str
    show_options: bool = False


@dataclass
class MultiTurnDecompositionPrompt:
    """Prompt text for decomposing triplet prediction across an ordered list of turns."""

    name: str
    system: str
    turns: list[DecompositionTurn]


@dataclass
class PairwiseComparisonPrompt:
    """Prompt text for the Compare-and-Rank auxiliary task: asking whether one
    instance's action/target is option_a or option_b (both drawn from a mined
    confusion pair -- see `utils.sft_qwen_utils.confusion_pairs`). `user` must
    contain `{instance_id}`, `{category}`, `{option_a}`, `{option_b}` placeholders.
    """

    name: str
    system: str
    user: str
