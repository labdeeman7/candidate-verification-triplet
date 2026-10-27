"""The authoritative 100-class triplet vocabulary, shared by training and eval.

Built from the same `TripletSegmentationVariables.categories["triplet"]` registry
used throughout the rest of this project for AP evaluation, so the ResNet
classifier's output space is identical to the registry Candidate Verification
and Direct SFT are implicitly constrained to -- a fair, apples-to-apples
comparison, not a different (easier or harder) label space.
"""

from common.tripletseg_dataset_utils.dataset_variables import TripletSegmentationVariables

# Sorted for a deterministic, reproducible class-index assignment independent of
# dict insertion order (not that OrderedDict's order would change, but this is
# cheap insurance).
TRIPLET_STRINGS = sorted(TripletSegmentationVariables.categories["triplet"].values())
NUM_CLASSES = len(TRIPLET_STRINGS)  # 100

TRIPLET_TO_INDEX = {t: i for i, t in enumerate(TRIPLET_STRINGS)}
INDEX_TO_TRIPLET = {i: t for i, t in enumerate(TRIPLET_STRINGS)}


def triplet_to_index(instrument: str, verb: str, target: str) -> int:
    """Raw LabelMe-format (underscored) instrument/verb/target -> class index.

    Raises KeyError if the combination isn't one of the 100 registered triplets
    (shouldn't happen for real GT data, but surfaces loudly if it does rather
    than silently mis-training).
    """
    key = f"{instrument},{verb},{target}"
    return TRIPLET_TO_INDEX[key]


def index_to_triplet(index: int) -> tuple[str, str, str]:
    """Class index -> (instrument, verb, target), raw LabelMe-format."""
    instrument, verb, target = INDEX_TO_TRIPLET[index].split(",")
    return instrument, verb, target


# Interaction-only (verb, target) vocabulary, instrument dropped. Used by the
# "VT-only" ResNet variant, which assumes the instrument is supplied by Stage 1
# (as the rest of the project's Stage-2 methods do -- see
# convert_qwen_to_labelme.py, which always writes the instrument field from
# Stage 1's own grounding, never from a Stage-2 re-prediction) rather than
# re-predicting it from the crop itself.
VT_STRINGS = sorted({",".join(t.split(",")[1:]) for t in TRIPLET_STRINGS})
NUM_VT_CLASSES = len(VT_STRINGS)  # 56

VT_TO_INDEX = {vt: i for i, vt in enumerate(VT_STRINGS)}
INDEX_TO_VT = {i: vt for i, vt in enumerate(VT_STRINGS)}


def vt_to_index(verb: str, target: str) -> int:
    """Raw LabelMe-format verb/target -> VT-class index (instrument dropped)."""
    key = f"{verb},{target}"
    return VT_TO_INDEX[key]


def index_to_vt(index: int) -> tuple[str, str]:
    """VT-class index -> (verb, target), raw LabelMe-format."""
    verb, target = INDEX_TO_VT[index].split(",")
    return verb, target
