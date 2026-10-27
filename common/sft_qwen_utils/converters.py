from .label_spaces import (
    VERB_LABELS,
    TARGET_LABELS,
    LABELME_INSTRUMENTS_TO_SFT,
    LABELME_VERBS_TO_SFT,
    LABELME_TARGETS_TO_SFT,
    SFT_INSTRUMENT_ALIASES,
)

def labelme_instrument_to_sft(labelme_instrument: str) -> str:
    try:
        return LABELME_INSTRUMENTS_TO_SFT[labelme_instrument]
    except KeyError:
        raise ValueError(f"Unknown LabelMe instrument: {labelme_instrument}")

def labelme_verb_to_sft(labelme_verb: str) -> str:
    v = LABELME_VERBS_TO_SFT.get(labelme_verb, labelme_verb)
    if v is None: 
        return v
    
    if v not in VERB_LABELS:
        raise ValueError(f"Invalid SFT verb: {labelme_verb} → {v}")
    return v

def labelme_target_to_sft(labelme_target: str) -> str:
    t = LABELME_TARGETS_TO_SFT.get(labelme_target, labelme_target)
    if t is None: 
        return t
    
    if t not in TARGET_LABELS:
        raise ValueError(f"Invalid SFT target: {labelme_target} → {t}")
    return t

def labelme_triplet_to_sft(labelme_triplet: str) -> str:
    labelme_inst, labelme_verb, labelme_target =  labelme_triplet.split(',')
    sft_inst = labelme_instrument_to_sft(labelme_inst)
    sft_verb = labelme_verb_to_sft(labelme_verb)
    sft_target = labelme_target_to_sft(labelme_target)
    
    sft_triplet = f'{sft_inst},{sft_verb},{sft_target}'
    return sft_triplet

def instrument_sft_to_labelme(sft_instrument: str) -> str:
    original_sft_instrument = sft_instrument
    sft_instrument = SFT_INSTRUMENT_ALIASES.get(original_sft_instrument, original_sft_instrument)
    for k, v in LABELME_INSTRUMENTS_TO_SFT.items():
        if v == sft_instrument:
            return k
    raise ValueError(f"Unknown SFT instrument: {original_sft_instrument}")   

def verb_sft_to_labelme(sft_verb: str) -> str:
    for k, v in LABELME_VERBS_TO_SFT.items():
        if v == sft_verb:
            return k 
    print(f"Warning: SFT verb '{sft_verb}' not found in LABELME_VERBS_TO_SFT mapping. Returning 'null_verb'.")
    return "null_verb"

def target_sft_to_labelme(sft_target: str) -> str:
    for k, v in LABELME_TARGETS_TO_SFT.items():
        if v == sft_target:
            return k
    print(f"Warning: SFT target '{sft_target}' not found in LABELME_TARGETS_TO_SFT mapping. Returning 'null_target'.")
    return "null_target"
    
