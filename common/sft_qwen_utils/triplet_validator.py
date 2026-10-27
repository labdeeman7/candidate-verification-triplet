from typing import List, Dict, Any, Tuple, Set
from common.tripletseg_dataset_utils.dataset_variables import TripletSegmentationVariables

def _get_label_maps():
    inst_id_to_class = TripletSegmentationVariables.categories["instrument"]
    inst_class_to_id = {cls: int(i) for i, cls in inst_id_to_class.items()}

    trip_id_to_class = TripletSegmentationVariables.categories["triplet"]
    trip_class_to_id = {cls: int(i) for i, cls in trip_id_to_class.items()}

    return inst_class_to_id, trip_class_to_id

def validate_triplets_against_registry(
    predictions: List[Dict[str, Any]],
    trip_class_to_id: Dict[str, int],
    inst_class_to_id: Dict[str, int],
    warn_on_unknown: bool = True
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """
    Validates predictions against the official triplet registry.
    Invalid triplets default to null_verb/null_target.
    
    Returns:
        - corrected predictions
        - set of invalid triplet names (for logging/stats)
    """
    corrected = []
    invalid_triplets = set()
    
    for pred in predictions:
        inst_sft = pred.get("instrument", "")
        verb_sft = pred.get("action", "")
        target_sft = pred.get("target", "")
        
        # Convert to LabelMe naming convention
        from common.sft_qwen_utils.converters import (
            instrument_sft_to_labelme,
            verb_sft_to_labelme,
            target_sft_to_labelme
        )
        
        inst_lm = instrument_sft_to_labelme(inst_sft)
        verb_lm = verb_sft_to_labelme(verb_sft)
        target_lm = target_sft_to_labelme(target_sft)
        
        # Build triplet name exactly as evaluation expects
        triplet_name = f"{inst_lm},{verb_lm},{target_lm}"
        
        # Validate against registry
        if triplet_name not in trip_class_to_id:
            if warn_on_unknown:
                invalid_triplets.add(triplet_name)
            
            # Fallback to null (matching eval behavior)
            verb_lm = "null_verb"
            target_lm = "null_target"
            triplet_name = f"{inst_lm},{verb_lm},{target_lm}"
            
            # Sanity check: null triplet must exist in registry
            if triplet_name not in trip_class_to_id:
                raise ValueError(f"Fallback triplet '{triplet_name}' not in registry!")
        
        # Update prediction with validated values
        pred["instrument"] = inst_sft  # Keep SFT format for consistency
        pred["action"] = verb_sft if (verb_lm != "null_verb") else "no action"
        pred["target"] = target_sft if (target_lm != "null_target") else "no target"
        pred["_validated"] = True  # Marker for debugging
        
        corrected.append(pred)
    
    return corrected, invalid_triplets