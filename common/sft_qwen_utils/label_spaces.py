  # Prompt label spaces (authoritative)
INSTRUMENT_LABELS = ["Grasping Forceps", "Monopolar Hook", "Suction-Irrigator", "Clip Applier", "Bipolar Forceps", "Laparoscopic Scissors"]

VERB_LABELS = [
    "grasp", "retract", "dissect", "coagulate", "clip", "cut",
    "aspirate", "irrigate", "pack", "no action"
]

TARGET_LABELS = [
    "gallbladder", "cystic plate", "cystic duct", "cystic artery",
    "cystic pedicle", "blood vessel", "fluid", "abdominal wall cavity",
    "liver", "adhesion", "omentum", "peritoneum",
    "gastrointestinal tract", "specimen bag", "no target"
]

LABELME_INSTRUMENTS_TO_SFT = {
    "grasper":   "Grasping Forceps",
    "hook":      "Monopolar Hook",
    "irrigator": "Suction-Irrigator",
    "clipper":   "Clip Applier",
    "bipolar":   "Bipolar Forceps",
    "scissors":  "Laparoscopic Scissors",
    }

SFT_INSTRUMENT_ALIASES = {
    # Maps a model's occasional shortened-form generation (e.g. "Scissors" instead of the
    # canonical "Laparoscopic Scissors") back to the real SFT instrument name, rather than
    # crashing convert_qwen_to_labelme.py's whole scoring run on one malformed prediction.
    "Irrigator": "Suction-Irrigator",
    "Laparoscopic Suction-Irrigator": "Suction-Irrigator",
    "Scissors": "Laparoscopic Scissors",
    "Hook": "Monopolar Hook",
    "Grasper": "Grasping Forceps",
    "Bipolar": "Bipolar Forceps",
    "Clipper": "Clip Applier",
}

LABELME_VERBS_TO_SFT = {
    "null_verb": "no action",
    "grasp": "grasp",
    "retract": "retract",
    "dissect": "dissect",
    "coagulate": "coagulate",
    "clip": "clip",
    "cut": "cut",
    "aspirate": "aspirate",
    "irrigate": "irrigate",
    "pack": "pack"
}

LABELME_TARGETS_TO_SFT = {
    "cystic_plate": "cystic plate",
    "cystic_duct": "cystic duct",
    "cystic_artery": "cystic artery",
    "cystic_pedicle": "cystic pedicle",
    "blood_vessel": "blood vessel",
    "abdominal_wall_cavity": "abdominal wall cavity",
    "specimen_bag": "specimen bag",
    "gut": "gastrointestinal tract",
    "null_target": "no target",
    "gallbladder": "gallbladder",
    "liver": "liver",
    "fluid": "fluid",
    "adhesion": "adhesion",
    "omentum": "omentum",
    "peritoneum": "peritoneum"            
}

instrument_overlay = {
    "grasper":   {"rgb": (34, 139, 34),  "name": "green"},
    "hook":      {"rgb": (148, 0, 211),  "name": "purple"},
    "irrigator": {"rgb": (0, 0, 255),    "name": "blue"},
    "clipper":   {"rgb": (255, 255, 0),  "name": "yellow"},
    "bipolar":   {"rgb": (0, 204, 204),  "name": "cyan"},
    "scissors":  {"rgb": (255, 215, 0),  "name": "gold"},
}
