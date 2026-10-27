from .types import Prompt

PROMPT = Prompt(
    name="baseline",
    system="""Return ONLY valid JSON. Output an object with a single key "predictions", containing a list of items. Each item must have keys: id, instrument, action, target. Use only the allowed labels.
If idle: action="no action", target="no target". No text outside JSON. """,
    user="""You are analysing a laparoscopic cholecystectomy frame.
The image contains multiple surgical instrument instances predicted by a neural network.
Each instance is marked in the image with a name and ID (e.g. grasper_1, hook_2).
For EACH instance ID, assign exactly one (instrument, action, target) triplet using visual evidence.
Allowed labels:
Instruments = ['Grasping Forceps','Bipolar Forceps','Monopolar Hook','Laparoscopic Scissors','Clip Applier','Suction-Irrigator']
Actions = ['grasp','retract','dissect','coagulate','clip','cut','aspirate','irrigate','pack','no action']
Targets = ['gallbladder','cystic plate','cystic duct','cystic artery','cystic pedicle','blood vessel','fluid',
           'abdominal wall cavity','liver','adhesion','omentum','peritoneum',
           'gastrointestinal tract','specimen bag','no target']           
Required output format (example):
{{
  "predictions": [
    {{"id": "grasper_1", "instrument": "Grasping Forceps", "action": "retract", "target": "gallbladder"}},
    {{"id": "hook_2", "instrument": "Monopolar Hook", "action": "dissect", "target": "cystic plate"}}
  ]
}}
Instrument Instances to label (from the overlay): {instrument_list}
Now produce the JSON for this image."""
)
