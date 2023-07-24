"""This script is used to transcribe nemo manifest with Whisper.
"""

import argparse
import json
import torch
import whisper  

from nemo.collections.asr.parts.utils.manifest_utils import read_manifest

parser = argparse.ArgumentParser()
parser.add_argument("--whisper_model", required=True, type=str, help="Whisper model, e.g. base, base.en, large")
parser.add_argument("--input_manifest", required=True, type=str, help="Input manifest")
parser.add_argument("--output_manifest", required=True, type=str, help="Output manifest")

args = parser.parse_args()

if torch.cuda.is_available():
    device="cuda"
else:
    device="cpu"

model = whisper.load_model(args.whisper_model, device=device)

test_data = read_manifest(args.input_manifest)

for i in range(len(test_data)):
    result = model.transcribe(test_data[i]["audio_filepath"], language="en")  
    test_data[i]["pred_text"] = result["text"]  

with open(args.output_manifest, "w", encoding="utf-8") as out:
    for d in test_data:
        line = json.dumps(d)
        out.write(line + "\n")
