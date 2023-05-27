"""This script is used to correct a bug with multiple space, observed in ASR results produced by Conformer-CTC.
Probably connected with this issue: https://github.com/NVIDIA/NeMo/issues/4034.

Since future post-processing relies on words to be separated by single space, we need to correct the manifests.
"""

import argparse
import json

from nemo.collections.asr.parts.utils.manifest_utils import read_manifest

parser = argparse.ArgumentParser()
parser.add_argument("--input_manifest", required=True, type=str, help="Manifest with transcription before correction")
parser.add_argument("--output_manifest", required=True, type=str, help="Manifest with transcription after correction")

args = parser.parse_args()


test_data = read_manifest(args.input_manifest)

for i in range(len(test_data)):
    # if there are multiple spaces in the string they will be merged to one
    test_data[i]["pred_text"] = " ".join(test_data[i]["pred_text"].split())

with open(args.output_manifest, "w", encoding="utf-8") as out:
    for d in test_data:
        line = json.dumps(d)
        out.write(line + "\n")
