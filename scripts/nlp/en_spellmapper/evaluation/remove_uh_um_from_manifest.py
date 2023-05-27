import argparse
import json

from tqdm.auto import tqdm

from nemo.collections.asr.parts.utils.manifest_utils import read_manifest

parser = argparse.ArgumentParser()
parser.add_argument("--input_manifest", required=True, type=str, help="Manifest with transcription before correction")
parser.add_argument("--output_manifest", required=True, type=str, help="Manifest with transcription after correction")

args = parser.parse_args()


test_data = read_manifest(args.input_manifest)

for i in range(len(test_data)):
    test_data[i]["pred_text"] = test_data[i]["pred_text"].replace(" um ", " ").replace(" uh ", " ")
    if test_data[i]["pred_text"].startswith("um ") or test_data[i]["pred_text"].startswith("uh "):
        test_data[i]["pred_text"] = test_data[i]["pred_text"][3:]

with open(args.output_manifest, "w", encoding="utf-8") as out:
    for d in test_data:
        line = json.dumps(d)
        out.write(line + "\n")
