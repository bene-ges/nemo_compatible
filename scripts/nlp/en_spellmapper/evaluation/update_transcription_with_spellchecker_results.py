"""This script is used to gather SpellMapper results from a folder.
Since the original sentences can be too long to fit into SpellMapper input, they are splitted to fragments of 10-15 words.
There can be multiple intersecting fragments, they all have their own 10 candidates and serve as separate inputs to SpellMapper inference.
This script collects the resulting replacements from all fragments and (after filtering) substitutes them back to the original sentence.

Arguments:
  --asr_hypotheses_folder - contains files that store information about what fragment corresponds to what sentence:
       short_sentence \t full_sentence 
    Example
       you see the scratchage will come        and because of this what you see a pre buying and where you see the scratchage will come
  --spellchecker_results_folder - contains files with SpellMapper results
  --input_manifest - contains json lines like any asr-manifest in NeMo
    {"text": "and because of this what do you see are prebuying and where you see the scrappage will come", \
        "audio_filepath": "/home/common/FBMF/studfbmf28/data/kensho_2/spgispeech/val/0018ad922e541b415ae60e175160b976/21.wav", \
            "doc_id": "0018ad922e541b415ae60e175160b976", \
                "pred_text": "and because of this what you see a pre buying and where you see the scratchage will come"}
  --output_manifest - will be a similar manifest, but "pred_text" will contain the corrected transcript, and old transcript will be stored in "pred_text_before_correction"
  --min_replacement_len - allows to control minimum length of replacement
  --min_prob" - allows to control minimum probability of replacement
"""

import argparse
import json
import os
from collections import defaultdict

from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from nemo.collections.nlp.data.spellchecking_asr_customization.utils import (
    read_spellmapper_predictions,
    apply_replacements_to_text,
    load_ngram_mappings_for_dp,
)

parser = argparse.ArgumentParser()
parser.add_argument("--asr_hypotheses_folder", required=True, type=str, help="Input folder with asr hypotheses")
parser.add_argument(
    "--spellchecker_results_folder", required=True, type=str, help="Input folder with spellchecker output"
)
parser.add_argument("--input_manifest", required=True, type=str, help="Manifest with transcription before correction")
parser.add_argument("--output_manifest", required=True, type=str, help="Manifest with transcription after correction")
parser.add_argument("--min_replacement_len", default=1, type=int, help="Minimum replacement length")
parser.add_argument("--min_prob", default=0.5, type=int, help="Minimum replacement probability")
parser.add_argument(
    "--min_dp_score_per_symbol",
    required=True,
    type=float,
    help="Minimum dynamic programming sum score averaged by hypothesis length",
)
parser.add_argument("--ngram_mappings", type=str, required=True, help="File with ngram mappings, needed for dynamic programming")


args = parser.parse_args()

joint_vocab, orig_vocab, misspelled_vocab, max_len = load_ngram_mappings_for_dp(args.ngram_mappings)

final_corrections = defaultdict(str)
banned_count = 0
for name in os.listdir(args.spellchecker_results_folder):
    doc_id, _ = name.split(".")
    short2full_sent = defaultdict(list)
    full_sent2corrections = defaultdict(dict)
    try:
        with open(args.asr_hypotheses_folder + "/" + doc_id + ".txt", "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                short_sent, full_sent = s.split("\t")
                short2full_sent[short_sent].append(full_sent)
                full_sent2corrections[full_sent] = []
        print("len(short2full_sent)=", len(short2full_sent))
    except:
        continue

    spellmapper_results = read_spellmapper_predictions(args.spellchecker_results_folder + "/" + doc_id + ".txt")
    for text, replacements, _ in spellmapper_results:
        short_sent = text
        if short_sent not in short2full_sent:
            continue
        for full_sent in short2full_sent[short_sent]:  # it can happen that one short sentence occurred in multiple full sentences
            offset = full_sent.find(short_sent)
            for begin, end, candidate, prob in replacements:
                if len(candidate) < args.min_replacement_len:
                    continue
                full_sent2corrections[full_sent].append((begin + offset, end + offset, candidate, prob))

    for full_sent in full_sent2corrections:
        corrected_full_sent = apply_replacements_to_text(full_sent, full_sent2corrections[full_sent], min_prob=args.min_prob, replace_hyphen_to_space=True, dp_data=(joint_vocab, orig_vocab, misspelled_vocab, max_len), min_dp_score_per_symbol=args.min_dp_score_per_symbol)
        final_corrections[doc_id + "\t" + full_sent] = corrected_full_sent


print("len(final_corrections)=", len(final_corrections))

test_data = read_manifest(args.input_manifest)

# extract just the text corpus from the manifest
pred_text = [data['pred_text'] for data in test_data]
audio_filepath = [data['audio_filepath'] for data in test_data]
doc_ids = []

for data in test_data:
    if "doc_id" in data:
        doc_ids.append(data["doc_id"])
    else:  # fix for Spoken Wikipedia format
        path = data["audio_filepath"]
        # example of path: ...clips/197_0000.wav   #doc_id=197
        path_parts = path.split("/")
        path_parts2 = path_parts[-1].split("_")
        doc_id = path_parts2[-2]
        doc_ids.append(doc_id)

for i in range(len(test_data)):
    sent, doc_id = pred_text[i], doc_ids[i]
    k = doc_id + "\t" + sent
    if k in final_corrections:
        test_data[i]["pred_text_before_correction"] = test_data[i]["pred_text"]
        test_data[i]["pred_text"] = final_corrections[k]

with open(args.output_manifest, "w", encoding="utf-8") as out:
    for d in test_data:
        line = json.dumps(d)
        out.write(line + "\n")

print("final_corrections=", len(final_corrections))
