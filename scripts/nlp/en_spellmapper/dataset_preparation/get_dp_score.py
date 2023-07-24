from argparse import ArgumentParser
from collections import Counter

from nemo.collections.nlp.data.spellchecking_asr_customization.utils import (
    get_alignment_by_dp,
    load_ngram_mappings_for_dp,
)

parser = ArgumentParser(
    description="Get dynamic programming scores for all phrase pairs"
)
parser.add_argument("--ngram_mappings", type=str, required=True, help="File with ngram mappings")
parser.add_argument("--input_file", required=True, type=str, help="Path to input file with phrases")
parser.add_argument("--output_file", type=str, required=True, help="Output file")

args = parser.parse_args()

dp_data = load_ngram_mappings_for_dp(args.ngram_mappings)


out = open(args.output_file, "w", encoding="utf-8")
with open(args.input_file, "r", encoding="utf-8") as f:
    for line in f:
        s = line.strip()
        phrase, misspelled_phrase, joint_freq, _, _ = s.split("\t")
        if misspelled_phrase != misspelled_phrase.strip():  # bug fix, found a phrase with space at the end
            continue
        if phrase == misspelled_phrase:
            out.write(s + "\t0.0\n")
            continue

        phrase_spaced = " ".join(list(phrase.replace(" ", "_")))
        misspelled_spaced = " ".join(list(misspelled_phrase.replace(" ", "_")))
        path = get_alignment_by_dp(phrase_spaced, misspelled_spaced, dp_data)
        # path[-1][3] is the sum of logprobs for best path of dynamic programming: divide sum_score by length
        dp_score = path[-1][3] / (len(phrase))
        dp_score2 = path[-1][3] / (len(misspelled_phrase))
        out.write(s + "\t" + str(dp_score) + "\t" + str(dp_score2) + "\n")
        #for hyp_ngram, ref_ngram, score, sum_score, joint_freq, orig_freq, misspelled_freq in path:
        #    out.write(
        #        "\t" +
        #        "hyp=" +
        #        hyp_ngram +
        #        "; ref=" +
        #        ref_ngram +
        #        "; score=" +
        #        str(score) +
        #        "; sum_score=" +
        #        str(sum_score) +
        #        "\t" + str(joint_freq) + "\n")


out.close()

