import math
from argparse import ArgumentParser
from typing import Set

from nemo.collections.nlp.data.spellchecking_asr_customization.utils import (
    get_index,
    load_ngram_mappings,
)

parser = ArgumentParser(description="Create index for custom phrases, allows to use parameters")
parser.add_argument("--input_file", required=True, type=str, help="Path to input file with custom phrases")
parser.add_argument(
    "--portion_size",
    type=int,
    default=150000,
    help="Random split of phrase set to portions of approximately that number of elements",
)
parser.add_argument("--ngram_mapping", type=str, required=True, help="Path to ngram mapping vocabulary")
parser.add_argument("--output_file", type=str, required=True, help="Output file")
parser.add_argument("--min_log_prob", type=float, default=-4.0, help="Minimum log probability")
parser.add_argument(
    "--max_phrases_per_ngram",
    type=int,
    default=100,
    help="Maximum phrases per ngram, ngrams with too many phrases will be deleted",
)
parser.add_argument(
    "--max_misspelled_freq", type=int, default=10000, help="Ngram mappings with higher misspelled frequency will be skipped"
)

args = parser.parse_args()


vocab, ban_ngram = load_ngram_mappings(args.ngram_mapping, args.max_misspelled_freq)
all_phrases = set()
with open(args.input_file, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        all_phrases.add(" ".join(list(parts[0].casefold().replace(" ", "_"))))

n_portions = math.ceil(len(all_phrases) / args.portion_size)
portion_size = math.ceil(len(all_phrases) / n_portions)

all_phrases = list(all_phrases)
for i in range(n_portions):
    portion = all_phrases[i * portion_size:(i * portion_size) + portion_size]

    phrases, ngram2phrases = get_index(
        portion,
        vocab,
        ban_ngram,
        min_log_prob=args.min_log_prob,
        max_phrases_per_ngram=args.max_phrases_per_ngram,
    )

    with open(args.output_file + "." + str(i), "w", encoding="utf-8") as out:
        for ngram in ngram2phrases:
            for phrase_id, b, size, lp in ngram2phrases[ngram]:
                phr = phrases[phrase_id]
                out.write(ngram + "\t" + phr + "\t" + str(b) + "\t" + str(size) + "\t" + str(lp) + "\n")
