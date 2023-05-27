from argparse import ArgumentParser

from utils import read_custom_phrases

from nemo.collections.nlp.data.spellchecking_asr_customization.utils import (
    get_index,
    load_ngram_mappings,
)

parser = ArgumentParser(description="Create index for custom phrases, allows to use parameters")
parser.add_argument("--input_file", required=True, type=str, help="Path to input file with custom phrases")
parser.add_argument(
    "--input_max_lines",
    type=int,
    default=-1,
    help="If set to a number > 0, only that many lines will be read from input file",
)
parser.add_argument(
    "--input_portion_size",
    type=int,
    default=-1,
    help="If set to a number > 0, leads to input file split to portions of that number of lines",
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
for idx, custom_phrases in enumerate(
    read_custom_phrases(args.input_file, max_lines=args.input_max_lines, portion_size=args.input_portion_size)
):
    print(idx, "len(custom_phrases)", len(custom_phrases))
    phrases, ngram2phrases = get_index(
        custom_phrases,
        vocab,
        ban_ngram,
        min_log_prob=args.min_log_prob,
        max_phrases_per_ngram=args.max_phrases_per_ngram,
    )
    print("len(phrases)=", len(phrases), "; len(ngram2phrases)=", len(ngram2phrases))

    with open(args.output_file + "." + str(idx), "w", encoding="utf-8") as out:
        for ngram in ngram2phrases:
            for phrase_id, b, size, lp in ngram2phrases[ngram]:
                phr = phrases[phrase_id]
                out.write(ngram + "\t" + phr + "\t" + str(b) + "\t" + str(size) + "\t" + str(lp) + "\n")
