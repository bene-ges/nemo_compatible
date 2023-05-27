from argparse import ArgumentParser

from nemo.collections.nlp.data.spellchecking_asr_customization.utils import (
    get_alignment_by_dp,
    load_ngram_mappings_for_dp,
)

parser = ArgumentParser(description="Get shortest path by n-gram mappings")
parser.add_argument("--ngram_mappings", required=True, type=str, help="Path to ngram mappings file")

args = parser.parse_args()


joint_vocab, orig_vocab, misspelled_vocab, max_len = load_ngram_mappings_for_dp(args.ngram_mappings)

hyp_phrase = "i n a c c e s s i b l e"
ref_phrase = "a c c e s s i b l e"
path = get_alignment_by_dp(ref_phrase, hyp_phrase, dp_data=(joint_vocab, orig_vocab, misspelled_vocab, max_len))
for hyp_ngram, ref_ngram, score, sum_score, joint_freq, orig_freq, misspelled_freq in path:
    print(
        "\t",
        "hyp=",
        hyp_ngram,
        "; ref=",
        ref_ngram,
        "; score=",
        score,
        "; sum_score=",
        sum_score,
        joint_freq,
        orig_freq,
        misspelled_freq,
    )
