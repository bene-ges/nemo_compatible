"""
This script is used to generate a file with examples of false positives.

Output file format:
    in the      eenie;ghinda;hinda ....
    how long     halong; ...

Capitalization file format: lowercase, orig_case, freq
    zsuzsanna       Zsuzsanna       87
    zsuzsanna kossuth       Zsuzsanna Kossuth       1
    kossuth Kossuth 10
    was     was     94822
    was     Was     74
    was     WAS     2
    hungarian       Hungarian       1879
    freedom freedom 132

Idf file format: phrase, idf, number of documents in which phrase occured
    in the  0.32193097471545545     2305627
    was     0.32511695805297663     2298293
    of the  0.3559607516604808      2228487
    ...
    emmanuel episcopal church       13.586499828372018      4
    cornewall       13.586499828372018      4
    george bryan    13.586499828372018      4

Sub misspells file format: original subphrase, misspelled subphrase, joint frequency, frequency of original subphrase, frequency of misspelled subphrase
    domchor dahmer  2       6       6
    domchor dummer  2       6       23
    battery battery 394     395     443
    aaaa battery    a battery       2       2       9
    aaaarrghh       are     2       2       1354
    auto    otto    38      314     2332
    auto    auto    239     314     824
    auto    outo    3       314     37
    auto    atto    16      314     253
    auto    out of  2       314     178
    auto    autto   4       314     18
    aaa auto        a otto  2       2       2
    aaa battery     a battery       2       2       9

Proper names file format: name
    A'isha
    A-jay
    Aa
    Aab
    Aaban
    Aaberg
        
"""

import argparse

from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--output_file", required=True, type=str, help="Output file")
parser.add_argument(
    "--sub_misspells_file", required=True, type=str, help="File with subphrase misspells from YAGO entities"
)
parser.add_argument(
    "--capitalization_file", required=True, type=str, help="File with subphrase capitalization variants"
)
parser.add_argument(
    "--proper_names_file", required=True, type=str, help="File with proper names vocabulary"
)
parser.add_argument("--idf_file", required=True, type=str, help="File with idf of YAGO entities and their subphrases")

args = parser.parse_args()


# these are common words/phrases that for some reason occur sub_misspells but do not occur in yago_entities and thus have idf = +inf
COMMON_PHRASES = {
    "abortion law",
    "acne",
    "act up",
    "action learning",
    "action research",
    "activated",
    "activator",
    "active directory",
    "acyclic",
    "adaptive reuse",
    "addictive",
    "additive",
    "admiration",
    "adulthood",
    "allegedly",
    "army's",
    "assaulted",
    "assert",
    "assuming",
    "atlas",
    "attend",
    "aug",
    "australians",
    "backgrounds",
    "beheaded",
    "chromosomes",
    "cited",
    "cites",
    "classmates",
    "combatant",
    "comics",
    "commandery",
    "considerable",
    "contestant",
    "contracted",
    "cups",
    "currently",
    "dec",
    "depending",
    "devastating",
    "devil",
    "disastrous",
    "documented",
    "entrepreneurship",
    "estranged",
    "evicted",
    "exceeding",
    "exists",
    "feb",
    "falsely",
    "family",
    "fantastic",
    "flags",
    "forthcoming",
    "gameplay",
    "grossing",
    "hardcover",
    "inflicted",
    "informs",
    "inhabit",
    "inn",
    "involve",
    "involved",
    "journal citation reports",
    "jul",
    "jun",
    "kilometres",
    "lab",
    "labs",
    "likewise",
    "listened",
    "listeners",
    "mad",
    "mar",
    "medalist",
    "mentions",
    "minds",
    "mon",
    "mutually",
    "needing",
    "neighbouring",
    "noted",
    "noting",
    "notoriety",
    "nouns",
    "nov",
    "obesity",
    "oblast",
    "obliged",
    "occur",
    "oct",
    "opposes",
    "papa",
    "prevents",
    "proclaimed",
    "promoted",
    "proposes",
    "provided",
    "pseudonym",
    "recipient",
    "recruits",
    "relegation",
    "remix",
    "reparation",
    "replaced",
    "resigned",
    "reuse",
    "revealing",
    "reverted",
    "rewarded",
    "school's",
    "scored",
    "sept",
    "sharing economy",
    "shortly",
    "sounded",
    "spared",
    "synonym",
    "synthesized",
    "tactical",
    "tel",
    "the germans",
    "the mask",
    "thorough",
    "toured",
    "translations",
    "tuesday",
    "ultra",
    "umm",
    "undrafted",
    "updated",
    "upgraded",
    "zombies",
    "vii",
    "visited",
    "wednesday",
    "worlds",
}


if __name__ == "__main__":
    n = 0

    proper_names = set()
    with open(args.proper_names_file, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip().lower()
            proper_names.add(text)

    non_capitalized = set()
    with open(args.capitalization_file, "r", encoding="utf-8") as f:
        keyphrase_sum = {}
        lower_freq = {}
        for line in f:
            keyphrase, orig_phrase, freq = line.strip().split("\t")
            freq = int(freq)
            if keyphrase not in keyphrase_sum:
                keyphrase_sum[keyphrase] = 0
            keyphrase_sum[keyphrase] += freq
            if orig_phrase == keyphrase:
                lower_freq[orig_phrase] = freq
        for keyphrase in lower_freq:
            if keyphrase in proper_names:
                continue
            if lower_freq[keyphrase] / keyphrase_sum[keyphrase] >= 0.01 or keyphrase_sum[keyphrase] > 5000:
                non_capitalized.add(keyphrase)

    idf = {}
    with open(args.idf_file, "r", encoding="utf-8") as f:
        for line in f:
            phrase, score, freq = line.strip().split("\t")
            score = float(score)
            idf[phrase] = score

    for phrase in COMMON_PHRASES:
        idf[phrase] = 4.0

    false_positive_vocab = defaultdict(set)

    with open(args.sub_misspells_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            orig_phrase, misspelled_phrase, joint_freq, _, _  = parts

            orig_words = orig_phrase.split()
            misspelled_words = misspelled_phrase.split()

            # kinlochi => kin lochi    - this can't serve as false positive example
            if orig_phrase.replace("-", "").replace(" ", "") == misspelled_phrase.replace("-", "").replace(" ", ""):
                continue

            if misspelled_phrase not in non_capitalized:
                continue

            max_orig_word_idf = 0.0  # here we store idf of rarest word in the phrase
            for w in orig_words:
                if w in idf:
                    if idf[w] > max_orig_word_idf:
                        max_orig_word_idf = idf[w]
                else:
                    max_orig_word_idf = 100.0

            max_misspelled_word_idf = 0.0  # here we store idf of rarest word in the phrase
            for w in misspelled_words:
                if w in idf:
                    if idf[w] > max_misspelled_word_idf:
                        max_misspelled_word_idf = idf[w]
                else:
                    max_misspelled_word_idf = 100.0

            if len(orig_phrase) < 5:
                continue
            if len(misspelled_phrase) < 4:
                continue
            if max_orig_word_idf - max_misspelled_word_idf < 2.0:
                continue

            false_positive_vocab[misspelled_phrase].add(orig_phrase)
            ## out.write(orig_phrase + "\t" + misspelled_phrase + "\t" + str(max_orig_word_idf) + "\t" + str(max_misspelled_word_idf) + "\n")

    with open(args.output_file, "w", encoding="utf-8") as out:
        for misspelled_phrase in false_positive_vocab:
            out.write(misspelled_phrase + "\t" + ";".join(list(false_positive_vocab[misspelled_phrase])) + "\n")

    
