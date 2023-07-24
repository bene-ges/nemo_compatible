"""
This script is used to calculate idf for words and short phrases from Yago Wikipedia articles.

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

Output file format: phrase, idf, number of documents in which phrase occured
    in the  0.32193097471545545     2305627
    was     0.32511695805297663     2298293
    of the  0.3559607516604808      2228487
    ...
    emmanuel episcopal church       13.586499828372018      4
    cornewall       13.586499828372018      4
    george bryan    13.586499828372018      4

"""

import argparse
import math
import os
import tarfile
from collections import defaultdict
from typing import Set

from utils import (
    get_paragraphs_from_json,
    load_yago_entities,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_folder",
    required=True,
    type=str,
    help="Input folder with tar.gz files each containing wikipedia articles in json format",
)
parser.add_argument(
    "--sub_misspells_file", required=True, type=str, help="File with subphrase misspells from YAGO entities"
)
parser.add_argument("--output_file", required=True, type=str, help="Output file")
args = parser.parse_args()


def get_idf(input_folder: str, key_phrases: Set[str]):
    """
    Args:
        input_folder: Input folder with tar.gz files each containing wikipedia articles in json format
        key_phrases: Set of phrases that we want to find in texts

    Returns:
        idf: a dictionary where the key is a phrase, value is its inverse document frequency
    """
    exclude_titles = set()
    n_documents = 0
    idf = defaultdict(int)
    for name in os.listdir(input_folder):
        print(name)
        tar = tarfile.open(os.path.join(input_folder, name), "r:gz")
        for member in tar.getmembers():
            f = tar.extractfile(member)
            if f is None:
                continue
            byte = f.read()
            text = byte.decode("utf-8")
            n_documents += 1
            phrases = set()
            for p, p_clean in get_paragraphs_from_json(text, exclude_titles, skip_if_len_mismatch=False):
                words = p_clean.split()
                for begin in range(len(words)):
                    for end in range(begin + 1, min(begin + 5, len(words) + 1)):
                        phrase = " ".join(words[begin:end])
                        if phrase in key_phrases:
                            phrases.add(phrase)
            for phrase in phrases:
                idf[phrase] += 1  # one count per document
        # delete too rare phrases (no need to store their idf)
        for phrase in list(idf.keys()):
            if idf[phrase] < 4:
                del idf[phrase]

    return idf, n_documents


if __name__ == "__main__":
    key_phrases = set()
    with open(args.sub_misspells_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            orig_phrase, misspelled_phrase, _, _, _  = parts
            # against rare multiple space occurrence
            key_phrases.add(" ".join(orig_phrase.split()))
            key_phrases.add(" ".join(misspelled_phrase.split()))

    idf, n_documents = get_idf(args.input_folder, key_phrases)

    with open(args.output_file, "w", encoding="utf-8") as out:
        for phrase, freq in sorted(idf.items(), key=lambda item: item[1], reverse=True):
            score = math.log(n_documents / freq)
            out.write(phrase + "\t" + str(score) + "\t" + str(freq) + "\n")
