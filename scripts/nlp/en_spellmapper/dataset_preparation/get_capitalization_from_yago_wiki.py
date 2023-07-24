"""
This script is used to collect capitalization variants with frequencies for all original and misspelled phrases from sub_misspells_file.
"""

import argparse
import os
import re
import tarfile
from collections import defaultdict
from typing import Set

from utils import (
    get_paragraphs_from_json,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_folder",
    required=True,
    type=str,
    help="Input folder with tar.gz files each containing wikipedia articles in json format",
)
parser.add_argument("--output_file", required=True, type=str, help="Output file")
parser.add_argument(
    "--sub_misspells_file", required=True, type=str, help="File with subphrase misspells from YAGO entities"
)
args = parser.parse_args()


def get_capitalization(input_folder: str, key_phrases: Set[str]):
    """
    Args:
        input_folder: Input folder with tar.gz files each containing wikipedia articles in json format
        key_phrases: Set of lowercase phrases that we want to find in texts

    Returns:
        capitalized_vocab: a dictionary where the key=(lowercase phrase),
          value is another dict with key=(phrase in original case), value is frequency
    """
    exclude_titles = set()
    n_documents = 0
    capitalized_vocab = defaultdict(dict)
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
            for p, p_clean in get_paragraphs_from_json(text, exclude_titles, skip_if_len_mismatch=True):
                p_clean_spaced = " " + p_clean + " "
                words = p_clean.split()
                for begin in range(len(words)):
                    for end in range(begin + 1, min(begin + 5, len(words) + 1)):
                        phrase = " ".join(words[begin:end])
                        if phrase in key_phrases:
                            pattern = " " + phrase + " "
                            matches = list(re.finditer(pattern, p_clean_spaced))
                            for m in matches:
                                begin = m.start()
                                end = m.end() - 2
                                phrase_orig = p[begin:end]
                                if phrase not in capitalized_vocab or phrase_orig not in capitalized_vocab[phrase]:
                                    capitalized_vocab[phrase][phrase_orig] = 0    
                                capitalized_vocab[phrase][phrase_orig] += 1
    return capitalized_vocab


if __name__ == "__main__":
    n = 0

    key_phrases = set()
    with open(args.sub_misspells_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            orig_phrase, misspelled_phrase, _, _, _  = parts
            # against rare multiple space occurrence
            key_phrases.add(" ".join(orig_phrase.split()))
            key_phrases.add(" ".join(misspelled_phrase.split()))
    
    capitalized_vocab = get_capitalization(args.input_folder, key_phrases)
    with open(args.output_file, "w", encoding="utf-8") as out:
        for key_phrase in capitalized_vocab:
            for orig_phrase in capitalized_vocab[key_phrase]:
                out.write(
                    key_phrase + "\t" + orig_phrase + "\t" + str(capitalized_vocab[key_phrase][orig_phrase]) + "\n"
                )


