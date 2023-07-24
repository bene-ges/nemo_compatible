"""
This script is used to select those records from keys2paragraph that contain occurrences of true positives for false positive pairs.

Input file format(keys2paragraph):  
    mcnab;mcnab bank building;eufaula \t The McNab Bank Building is a historic building in Eufaula, Alabama, U.S.. It was built in the 1850s for John McNab, a Scottish-born banker.

Output file format: same as input

False positives file format:
    bronze and      bronzen;bronson
        
"""

import argparse

from collections import defaultdict

from utils import (
    CHARS_TO_IGNORE_REGEX,
    preprocess_apostrophes_space_diacritics,
)

parser = argparse.ArgumentParser()
parser.add_argument("--output_file", required=True, type=str, help="Output file")
parser.add_argument(
    "--false_positives_file", required=True, type=str, help="File with false positives groups"
)
parser.add_argument(
    "--input_file", required=True, type=str, help="File with input in keys2paragraph format"
)

args = parser.parse_args()

if __name__ == "__main__":
    n = 0

    common_phrases = defaultdict(int)  # value=number of paragraphs, containing this common phrase
    custom_phrases = defaultdict(int)  # value=number of paragraphs, containing this custom phrase
    with open(args.false_positives_file, "r", encoding="utf-8") as f:
        for line in f:
            ngram, keys_str = line.strip().split("\t")
            cur_keys = keys_str.split(";")
            for k in cur_keys:
                custom_phrases[k] = 0
            common_phrases[ngram] = 0

    out = open(args.output_file, "w", encoding="utf-8")
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                keys_str, paragraph = line.strip().split("\t")
            except:
                print("bad line format: ", line)
                continue
            cur_keys = keys_str.split(";")
            cur_keys_marked = []
            ok = False
            for k in cur_keys:
                if k in custom_phrases:
                    custom_phrases[k] += 1
                    cur_keys_marked.append("*" + k)
                    ok = True
                else:
                    cur_keys_marked.append(k)

            # search common phrases
            p = preprocess_apostrophes_space_diacritics(paragraph)
            p_clean = CHARS_TO_IGNORE_REGEX.sub(" ", p).lower()  # number of characters is the same in p and p_clean
            p_clean = " ".join(p_clean.split())
            words = p_clean.split()
            ok = False
            common_list = set()
            for begin in range(len(words)):
                for end in range(begin + 1, min(begin + 5, len(words) + 1)):
                    ngram = " ".join(words[begin:end])
                    if ngram in common_phrases:
                        ok = True
                        common_list.add(ngram)
                        common_phrases[ngram] += 1
            if ok:
                out.write(";".join(list(common_list)) + "\t" + paragraph + "\n")


    out.close()

    with open(args.output_file + ".custom_freqs", "w", encoding="utf-8") as out:
        for k, freq in sorted(custom_phrases.items(), key=lambda item: item[1], reverse=True):
            out.write(k + "\t" + str(freq) + "\n")

    with open(args.output_file + ".common_freqs", "w", encoding="utf-8") as out:
        for k, freq in sorted(common_phrases.items(), key=lambda item: item[1], reverse=True):
            out.write(k + "\t" + str(freq) + "\n")
