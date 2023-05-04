# This Python file uses the following encoding: utf-8

import re

from argparse import ArgumentParser

from utils import clean_russian_text_for_tts

parser = ArgumentParser(description="Extract all unique words in lowercase")
parser.add_argument("--input_name", type=str, required=True, help="Input file")
parser.add_argument("--output_name", type=str, required=True, help="Output file")
parser.add_argument("--split_to_letters", action="store_true", help="Whether to split to letters")

args = parser.parse_args()

all_words = set()

with open(args.input_name, "r", encoding="utf-8") as inp:
    for line in inp:
        text = line.strip()
        words = re.compile('\w+').findall(text)
        for w in words:
            all_words.add(clean_russian_text_for_tts(w))            

with open(args.output_name, "w", encoding="utf-8") as out:
    for w in all_words:
        if args.split_to_letters:
            out.write(" ".join(list(w)) + "\n")
        else:
            out.write(w + "\n")
