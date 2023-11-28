# This Python file uses the following encoding: utf-8

"""
Input format: canonical \t ipa \t json
соб`ака sɐˈbakə {'word': 'собака', 'pos': 'noun', 'forms': [{'form': 'соб`ака', 'tags': ['canonical']}, {'form': 'соб`аки', 'tags': ['genitive']}...]}
"""


import json
import re

from argparse import ArgumentParser
from collections import Counter

parser = ArgumentParser(description="Get word endings for lemmatization/stemming purposes")
parser.add_argument("--input_name", type=str, required=True, help="Input file")
parser.add_argument("--output_name", type=str, required=True, help="Output file")

args = parser.parse_args()

pairs = Counter()

with open(args.input_name, "r", encoding="utf-8") as inp:
    for line in inp:
        canonical, _, json_data = line.strip().split("\t")
        # skip multiword
        if "_" in canonical:
            continue
        if "-" in canonical:
            continue
        data = json.loads(json_data)
        canonical = canonical.replace("`", "")
        canonical = canonical.replace("ё", "е")
        if not re.match(r"^[абвгдеёжзийклмнопрстуфхцчшщъыьэюя]+$", canonical):
            continue
        if "forms" in data:
            forms = set()
            for item in data["forms"]:
                form = item["form"]
                form = form.replace("`", "")
                form = form.replace("ё", "е")
                if "_" in form:
                    continue
                if "-" in form:
                    continue
                forms.add(form)
        for form in forms:
            for i in range(min(len(form), len(canonical))):
                if form[i] != canonical[i]:
                    # if found different letter, except in first position
                    if i != 0:
                        k = (form[i:], canonical[i:])
                        pairs[k] += 1
                    break

with open(args.output_name, "w", encoding="utf-8") as out:
    for pair, freq in pairs.most_common(100000):
        ending1, ending2 = pair
        out.write(ending1 + "\t" + ending2 + "\t" + str(freq) + "\n")
