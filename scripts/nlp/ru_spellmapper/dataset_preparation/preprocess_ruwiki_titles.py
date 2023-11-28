# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
This script can be used to preprocess ruwiki titles.
## Before running this script, download and unpack
  https://dumps.wikimedia.org/ruwiki/20230301/ruwiki-20230301-all-titles.gz
And run
  awk 'BEGIN {FS="\t"} ($1 == "0"){print $2}($1 == "1"){print $2}' < ruwiki-20230301-all-titles | sort -u > ruwiki_titles.uniq
to get an input file for this script.

The input file looks like this:
    Беккет,_Сэмюэль
    Беккетт
    Беккет_(фильм)
    Беккет_(фильм,_2021)
    Беккет,_Эдвард,_5-й_барон_Гримторп
    
The output file has two columns and looks like this:
    Беккет,_Сэмюэль    беккет_сэмюэль
    Беккетт            беккетт
    Беккет_(фильм)     беккет_фильм
    Беккет,_Эдвард,_5-й_барон_Гримторп  беккет_эдвард
    
"""

import re
from argparse import ArgumentParser
from collections import defaultdict

from nemo.collections.nlp.data.spellchecking_asr_customization.utils import replace_diacritics

parser = ArgumentParser(description="Clean Ruwiki titles")
parser.add_argument("--input_name", type=str, required=True, help="Input file")
parser.add_argument("--output_name", type=str, required=True, help="Output file")

args = parser.parse_args()

out = open(args.output_name, "w", encoding="utf-8")

name2id = defaultdict(int)

with open(args.input_name, "r", encoding="utf-8") as inp:
    for line in inp:
        s = line.strip()
        if "/Архив/" in s:
            continue
        if re.match(r".*\d", s):
            continue
        if not re.match(r".*\w", s):  #no letters
            continue

        key = s
        s = s.casefold()  # lowercase
        s = re.sub(r"\(.+\)", r"", s)  # delete brackets and what is inside it
        s = s.replace("&", " and ")
        s = s.replace("!", "")
        s = s.replace("?", "")
        s = s.replace("\"", "")
        s = s.replace("»", "")
        s = s.replace("«", "")
        s = s.replace("’", "'")
        s = s.replace("ʻ", "'")
        s = s.replace("_", " ")
        s = s.replace("/", ",")
        s = s.replace(":", ",")
        s = replace_diacritics(s)
        if len(set(list(s)) - set(list(", -'abcdefghijklmnopqrstuvwxyzабвгдежзийклмнопрстуфхцчшщъыьэюя"))) == 0:
            s = s.replace(",", " ")
            s = " ".join(s.split())
            s = s.replace(" ", "_")
            # this is needed to distinguish between different keys mapping to same clean phrases
            if s not in name2id:
                name2id[s] = 1
            else:
                name2id[s] += 1
            out.write(key + "\t" + s + "__" + str(name2id[s]) + "\n")
        else:        
            parts = s.split(",")
            for p in parts:
                sp = p.strip()
                if len(sp) < 3:
                    continue
                if "." in sp:
                    continue
                if re.match(r".*\d", sp):
                    continue
                sp = "_".join(sp.split())
                if len(set(list(sp)) - set(list(" _-'abcdefghijklmnopqrstuvwxyzабвгдежзийклмнопрстуфхцчшщъыьэюя"))) == 0:
                    # this is needed to distinguish between different keys mapping to same clean phrases
                    if sp not in name2id:
                        name2id[sp] = 1
                    else:
                        name2id[sp] += 1
                    out.write(key + "\t" + sp + "__" + str(name2id[sp]) + "\n")
                else:
                    print(sp, str(set(list(sp)) - set(list(" _-'abcdefghijklmnopqrstuvwxyzабвгдеёжзийклмнопрстуфхцчшщъыьэюя"))))

out.close()
