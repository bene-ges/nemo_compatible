"""
This script can be used to sample phrases from yago_wiki intermediate file. Example:
   grammatica      Grammatica Latina (Leutschoviae, 1717)
   veteris;praecepta       Rhetorices veteris et novae praecepta (Lipsiae, 1717)
   institutiones;germanicae;hungaria;ortu  Institutiones linguac germanicae et slavicae in Hungaria ortu (Leutschoviae, 1718)
"""

from argparse import ArgumentParser
from collections import Counter

parser = ArgumentParser(description="Sample phrases")
parser.add_argument("--input_name", required=True, type=str, help="File with input data")
parser.add_argument("--output_name", required=True, type=str, help="File with output data")
parser.add_argument(
    "--max_count",
    required=True,
    type=int,
    help="Maximum count after which we ignore lines that do not contain any new phrases",
)
args = parser.parse_args()

vocab = Counter()

out = open(args.output_name, "w", encoding="utf-8")

with open(args.input_name, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        phrase_str = parts[0]
        paragraph = " ".join(parts[1:])

        phrases = phrase_str.split(";")
        ok = False
        for phrase in phrases:
            if phrase not in vocab:
                vocab[phrase] = 0
            if vocab[phrase] < args.max_count:
                ok = True
                vocab[phrase] += 1
        if ok:
            out.write(line)
out.close()

print ("total number of unique phrases: ", len(vocab))
