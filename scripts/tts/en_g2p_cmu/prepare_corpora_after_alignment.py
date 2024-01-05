import glob
import os
import re
from argparse import ArgumentParser
from collections import Counter
from typing import Dict, Optional, TextIO, Tuple


parser = ArgumentParser(description="Produce data for English G2P model")
parser.add_argument(
    "--mode",
    required=True,
    type=str,
    help='Mode, one of ["get_replacement_vocab", "filter_by_vocab", "get_labeled_corpus"]',
)
parser.add_argument(
    "--alignment_filename", required=True, type=str, help='Name of alignment file, like "itn.out", "itn.out.vocab2000"'
)
parser.add_argument("--out_filename", required=True, type=str, help='Output file')
parser.add_argument("--vocab_filename", required=True, type=str, help='Vocab name')
args = parser.parse_args()


def process_line(line: str) -> Optional[Tuple[str, str, str, int]]:
    """A helper function to read the file with alignment results"""

    parts = line.strip().split("\t")
    if len(parts) != 4:
        return None
    if parts[0] != "good:":
        return None

    src, dst, align = parts[1], parts[2], parts[3]
    
    return src, dst, align


def get_replacement_vocab() -> None:
    """Loops through the file with alignment results, counts frequencies of different replacement segments.
    """

    full_vocab = Counter()
    with open(args.alignment_filename, "r", encoding="utf-8") as f:
        for line in f:
            t = process_line(line)
            if t is None:
                continue
            src, dst, replacement = t
            inputs = src.split(" ")
            replacements = replacement.split(" ")
            if len(inputs) != len(replacements):
                raise ValueError("Length mismatch in: " + line)
            for inp, rep in zip(inputs, replacements):
                full_vocab[rep] += 1

    with open(args.vocab_filename, "w", encoding="utf-8") as out:
        for k, v in full_vocab.most_common(1000000000):
            out.write(k + "\t" + str(v) + "\n")


def filter_by_vocab() -> None:
    """Given a restricted vocabulary of replacements,
    loops through the file with alignment results,
    discards the examples containing a replacement which is not in our restricted vocabulary.
    """

    if not os.path.exists(args.vocab_filename):
        raise ValueError(f"Alignments dir {args.giza_dir} does not exist")
    # load vocab from file
    vocab = {}
    with open(args.vocab_filename, "r", encoding="utf-8") as f:
        for line in f:
            k, v = line.strip().split("\t")
            vocab[k] = int(v)
    print("len(vocab)=", len(vocab))
    out = open(args.out_filename, "w", encoding="utf-8")
    with open(args.alignment_filename, "r", encoding="utf-8") as f:
        for line in f:
            t = process_line(line)
            if t is None:
                continue
            src, dst, replacement = t
            ok = True
            for s, r in zip(src.split(" "), replacement.split(" ")):
                if s != r and r not in vocab:
                    ok = False
            if ok:
                out.write(src + "\t" + dst + "\t" + replacement + "\n")
    out.close()


def main() -> None:
    if args.mode == "get_replacement_vocab":
        get_replacement_vocab()
    elif args.mode == "filter_by_vocab":
        filter_by_vocab()
    else:
        raise ValueError("unknown mode: " + args.mode)


if __name__ == "__main__":
    main()
