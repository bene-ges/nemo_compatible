import re

from argparse import ArgumentParser
from random import random

from utils import (
    clean_russian_g2p_trascription,
    clean_russian_text_for_tts,
)


parser = ArgumentParser(description="Prepare input for Russian tts inference")
parser.add_argument("--input_name", type=str, required=True, help="Input file with plain text")
parser.add_argument("--g2p_name", type=str, required=True, help="Results of g2p, each unique word on a separate line")
parser.add_argument("--heteronyms_name", type=str, required=True, help="Text file with words that can have different transcriptions, they will be kept as graphemes")
parser.add_argument("--g2p_correct_name", type=str, required=True, help="Text file with words and transcriptions that override g2p output")
parser.add_argument("--keep_grapheme_ratio", type=float, default=0.0, help="Ratio of cases when graphematic word is kept")
parser.add_argument("--output_name", type=str, required=True, help="Output file with input to tts")

args = parser.parse_args()

heteronyms = set()
with open(args.heteronyms_name, "r", encoding="utf-8") as f:
    for line in f:
        inp = line.strip()
        heteronyms.add(inp)

g2p_vocab = {}

with open(args.g2p_name, "r", encoding="utf-8") as f:
    for line in f:
        try:
            # Example input: b lʲ 'ʉ xʲ ɪ r  б л ю х е р     b lʲ 'ʉ xʲ ɪ r  b lʲ 'ʉ xʲ ɪ r  PLAIN PLAIN PLAIN PLAIN PLAIN PLAIN
            _, inp, transcription, _, _ = line.strip().split("\t")
        except:
            print("cannot read line: " + line)
            continue
        inp = inp.replace(" ", "")
        g2p_vocab[inp] = clean_russian_g2p_trascription(transcription)

with open(args.g2p_correct_name, "r", encoding="utf-8") as f:
    for line in f:
        try:
            # Example input: ледок   lʲɪd`ok
            inp, transcription = line.strip().split("\t")
        except:
            print("cannot read line: " + line)
            continue
        g2p_vocab[inp] = transcription

out = open(args.output_name, "w", encoding="utf-8")

with open(args.input_name, "r", encoding="utf-8") as inp:
    for line in inp:
        text = line.strip()
        text = clean_russian_text_for_tts(text)
        phonemized_text = ""
        m = re.search(r"[\w\-]+", text)
        while m is not None:
            begin = m.start()
            end = m.end()
            phonemized_text += text[0:begin]
            w = text[begin:end]
            if w in heteronyms:
                phonemized_text += w
            elif w in g2p_vocab and random() >= args.keep_grapheme_ratio:
                phonemized_text += clean_russian_g2p_trascription(g2p_vocab[w])
            else:  # shouldn't go here as all words are expected to pass through g2p
                phonemized_text += w

            if end >= len(text):
                break
            text = text[end:]
            end = 0
            m = re.search(r"[\w\-]+", text)
        if end < len(text):
            phonemized_text += text[end:]
        
        out.write(phonemized_text + "\n")

out.close()
