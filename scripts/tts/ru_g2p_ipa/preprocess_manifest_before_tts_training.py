import json
import re
import sox

from argparse import ArgumentParser
from random import random

from utils import (
    clean_russian_g2p_trascription,
    clean_russian_text_for_tts,
)


parser = ArgumentParser(description="Prepare input for tts training")
parser.add_argument("--mode", type=str, required=True, help="Mode, one of [before_g2p, after_g2p]")
parser.add_argument("--input_name", type=str, required=True, help="Input file")
parser.add_argument("--g2p_name", type=str, required=True, help="Results of g2p")
parser.add_argument("--vocab_mono_name", type=str, required=True, help="Vocabulary with single (unambiguous) transcription")
parser.add_argument("--vocab_multi_name", type=str, required=True, help="Vocabulary with multiple (ambiguous) transcriptions")
parser.add_argument("--keep_grapheme_ratio", type=float, default=0.0, help="Ratio of cases when graphematic word is kept")
parser.add_argument("--output_name", type=str, required=True, help="Output manifest file")

args = parser.parse_args()

ambiguous = set()
with open(args.vocab_multi_name, "r", encoding="utf-8") as f:
    for line in f:
        inp, _ = line.strip().split("\t")
        ambiguous.add(inp)

mono_vocab = {}
with open(args.vocab_mono_name, "r", encoding="utf-8") as f:
    for line in f:
        inp, transcription = line.strip().split("\t")
        mono_vocab[inp] = transcription

if args.mode == "after_g2p":
    with open(args.g2p_name, "r", encoding="utf-8") as f:
        for line in f:
            try:
                _, inp, transcription, _, _ = line.strip().split("\t")
            except:
                print("cannot read line: " + line)
                continue
            mono_vocab[inp] = transcription

    out = open(args.output_name, "w", encoding="utf-8")

    with open(args.input_name, "r", encoding="utf-8") as inp:
        for line in inp:
            audio_path, text = line.strip().split("|")
            data = {}
            data["orig_text"] = text
            text = clean_russian_text_for_tts(text)
            phonemized_text = ""
            m = re.search(r"[\w\-]+", text)
            while m is not None:
                begin = m.start()
                end = m.end()
                phonemized_text += text[0:begin]
                w = text[begin:end]
                w_lettered = " ".join(list(w))
                if w_lettered in mono_vocab and random() > args.keep_grapheme_ratio:
                    phonemized_text += clean_russian_g2p_trascription(mono_vocab[w_lettered])
                else:
                    phonemized_text += w
                if end >= len(text):
                    break
                text = text[end:]
                m = re.search(r"[\w\-]+", text)
            if end < len(text):
                phonemized_text += text[end:]
            
            data["audio_filepath"] = audio_path
            data["duration"] = sox.file_info.duration(audio_path)
            data["text"] = phonemized_text
            out.write(json.dumps(data, ensure_ascii=False) + "\n")

    out.close()
else:
    unknown_words = set()

    with open(args.input_name, "r", encoding="utf-8") as inp:
        for line in inp:
            audio_path, text = line.strip().split("|")
            text = clean_russian_text_for_tts(text)
            m = re.search(r"[\w\-]+", text)
            while m is not None:
                begin = m.start()
                end = m.end()
                w = " ".join(list(text[begin:end]))
                if w not in mono_vocab and w not in ambiguous:
                    unknown_words.add(w)
                if end >= len(text):
                    break
                text = text[end:]
                m = re.search(r"[\w\-]+", text)

    with open(args.output_name, "w", encoding="utf-8") as out:
        for w in unknown_words:
            out.write(w + "\n")
