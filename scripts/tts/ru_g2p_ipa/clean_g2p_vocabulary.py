from argparse import ArgumentParser

from utils import (
    clean_russian_g2p_trascription,
    clean_russian_text_for_tts,
)

parser = ArgumentParser(description="Clean g2p vocabulary")
parser.add_argument("--input_name", type=str, required=True, help="Input file")
parser.add_argument("--output_name", type=str, required=True, help="Output file")

args = parser.parse_args()

out = open(args.output_name, "w", encoding="utf-8")

with open(args.input_name, "r", encoding="utf-8") as inp:
    for line in inp:
        # Example input: л е д о к       lʲ ɪ d 'o k     lʲ 'ɵ d ə k
        parts = line.strip().split("\t")
        if len(parts) < 2:
            raise ValueError("expect at least two columns: ", line)
        text, transcription = parts[0], parts[1]
        text = text.replace(" ", "")
        text = clean_russian_text_for_tts(text)
        transcription = clean_russian_g2p_trascription(transcription)
        out.write(text + "\t" + transcription + "\n")

out.close()
