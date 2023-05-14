import json
import sox

from argparse import ArgumentParser

parser = ArgumentParser(description="Create manifest for tts training")
parser.add_argument("--orig_text_name", type=str, required=True, help="Input file with original text")
parser.add_argument("--preprocessed_text_name", type=str, required=True, help="Input file with preprocessed text")
parser.add_argument("--audio_paths_name", type=str, required=True, help="Input file with audio paths")
parser.add_argument("--output_name", type=str, required=True, help="Output manifest file")

args = parser.parse_args()

with open(args.orig_text_name, "r", encoding="utf-8") as f:
    orig_texts = [line.strip() for line in f.readlines()]

with open(args.preprocessed_text_name, "r", encoding="utf-8") as f:
    texts = [line.strip() for line in f.readlines()]

with open(args.audio_paths_name, "r", encoding="utf-8") as f:
    audio_paths = [line.strip() for line in f.readlines()]

if len(texts) != len(audio_paths) or len(texts) != len(orig_texts):
    raise IndexError("length mismatch: len(texts)=" + str(len(texts)) + "; len(audio_paths)=" + str(audio_paths) + "; len(orig_texts)="  + str(orig_texts))

out = open(args.output_name, "w", encoding="utf-8")

for i in range(len(texts)):
    text = texts[i]
    orig_text = orig_texts[i]
    audio_path = audio_paths[i]
    data = {}
    data["text"] = text
    data["orig_text"] = orig_text
    data["audio_filepath"] = audio_path
    data["duration"] = sox.file_info.duration(audio_path)
    out.write(json.dumps(data, ensure_ascii=False) + "\n")

out.close()
